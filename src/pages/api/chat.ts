import type { APIRoute } from 'astro';
import { openaiClient } from '../../config/openai';
import { supabaseClient } from '../../config/supabase';
import { Cache } from '../../utils/cache';
import { Logger } from '../../utils/callbacks/logger';
import { MarkdownLoader } from '../../utils/document-loaders/markdown-loader';
import { TextSplitter } from '../../utils/document-transformers/text-splitter';
import { OpenAIEmbeddings } from '../../utils/embeddings/openai-embeddings';
import { ConversationMemory } from '../../utils/memory/conversation-memory';

import { BASIC_CHAT_PROMPT, SYSTEM_PROMPT } from '../../prompts/emplates';


export const prerender = false;
export const output = 'server';

export class ChatService {
  private cache: Cache;
  private logger: Logger;
  private loader: MarkdownLoader;
  private splitter: TextSplitter;
  private embeddings: OpenAIEmbeddings;
  private memory: ConversationMemory;

  constructor() {
    this.cache = new Cache();
    this.logger = new Logger();
    this.loader = new MarkdownLoader();
    this.splitter = new TextSplitter();
    this.embeddings = new OpenAIEmbeddings(openaiClient);
    this.memory = new ConversationMemory(supabaseClient);
  }

  private async shouldSearchDocs(message: string): Promise<boolean> {
    const response = await openaiClient.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [
        {
          role: 'system',
          content: 'Analyze whether the message requires consulting the documentation. Return only true or false. Examples:\n"hi" -> false\n"how to install?" -> true\n"how are you?" -> false\n"what configuration is required?" -> true'
        },
        { role: 'user', content: message }
      ],
      temperature: 0.1,
      max_tokens: 5
    });

    return response.choices[0].message.content?.toLowerCase().includes('true') ?? false;
  }

  async processMessage(message: string, userId: string) {

    try {
      debugger
      const cacheKey = `chat_${userId}_${message}`;
      const cached = this.cache.get<{ response: string, sources: string[] }>(cacheKey);
      if (cached) {
        return cached;
      }

      const history = await this.memory.getHistory(userId);
      const needsDocs = await this.shouldSearchDocs(message);
      let gradedDocs = [];

      if (needsDocs) {
        const docs = await this.findRelevantDocs(message);
        if (docs.length > 0) {
          gradedDocs = await this.gradeDocs(docs, message);
          console.log(`\n🔍 Found ${docs.length} documents in Supabase`);
        } else {
          console.log('❌ No relevant documents found in Supabase');
        }
      }

      const systemPrompt = this.createPrompt(
        needsDocs ? SYSTEM_PROMPT : BASIC_CHAT_PROMPT,
        history,
        message,
        gradedDocs
      ); console.warn("SYSTEM_PROMPT")
      console.warn(systemPrompt)

      const response = await this.generateResponse(systemPrompt, message);

      console.log("\n\nresponse")
      console.log(response)

      await this.memory.saveInteraction(userId, message, response);

      const result = {
        response,
        // sources: splitDocs.map(doc => doc.url)
      };

      // Save to cache
      this.cache.set(cacheKey, result, 3600); // Cache for 1 hour

      return result;
    } catch (error) {
      this.logger.error((error as Error).message);
      throw error;
    }
  }

  private async findRelevantDocs(message: string) {
    // Cache embeddings
    const cacheKey = `embedding_${message}`;
    const cachedEmbedding = this.cache.get<number[]>(cacheKey);

    if (cachedEmbedding) {
      return this.vectorSearch(cachedEmbedding);
    }

    const embedding = await this.embeddings.createEmbedding(message);
    this.cache.set(cacheKey, embedding, 3600);

    return this.vectorSearch(embedding);
  }

  private async gradeDocs(docs: any[], message: string) {
   // Limit maximum number of documents for processing
    const MAX_DOCS = 5;

    // Sort docs by relevance before processing
    const gradingPromises = docs.slice(0, MAX_DOCS).map(async (doc) => {
      const gradedResponse = await this.docsGrader({
        document: doc.content.substring(0, 1000), 
        question: message
      });

      try {
        const parsedResponse = JSON.parse(gradedResponse.content as string);
        // Increase confidence threshold
        return (parsedResponse.relevant && parsedResponse.confidence > 0.5) ? {
          ...doc,
          confidence: parsedResponse.confidence
        } : null;
      } catch (error) {
        this.logger.error(`Error parsing grader response: ${error}`);
        return null;
      }
    });

    const gradedDocs = await Promise.all(gradingPromises);
    // Sort by confidence and get only the most relevant ones
    const goodDocuments = gradedDocs
      .filter(Boolean)
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);

    console.log(`\n----- ANALYSIS OF RELEVANT DOCUMENTS ------`);
    console.log(`Total documents found: ${docs.length}`);
    console.log(`Documents considered relevant: ${goodDocuments.length}`);

    return goodDocuments;
  }

  private async vectorSearch(embedding: number[]) {
    const { data, error } = await supabaseClient.rpc(
      'match_documents_vector',
      {
        query_embedding: embedding,
        match_threshold: 0.2,
        match_count: 15
      }
    );

    if (error) throw error;
    return data || [];
  }

  private createPrompt(
    systemPrompt: string,
    history: any[],
    message: string,
    docs: any[]
  ): string {
    // Get only the last 3 interactions from the history
    const recentHistory = history.slice(-3);
    const historyText = recentHistory
      .map(item => `Usuário: ${item.user}\nMensagem: ${item.message}`)
      .join('\n\n');

    // Format relevant documents in a more structured way
    const docsText = docs
      .map(doc => `
       Source: ${doc.url}
        Confidence: ${doc.confidence}
        Content: ${doc.content}
        ---
      `).join('\n');

    return systemPrompt
      .replace("{total_docs}", docs.length.toString())
      .replace("{relevant_docs}", docs.length.toString())
      .replace("{history}", historyText)
      .replace("{documentation}", docsText);
  }

  private async docsGrader({ document, question }: { document: string, question: string }) {
    const gradingPrompt = `
        Analyze whether the document is relevant to the question.
      Return a JSON with the following format:
      {
      "relevant": boolean,
      "reason": string,
      "confidence": number // 0 to 1
      }
      Consider it relevant if:
      - The document directly mentions the topic
      - The document contains related information that can help answer
      - The document has examples or explanations about the topic
      - Even if it does not mention exactly the same terms, but the context is relevant
        Question: ${question}
      Document: ${document}
    `;

    const response = await openaiClient.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [
        { role: 'system', content: gradingPrompt },
      ],
      temperature: 0.3,
      max_tokens: 500
    });

    return {
      content: response.choices[0].message.content
    }
  }


  private async generateResponse(prompt: string, message: string): Promise<string> {
    const completion = await openaiClient.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [
        { role: 'system', content: prompt },
        { role: 'user', content: message }
      ],
      temperature: 0.1,
      max_tokens: 500
    });

    return completion.choices[0].message.content ?? '';
  }
}

// API Route handler
export const POST: APIRoute = async ({ request }) => {
  const service = new ChatService();

  try {
    const { message, user_id } = await request.json();

    const result = await service.processMessage(message, user_id);

    return new Response(JSON.stringify(result), {
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    return new Response(
      JSON.stringify({ error: (error as Error).message }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
};