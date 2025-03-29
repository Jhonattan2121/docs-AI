import type { APIRoute } from 'astro';
import { openaiClient } from '../../config/openai';
import { supabaseClient } from '../../config/supabase';
import { SYSTEM_PROMPT } from '../../prompts/emplates';
import { Cache } from '../../utils/cache';
import { Logger } from '../../utils/callbacks/logger';
import { MarkdownLoader } from '../../utils/document-loaders/markdown-loader';
import { TextSplitter } from '../../utils/document-transformers/text-splitter';
import { OpenAIEmbeddings } from '../../utils/embeddings/openai-embeddings';

export const prerender = false;
export const output = 'server';

export class RAGService {
  private cache: Cache;
  private logger: Logger;
  private loader: MarkdownLoader;
  private splitter: TextSplitter;
  private embeddings: OpenAIEmbeddings;

  constructor() {
    this.cache = new Cache();
    this.logger = new Logger();
    this.loader = new MarkdownLoader();
    this.splitter = new TextSplitter();
    this.embeddings = new OpenAIEmbeddings(openaiClient);
  }

  async processQuery(query: string) {
    try {
      const cacheKey = `rag_query_${query}`;
      const cached = this.cache.get<{ response: string, sources: string[] }>(cacheKey);
      if (cached) {
        return cached;
      }

      // Find relevant documents
      const docs = await this.findRelevantDocs(query);
      let gradedDocs = [];
      
      if (docs.length > 0) {
        gradedDocs = await this.gradeDocs(docs, query);
        console.log(`\n🔍 Found ${docs.length} documents, ${gradedDocs.length} relevant in Supabase`);
      } else {
        console.log('❌ No relevant documents found in Supabase');
      }

      // Create prompt with relevant documents
      const systemPrompt = this.createPrompt(
        SYSTEM_PROMPT,
        [],
        query,
        gradedDocs
      );

      // Generate response
      const response = await this.generateResponse(systemPrompt, query);

      // Prepare result with sources
      const result = {
        response,
        sources: gradedDocs.map(doc => doc.url)
      };

      // Save to cache
      this.cache.set(cacheKey, result, 3600); // Cache for 1 hour

      return result;
    } catch (error) {
      this.logger.error((error as Error).message);
      throw error;
    }
  }

  private async findRelevantDocs(query: string) {
    // Cache embeddings
    const cacheKey = `embedding_${query}`;
    const cachedEmbedding = this.cache.get<number[]>(cacheKey);

    if (cachedEmbedding) {
      return this.vectorSearch(cachedEmbedding);
    }

    const embedding = await this.embeddings.createEmbedding(query);
    this.cache.set(cacheKey, embedding, 3600);

    return this.vectorSearch(embedding);
  }

  private async gradeDocs(docs: any[], query: string) {
    // Limit maximum number of documents for processing
    const MAX_DOCS = 5;

    // Sort docs by relevance before processing
    const gradingPromises = docs.slice(0, MAX_DOCS).map(async (doc) => {
      const gradedResponse = await this.docsGrader({
        document: doc.content.substring(0, 1000), 
        question: query
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
    query: string,
    docs: any[]
  ): string {
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
      .replace("{history}", "")
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

  private async generateResponse(prompt: string, query: string): Promise<string> {
    const completion = await openaiClient.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [
        { role: 'system', content: prompt },
        { role: 'user', content: query }
      ],
      temperature: 0.1,
      max_tokens: 500
    });

    return completion.choices[0].message.content ?? '';
  }
}

// API Route handler
export const POST: APIRoute = async ({ request }) => {
  const service = new RAGService();

  try {
    const { query } = await request.json();

    if (!query) {
      return new Response(
        JSON.stringify({ error: "Query parameter is required" }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const result = await service.processQuery(query);

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