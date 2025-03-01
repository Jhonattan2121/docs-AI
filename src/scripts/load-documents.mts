import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import path, { dirname } from 'path';
import { fileURLToPath } from 'url';
import { openaiClient } from '../config/openai.js';
import { supabaseClient } from '../config/supabase.js';
import { MarkdownLoader } from '../utils/document-loaders/markdown-loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function checkExistingDocument(url: string, content: string) {
  const { data, error } = await supabaseClient
    .from('documents')
    .select('content, url')
    .eq('url', url)
    .single();

  if (error) return null;
  return { exists: true, hasChanged: data.content !== content };
}

async function loadDocuments() {
  try {
    const docsPath = path.resolve(__dirname, '../../src/content/docs');
    const docs = await MarkdownLoader.loadFromDirectory(docsPath);

    console.log(`📚 Found ${docs.length} documents`);
    let totalProcessed = 0;
    let totalSkipped = 0;
    let totalUpdated = 0;

    const textSplitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
      chunkSize: 1200,
      chunkOverlap: 125,
    });

    for (const doc of docs) {
      console.log(`\n📄 Processing: ${doc.url}`);

      try {
        const chunks = await textSplitter.splitText(doc.content);
        console.log(`✂️  Generated ${chunks.length} chunks`);

        for (let i = 0; i < chunks.length; i++) {
          const chunk = chunks[i];
          const chunkUrl = `${doc.url}#chunk-${i + 1}`;

          // Check if the chunk already exists and if there have been any changes
          const existing = await checkExistingDocument(chunkUrl, chunk);
          
          if (existing?.exists && !existing.hasChanged) {
            console.log(`⏭️  Jumping chunk ${i + 1}/${chunks.length} (no changes)`);
            totalSkipped++;
            continue;
          }

          if (existing?.exists && existing.hasChanged) {
            console.log(`🔄 Updating chunk ${i + 1}/${chunks.length} (changed content)`);
            totalUpdated++;
          } else {
            console.log(`✨ Processing new chunk ${i + 1}/${chunks.length}`);
          }

          // Generate embedding
          const response = await openaiClient.embeddings.create({
            model: 'text-embedding-ada-002',
            input: chunk,
          });

          // Insert or update in Supabase
          const { error } = existing?.exists 
            ? await supabaseClient
                .from('documents')
                .update({
                  content: chunk,
                  embedding: response.data[0].embedding,
                  updated_at: new Date().toISOString()
                })
                .eq('url', chunkUrl)
            : await supabaseClient
                .from('documents')
                .insert({
                  content: chunk,
                  url: chunkUrl,
                  embedding: response.data[0].embedding,
                  chunk_index: i,
                  total_chunks: chunks.length,
                  created_at: new Date().toISOString()
                });

          if (error) {
            throw new Error(`Error when ${existing?.exists ? 'to update' : 'insert'} chunk: ${error.message}`);
          }

          totalProcessed++;
          console.log(`✅ Chunk ${i + 1}/${chunks.length} ${existing?.exists ? 'updated' : 'inserted'} successfully`);

         // Delay to avoid rate limiting
          await new Promise(resolve => setTimeout(resolve, 200));
        }
      } catch (error) {
        console.error(`❌ Error processing document ${doc.url}:`, error);
      }
    }

    console.log('\n📊 Processing Summary:');
    console.log(`✅ Processed: ${totalProcessed}`);
    console.log(`⏭️  Jumped: ${totalSkipped}`);
    console.log(`🔄 Updated: ${totalUpdated}`);

  } catch (error) {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  }
}

// Run the script
loadDocuments();