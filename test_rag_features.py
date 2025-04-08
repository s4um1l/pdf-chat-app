import asyncio
import os
from aicoolspace.text_utils import TextFileLoader, CharacterTextSplitter, SentenceTextSplitter, PDFFileLoader
from aicoolspace.vectordatabase import VectorDatabase, euclidean_distance, cosine_similarity
from aicoolspace.openai_utils.chatmodel import ChatOpenAI
from aicoolspace.openai_utils.prompts import SystemRolePrompt, UserRolePrompt

# Make sure OpenAI API Key is set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

async def main():
    print("=== RAG with Enhanced Features Demo ===")
    
    # 1. Test different chunking strategies
    print("\n=== Testing Different Chunking Strategies ===")
    text_loader = TextFileLoader("data/PMarcaBlogs.txt")
    documents = text_loader.load_documents()
    
    char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    sentence_splitter = SentenceTextSplitter(max_sentences_per_chunk=10, sentence_overlap=2)
    
    char_chunks = char_splitter.split_texts(documents)
    sentence_chunks = sentence_splitter.split_texts(documents)
    
    print(f"Character-based chunks: {len(char_chunks)}")
    print(f"Sentence-based chunks: {len(sentence_chunks)}")
    
    print("\nSample Character Chunk:")
    print(char_chunks[10][:200] + "...")
    
    print("\nSample Sentence Chunk:")
    print(sentence_chunks[10][:200] + "...")

    # 1b. Test PDF loader if a PDF file exists 
    try:
        pdf_path = "./data/bert.pdf"
        if os.path.exists(pdf_path):
            print("\n=== Testing PDF File Loading ===")
            pdf_loader = PDFFileLoader(pdf_path)
            pdf_docs = pdf_loader.load_documents()
            
            # Process PDF documents similar to text documents
            pdf_chunks = sentence_splitter.split_texts(pdf_docs)
            print(f"Number of PDF chunks: {len(pdf_chunks)}")
            
            if pdf_chunks:
                print("\nSample PDF Chunk:")
                print(pdf_chunks[0][:200] + "..." if len(pdf_chunks[0]) > 200 else pdf_chunks[0])
        else:
            print("\nSkipping PDF test - no sample.pdf found in data directory")
    except (ImportError, Exception) as e:
        print(f"\nPDF loading failed: {str(e)}")

    # 2. Test different distance metrics
    print("\n=== Testing Different Distance Metrics ===")
    
    # Add metadata for the examples
    metadata_list = []
    for i, chunk in enumerate(sentence_chunks):
        # Simple metadata: categorize by chunk number and add some categories
        metadata = {
            "chunk_id": i,
            "category": "startup" if "startup" in chunk.lower() else 
                       ("career" if "career" in chunk.lower() else "other")
        }
        metadata_list.append(metadata)
    
    # Build VectorDB with metadata
    vector_db = VectorDatabase()
    vector_db = await vector_db.abuild_from_list(sentence_chunks, metadata_list)
    
    query = "What advice does Marc have about startups and venture capital?"
    
    print(f"\nQuery: {query}")
    print("\nTop 3 results using Cosine Similarity:")
    cosine_results = vector_db.search_by_text(query, k=3, distance_measure=cosine_similarity)
    for i, (text, score) in enumerate(cosine_results):
        print(f"{i+1}. Score: {score:.4f}, Text: {text[:100]}...")
    
    print("\nTop 3 results using Euclidean Distance:")
    euclidean_results = vector_db.search_by_text(query, k=3, distance_measure=euclidean_distance)
    for i, (text, score) in enumerate(euclidean_results):
        print(f"{i+1}. Score: {score:.4f}, Text: {text[:100]}...")

    # 3. Test metadata filtering
    print("\n=== Testing Metadata Filtering ===")
    
    # Filter by category
    category_filter = {"category": "startup"}
    filtered_results = vector_db.search_by_text(
        query, 
        k=3, 
        metadata_filter=category_filter
    )
    
    print("\nResults filtered by category 'startup':")
    for i, (text, score) in enumerate(filtered_results):
        metadata = vector_db.metadata.get(text, {})
        print(f"{i+1}. Score: {score:.4f}, Category: {metadata.get('category')}, Text: {text[:100]}...")
    
    # 4. Simple RAG Example with our enhancements
    print("\n=== RAG With Enhanced Features ===")
    
    # Create a simple RAG prompt
    RAG_PROMPT_TEMPLATE = """Use the provided context to answer the user's query.
If you do not know the answer, please respond with "I don't know".
"""
    
    USER_PROMPT_TEMPLATE = """Context:
{context}

User Query:
{user_query}
"""
    
    rag_prompt = SystemRolePrompt(RAG_PROMPT_TEMPLATE)
    user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE)
    
    # Get a response using RAG
    chat_openai = ChatOpenAI()
    
    user_query = "What is the Michael Eisner Memorial Weak Executive Problem?"
    # Get results using metadata filter and euclidean distance
    context_list = vector_db.search_by_text(
        user_query, 
        k=3, 
        distance_measure=euclidean_distance,
        metadata_filter=None  # Can add filters like {"category": "startup"}
    )
    
    context_prompt = ""
    for context in context_list:
        context_prompt += context[0] + "\n"
    
    formatted_system_prompt = rag_prompt.create_message()
    formatted_user_prompt = user_prompt.create_message(user_query=user_query, context=context_prompt)
    
    response = chat_openai.run([formatted_system_prompt, formatted_user_prompt])
    
    print(f"\nQuery: {user_query}")
    print(f"\nRAG Response:\n{response}")

if __name__ == "__main__":
    asyncio.run(main()) 