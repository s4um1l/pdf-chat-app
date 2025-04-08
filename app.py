import os
import asyncio
import tempfile
from typing import List, Dict, Tuple, Any, Optional

import chainlit as cl
from chainlit.prompt import Prompt, PromptMessage
from chainlit.playground.providers import ChatOpenAI

from openai import AsyncOpenAI

# Import aicoolspace utilities
from aicoolspace.text_utils import (
    TextFileLoader, 
    CharacterTextSplitter, 
    SentenceTextSplitter, 
    PDFFileLoader
)
from aicoolspace.vectordatabase import (
    VectorDatabase, 
    euclidean_distance, 
    cosine_similarity
)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify OpenAI API Key is set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# RAG Prompt Templates
RAG_SYSTEM_TEMPLATE = """Use the provided context to answer the user's query.
If you do not know the answer, please respond with "I don't know".
"""

RAG_USER_TEMPLATE = """Context:
{context}

User Query:
{user_query}
"""

# Dropdown options
CHUNKING_OPTIONS = ["Character-based", "Sentence-based"]
DISTANCE_OPTIONS = ["Cosine Similarity", "Euclidean Distance"]

# Global variables to store state
uploaded_files = {}
vector_dbs = {}
current_chunking = "Sentence-based"
current_distance = "Cosine Similarity"
current_filter_category = None

# Define these REQUIRED callbacks early to ensure they're registered
@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with welcome message and UI elements"""
    # Set OpenAI settings
    settings = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 1000,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    cl.user_session.set("settings", settings)
    cl.user_session.set("chunking_strategy", "Sentence-based")
    cl.user_session.set("distance_metric", "Cosine Similarity")
    cl.user_session.set("category_filter", None)
    cl.user_session.set("ready_for_query", False)
    
    # Send welcome messages
    welcome_msg = """# Welcome to Enhanced RAG Demo
    
Upload a text (.txt) or PDF (.pdf) file to get started with Retrieval-Augmented Generation.

This app demonstrates several key RAG features:
- **Document Processing** with different chunking strategies
- **Vector Search** with different distance metrics 
- **Metadata Filtering** for more targeted results

## Quick Setup After Uploading:
Use the all-in-one setup command:
`/setup char cos p` - Sets character-based chunking, cosine similarity, and processes immediately

Or use individual commands:
- `/c` - Use character-based chunking
- `/s` - Use sentence-based chunking  
- `/cos` - Use cosine similarity
- `/e` - Use euclidean distance
- `/p` - Process the file
- `/help` - Show all commands
"""
    
    await cl.Message(content=welcome_msg).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages/queries"""
    content = message.content.strip()
    
    # Check for file uploads in message
    if hasattr(message, 'elements') and message.elements and len(message.elements) > 0:
        files = []
        for element in message.elements:
            if hasattr(element, 'type') and element.type == 'file':
                files.append(element)
            elif isinstance(element, dict) and element.get('type') == 'file':
                files.append(element)
        
        if files:
            await handle_file_upload(files)
            return
    
    # Handle commands
    if content.startswith("/"):
        # Quick setup command
        if content.lower().startswith("/setup"):
            await handle_setup_command(content)
            return
            
        # Single command processing
        command = content[1:].lower().strip()
        
        # Chunking strategy commands
        if command in ["character-based", "char", "c"]:
            cl.user_session.set("chunking_strategy", "Character-based")
            await cl.Message(content="✅ Chunking strategy set to: Character-based").send()
            return
        elif command in ["sentence-based", "sent", "s"]:
            cl.user_session.set("chunking_strategy", "Sentence-based")
            await cl.Message(content="✅ Chunking strategy set to: Sentence-based").send()
            return
            
        # Distance metric commands
        elif command in ["cosine", "cos"]:
            cl.user_session.set("distance_metric", "Cosine Similarity")
            await cl.Message(content="✅ Distance metric set to: Cosine Similarity").send()
            return
        elif command in ["euclidean", "euc", "e"]:
            cl.user_session.set("distance_metric", "Euclidean Distance")
            await cl.Message(content="✅ Distance metric set to: Euclidean Distance").send()
            return
            
        # NEW: Sentence embedding command
        elif command in ["embed", "embedding"]:
            example_embedding = "Example sentence embedding vector: [0.021, -0.113, 0.027, 0.094, -0.222, ...]"
            embedding_info = f"""## Sentence Embedding Information
            
This demo uses OpenAI's embedding model to convert text chunks into vectors.

**Current embedding model:** text-embedding-3-small
**Vector dimensions:** 1536
**Embedding process:**
1. Text is divided into chunks
2. Each chunk is converted to a vector using the embedding model
3. Query is converted to a vector using the same model
4. Similarity is calculated between query vector and chunk vectors

{example_embedding}

To search with embedded vectors, just process your document normally and ask questions."""
            await cl.Message(content=embedding_info).send()
            return
            
        # Process file command
        elif command in ["process", "p"]:
            current_file = cl.user_session.get("current_file")
            if current_file and current_file in uploaded_files:
                await process_file(current_file)
            else:
                await cl.Message(content="⚠️ No file selected. Please upload a document first.").send()
            return
            
        # Filter commands
        elif command == "clear-filter":
            await set_category_filter(None)
            return
        elif command.startswith("filter "):
            category = command[7:].strip()
            await set_category_filter(category)
            return
            
        # Help command
        elif command == "help":
            help_text = """### Available Commands:
- `/character-based` or `/c` - Use character-based chunking
- `/sentence-based` or `/s` - Use sentence-based chunking
- `/cosine` or `/cos` - Use cosine similarity distance
- `/euclidean` or `/e` - Use euclidean distance
- `/process` or `/p` - Process the uploaded file
- `/filter category` - Filter by category
- `/clear-filter` - Clear category filter
- `/embed` - Show information about sentence embeddings
- `/help` - Show this help message

### Quick Setup:
- `/setup char cos p` - Set Character-based chunking with Cosine similarity and process
- `/setup sent euc p` - Set Sentence-based chunking with Euclidean distance and process
"""
            await cl.Message(content=help_text).send()
            return
            
        # Status command
        elif command == "status":
            chunking = cl.user_session.get("chunking_strategy", "Not set")
            distance = cl.user_session.get("distance_metric", "Not set")
            category = cl.user_session.get("category_filter", "None")
            current_file = cl.user_session.get("current_file", "No file uploaded")
            
            status_text = f"""### Current Settings:
- **Chunking Strategy:** {chunking}
- **Distance Metric:** {distance}
- **Category Filter:** {category}
- **Current File:** {current_file}
"""
            await cl.Message(content=status_text).send()
            return
            
        # Unknown command
        else:
            await cl.Message(content=f"⚠️ Unknown command: {command}\nType `/help` to see available commands.").send()
            return
    
    # Normal RAG queries
    await process_query(content)

# Helper functions below

async def handle_file_upload(files):
    """Process uploaded files from message attachments"""
    if not files:
        return None
    
    file = files[0]  # Get the first file
    file_name = getattr(file, 'name', 'uploaded_file')
    file_ext = os.path.splitext(file_name)[1].lower()
    
    if file_ext not in ['.txt', '.pdf']:
        await cl.Message(
            content=f"⚠️ Unsupported file type: {file_ext}. Please upload a .txt or .pdf file."
        ).send()
        return None
    
    # Store file information
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    
    # Get file content - adapted for different Chainlit versions
    try:
        # Try to get file content using content attribute
        if hasattr(file, 'content'):
            file_content = file.content
        # Try to get using path attribute
        elif hasattr(file, 'path'):
            with open(file.path, "rb") as f_in:
                file_content = f_in.read()
        # Try to get using get_content method
        elif hasattr(file, 'get_content'):
            file_content = await file.get_content()
        else:
            await cl.Message(content="❌ Could not read file content. Unsupported file object.").send()
            return None
            
        # Save content to temp file
        with open(temp_file.name, "wb") as f_out:
            f_out.write(file_content)
    except Exception as e:
        await cl.Message(content=f"❌ Error processing file: {str(e)}").send()
        return None
    
    uploaded_files[file_name] = {
        "path": temp_file.name,
        "type": file_ext
    }
    
    cl.user_session.set("current_file", file_name)
    
    # Notify user with a success message and offer one-click setup
    success_msg = f"""✅ **{file_name}** uploaded successfully!

**Quick Setup Options:**
- **Copy-paste these commands:**
  ```
  /setup char cos p
  ```
  ```
  /setup sent cos p
  ```
  
Or set up manually:
1. Choose chunking: `/c` (character) or `/s` (sentence)
2. Choose similarity: `/cos` (cosine) or `/e` (euclidean)
3. Process file: `/p`

**Want to try sentence embeddings?**
Copy this command:
```
/embed
```"""
    
    await cl.Message(content=success_msg).send()
    
    # Automatically suggest processing with default settings
    await cl.Message(content="Would you like to process with the recommended settings? Type `/setup sent cos p` to process with sentence-based chunking and cosine similarity.").send()
    
    return file_name

async def set_category_filter(category: str = None):
    """Set category filter for search"""
    cl.user_session.set("category_filter", category)
    if category:
        await cl.Message(content=f"✅ Category filter set to: **{category}**").send()
    else:
        await cl.Message(content="✅ Category filter cleared").send()

async def handle_setup_command(content):
    """Handle the setup command which can configure multiple settings at once"""
    parts = content.lower().split()
    
    # Defaults
    chunking_set = False
    distance_set = False
    process_requested = False
    
    # Remove /setup
    parts.pop(0)
    
    for part in parts:
        # Handle chunking strategy
        if part in ["character-based", "char", "c"]:
            cl.user_session.set("chunking_strategy", "Character-based")
            chunking_set = True
        
        elif part in ["sentence-based", "sent", "s"]:
            cl.user_session.set("chunking_strategy", "Sentence-based")
            chunking_set = True
        
        # Handle distance metric
        elif part in ["cosine", "cos"]:
            cl.user_session.set("distance_metric", "Cosine Similarity")
            distance_set = True
        
        elif part in ["euclidean", "euc", "e"]:
            cl.user_session.set("distance_metric", "Euclidean Distance")
            distance_set = True
            
        # Handle process command
        elif part in ["process", "p"]:
            process_requested = True
    
    # Prepare response
    response = []
    if chunking_set:
        chunking = cl.user_session.get("chunking_strategy")
        response.append(f"✅ Chunking strategy set to: {chunking}")
        
    if distance_set:
        distance = cl.user_session.get("distance_metric")
        response.append(f"✅ Distance metric set to: {distance}")
    
    # Send response for settings
    if response:
        await cl.Message(content="\n".join(response)).send()
    
    # Process file if requested
    if process_requested:
        current_file = cl.user_session.get("current_file")
        if current_file and current_file in uploaded_files:
            await process_file(current_file)
        else:
            await cl.Message(content="⚠️ No file selected. Please upload a document first.").send()

    # If nothing was set, show help
    if not (chunking_set or distance_set or process_requested):
        await cl.Message(content="""### Setup Command Usage:
- `/setup char cos` - Set Character-based chunking with Cosine similarity
- `/setup sent euc` - Set Sentence-based chunking with Euclidean distance
- `/setup char cos p` - Set options and process immediately
        
Type `/help` for all available commands.""").send()

async def process_file(filename: str):
    """Process the uploaded file with the selected chunking strategy"""
    file_info = uploaded_files[filename]
    chunking_strategy = cl.user_session.get("chunking_strategy", "Sentence-based")
    
    # Status message with progress indicator
    process_msg = cl.Message(content=f"⏳ Processing document with **{chunking_strategy}** chunking...")
    await process_msg.send()
    
    try:
        # Load documents based on file type
        if file_info["type"] == ".txt":
            loader = TextFileLoader(file_info["path"])
        else:  # .pdf
            loader = PDFFileLoader(file_info["path"])
        
        documents = loader.load_documents()
        
        # Split documents based on selected chunking strategy
        if chunking_strategy == "Character-based":
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        else:  # Sentence-based
            splitter = SentenceTextSplitter(max_sentences_per_chunk=10, sentence_overlap=2)
        
        chunks = splitter.split_texts(documents)
        
        # Add metadata for each chunk
        metadata_list = []
        categories = set()
        for i, chunk in enumerate(chunks):
            # Simple metadata: categorize by chunk number and content keywords
            category = "other"
            if "startup" in chunk.lower():
                category = "startup"
            elif "career" in chunk.lower():
                category = "career"
            elif "technology" in chunk.lower():
                category = "technology"
            
            metadata = {
                "chunk_id": i,
                "category": category
            }
            metadata_list.append(metadata)
            categories.add(category)
        
        # Build vector database
        vector_db = VectorDatabase()
        vector_db = await vector_db.abuild_from_list(chunks, metadata_list)
        
        # Store vector database
        vector_dbs[filename] = vector_db
        cl.user_session.set("current_vector_db", vector_db)
        
        # Update status message with success indicator
        process_msg.content = f"""✅ **Document processed successfully!**

**Document Details:**
- **File:** {filename}
- **Chunking:** {chunking_strategy}
- **Chunks Created:** {len(chunks)}
- **Categories Found:** {", ".join(categories)}

You can now ask questions about your document!"""
        
        await process_msg.update()
        
        # Show sample chunks
        if chunks:
            sample_chunk = chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0]
            await cl.Message(
                content=f"""### Document Preview
```
{sample_chunk}
```"""
            ).send()
        
        # Display available categories for filtering
        if categories:
            categories_text = ", ".join([f"`{cat}`" for cat in categories])
            await cl.Message(
                content=f"""### Available Categories
{categories_text}

To filter by category, use the command: `/filter category_name`"""
            ).send()
        
        # Enable querying
        cl.user_session.set("ready_for_query", True)
        
    except Exception as e:
        # Update error message
        process_msg.content = f"❌ Error processing file: {str(e)}"
        await process_msg.update()

def get_distance_func(distance_name: str):
    """Return the distance function based on the selected option"""
    if distance_name == "Cosine Similarity":
        return cosine_similarity
    else:  # Euclidean Distance
        return euclidean_distance

async def process_query(query_text):
    """Process a user query against the RAG system"""
    if not cl.user_session.get("ready_for_query", False):
        await cl.Message(
            content="⚠️ Please upload and process a document first before asking questions."
        ).send()
        return
    
    vector_db = cl.user_session.get("current_vector_db")
    if not vector_db:
        await cl.Message(
            content="⚠️ No vector database found. Please process a document first."
        ).send()
        return
    
    # Get settings
    settings = cl.user_session.get("settings").copy()
    
    # Get user preferences
    distance_metric = cl.user_session.get("distance_metric", "Cosine Similarity")
    category_filter = cl.user_session.get("category_filter", None)
    
    # Create a message with thinking indicator
    response_msg = cl.Message(content="🤔 Searching for relevant information...")
    await response_msg.send()
    
    try:
        # Prepare metadata filter
        metadata_filter = None
        if category_filter:
            metadata_filter = {"category": category_filter}
        
        # Get relevant context using RAG
        distance_func = get_distance_func(distance_metric)
        context_results = vector_db.search_by_text(
            query_text,
            k=3,
            distance_measure=distance_func,
            metadata_filter=metadata_filter
        )
        
        # Build context string
        context_text = ""
        
        for i, (text, score) in enumerate(context_results):
            metadata = getattr(vector_db, 'metadata', {}).get(text, {})
            context_part = f"Chunk {i+1} (Score: {score:.4f}, Category: {metadata.get('category', 'unknown')}):\n"
            context_part += text + "\n\n"
            context_text += context_part
        
        # If no context found
        if not context_text:
            response_msg.content = "⚠️ No relevant information found in the document. Try a different query or adjust the category filter."
            await response_msg.update()
            return
        
        # Create OpenAI messages
        openai_messages = [
            {"role": "system", "content": RAG_SYSTEM_TEMPLATE},
            {"role": "user", "content": RAG_USER_TEMPLATE.format(
                context=context_text,
                user_query=query_text
            )}
        ]
        
        # Update message to show searching for answer
        response_msg.content = "🔍 Found relevant information. Generating answer..."
        await response_msg.update()
        
        # Call OpenAI
        client = AsyncOpenAI()
        response_content = ""
        
        async for stream_resp in await client.chat.completions.create(
            messages=openai_messages,
            stream=True,
            **settings
        ):
            token = stream_resp.choices[0].delta.content
            if token:
                response_content += token
                await response_msg.stream_token(token)
        
        # Create enhanced source display
        sources_text = "## Reference Sources\n\n"
        
        for i, (text, score) in enumerate(context_results):
            metadata = getattr(vector_db, 'metadata', {}).get(text, {})
            chunk_id = metadata.get('chunk_id', i)
            category = metadata.get('category', 'unknown')
            
            # Format source with more details
            sources_text += f"### Source {i+1} (Relevance: {score:.4f})\n"
            sources_text += f"**Chunk ID:** {chunk_id} | **Category:** {category}\n\n"
            
            # Show more of the text content with better formatting
            # Format the full text, using a code block for clarity
            # Break text into paragraphs for readability
            paragraphs = text.split('\n\n')
            formatted_text = ""
            
            # Show first few paragraphs in full, then summarize the rest
            if len(paragraphs) > 3:
                # Show first 3 paragraphs
                for p in paragraphs[:3]:
                    if p.strip():
                        formatted_text += p.strip() + "\n\n"
                if len(paragraphs) > 3:
                    formatted_text += f"... _{len(paragraphs)-3} more paragraphs_ ...\n\n"
            else:
                formatted_text = text
                
            # Wrap in code block for clear reading
            sources_text += "```\n" + formatted_text.strip() + "\n```\n\n"
            
            # Include a separator between sources
            if i < len(context_results) - 1:
                sources_text += "---\n\n"
        
        # Add a note about how sources were chosen
        sources_text += f"\n\n*Sources selected using {distance_metric} with a {'category filter for ' + category_filter if category_filter else 'no category filter'}*"
        
        # Send source information in a separate message after the answer
        await cl.Message(content=sources_text).send()
        
    except Exception as e:
        response_msg.content = f"❌ Error processing query: {str(e)}"
        await response_msg.update()

if __name__ == "__main__":
    # Run the Chainlit app explicitly
    cl.run() 