# RAG Features Demo

This app demonstrates several key RAG capabilities:

## Features

- **Document Processing**: Upload text or PDF files
- **Multiple Chunking Strategies**: Compare character-based vs. sentence-based text splitting
- **Vector Search Options**: Compare cosine similarity vs. euclidean distance
- **Metadata Filtering**: Filter results by categories
- **Question Answering**: Ask questions about your documents

## How to Use

1. **Upload a Document**: Start by uploading a text (.txt) or PDF (.pdf) file
2. **Choose Settings**: Select your preferred chunking strategy and distance metric
3. **Process the File**: Click the "Process File" button to create the vector database
4. **Ask Questions**: Enter your queries in the chat to get answers based on the document

## Example Queries

- "What are the main topics discussed in this document?"
- "Give me a summary of the key points about startups"
- "What does the document say about technology?"

## Behind the Scenes

This app uses:
- Text chunking algorithms to break down documents
- Vector embeddings to represent text semantically
- Metadata tagging for improved filtering
- RAG to retrieve context and answer questions

Try different combinations of settings to see how they affect the quality of responses! 