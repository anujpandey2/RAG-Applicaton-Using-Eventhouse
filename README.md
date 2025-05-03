Build RAG Application using Eventhouse
![alt text](excalidraw-animate.svg)
After newly added AI functions in eventhouse and vector type compatibility in Eventhouse I thought to do a POC. 
In just 9 simple steps :
1. Text Extraction from PDFs
The content is converted to markdown format using pymupdf4llm, which preserves structural information while making the text more processable.
2. Text Chunking
The extracted text is split into smaller, manageable chunks using LangChain's.
3. Text Cleaning and Preprocessing
The chunks undergo cleaning to remove markdown syntax (like hashtags and asterisks) and unnecessary whitespace. This creates cleaner text that's better suited for embedding.
4. Token Counting (Optional)
The system implements token counting functionality using tiktoken, which helps monitor the size of text being processed and ensures it stays within model limits.
5. Embedding Generation
For each chunk of text, the system generates embeddings using Azure OpenAI's text-embedding-ada-002 model. These embeddings are vector representations that capture the semantic meaning of the text.
6. Creating a Structured Dataset
All processed text chunks and their corresponding embeddings are organized into a pandas DataFrame, which is then converted to a Spark DataFrame for distributed processing.
7. Write to Eventhouse to store Vectors
The system connects to a Kusto (Azure Data Explorer) database and stores the processed data, including the text chunks, metadata, and embeddings.
8. Implementing Semantic Search
A similarity search function is developed using newly added vector similarity (cosine similarity) in Eventhouse to find the most relevant chunks of text based on user queries. This allows natural language questions to be matched with the most semantically relevant parts of the resumes.
9. RAG
The system integrates with OpenAI's GPT model (gpt-4o-mini) to generate answers based on the retrieved relevant information, providing a natural language interface to the resume data.

Optional step
1. Visualization and Clustering
The POC includes visualization capabilities using dimensionality reduction techniques to represent high-dimensional embeddings in 2D space. K-means clustering is applied to group similar resumes, which helps identify patterns in the data.
Core Techniques Used
1.	Vector Embeddings: Transforming text into numerical vectors that capture semantic meaning
2.	Semantic Search: Using cosine similarity to find relevant information based on meaning rather than keywords
3.	Text Chunking: Breaking documents into manageable pieces while preserving context
4.	Dimensionality Reduction: Visualizing high-dimensional data in 2D space
5.	Clustering: Grouping similar documents to identify patterns
6.	Integration of Multiple Services: Combining Azure OpenAI, Spark, and Kusto database for a complete workflow


