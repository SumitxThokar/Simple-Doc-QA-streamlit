# Document Q&A Chatbot with LangChain
This web-app allows you to interact with a document by asking questions and receiving concise answers. The app uses LangChain for document processing and a conversational retrieval chain to provide responses.
## Project Flow
- Upload a PDF document
- Split the document into chunks
- Create embeddings and a vector database
- Interact with the document through a chatbot interface
- Collect user information for follow-up (if user's input is "call me") 

Also, we can use function calling to collect user information (I have used if statement instead)
