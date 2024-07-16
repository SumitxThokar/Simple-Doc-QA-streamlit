import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import  create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever


# Statefully manage chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def collect_user_info():
    name = st.text_input("Please enter your name:")
    phone = st.text_input("Please enter your phone number:")
    email = st.text_input("Please enter your email:")
    if name and phone and email:
        return {"name": name, "phone": phone, "email": email}
    return None

def main():
    st.title("Document Question Answering with LangChain")
    
    # Step 1: Load the document
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader("uploaded_file.pdf")
        docs = loader.load()
        st.success("Document loaded successfully!")

        # Step 2: Split the document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = text_splitter.split_documents(docs)
        st.success("Document split into chunks successfully!")

        # Step 3: Embedding
        GOOGLE_API_KEY = st.text_input("Enter your Google API key:", type="password")
        if GOOGLE_API_KEY:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

            # Step 4: Vector Database
            vector_database = Chroma.from_documents(documents, embedding=embeddings)
            retriever = vector_database.as_retriever()
            st.success("Vector database created successfully!")
            model = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2,
                convert_system_message_to_human=True
            )
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            history_aware_retriever = create_history_aware_retriever(
                model, retriever, contextualize_q_prompt
            )
            ### Answer question ###
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(model, qa_prompt)         
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            st.success("Retrieval chain created successfully!")

            # Step 6: Chatbot Interaction
            st.header("Chat with the Document")
            user_input = st.text_input("You:", "")
            if user_input:
                if "call me" in user_input.lower():
                    user_info = collect_user_info()
                    if user_info:
                        st.write(f"Thank you, {user_info['name']}. We will call you at {user_info['phone']} or email you at {user_info['email']}.")
                else:
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={
                            "configurable": {"session_id": "abc123"}
                        }
                    )
                    st.write(f"Bot: {response['answer']}")


if __name__ == "__main__":
    main()
