import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- UI Header ---
st.set_page_config(page_title="PDF RAG Q&A", page_icon="📄")
st.header("Chat with your PDF 📄")

# --- API Key Input ---
# This allows users to input their key safely, or reads it from Streamlit secrets during deployment
api_key = st.text_input("Enter your Hugging Face Access Token to proceed:", type="password")

if api_key:
    # --- PDF Upload ---
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
    
    if uploaded_file is not None:
        # 1. Extract text from PDF
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
                
        # 2. Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200, # Overlap helps maintain context between chunks
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # 3 & 4. Create embeddings and store in FAISS vector database
        with st.spinner("Processing PDF and creating vector embeddings..."):
            embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name="sentence-transformers/all-MiniLM-l6-v2")
            vector_store = FAISS.from_texts(chunks, embeddings)
            st.success("PDF processed successfully!")
            
        # 5. User Query Input
        user_question = st.text_input("Ask a question about your PDF:")
        
        if user_question:
            # 6. Retrieve relevant chunks
            docs = vector_store.similarity_search(user_question, k=3)
            
            # 7. Generate answers using the LLM
            llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=api_key, max_new_tokens=512)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the user's question using only the following context. If you don't know the answer based on the context, say so.\n\nContext:\n{context}"),
                ("human", "{question}")
            ])
            
            chain = prompt | llm | StrOutputParser()
            
            with st.spinner("Generating answer..."):
                context_text = "\n\n".join([doc.page_content for doc in docs])
                response_text = chain.invoke(
                    {"context": context_text, "question": user_question}
                )
                
            # --- Interactive UI Output ---
            st.write("### Answer:")
            st.write(response_text)