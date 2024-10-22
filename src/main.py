import streamlit as st
import PyPDF2
import faiss
import numpy as np
from groq import Groq
import uuid
import os

class VectorDBRAGChat:
    def __init__(self, persist_directory="./faiss_index"):
        # Initialize Groq client for embeddings and model
        self.groq_client = Groq(api_key='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')
        self.persist_directory = persist_directory
        
        # FAISS setup
        self.dimension = 4096  # The dimensionality of LLaMA embeddings (example)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_to_chunk = {}  # To store chunks by their IDs
        
        self.pdf_text = ""
        self.current_pdf_id = None

    @staticmethod
    def read_pdf(file):
        """Extract text from PDF file"""
        text = ""
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def chunk_text(self, text, chunk_size=1000, overlap=100):
        """Split text into overlapping chunks"""
        chunks = []
        chunk_ids = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            if end < text_length:
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                split_point = max(last_period, last_newline)
                if split_point > start:
                    end = split_point + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                chunk_ids.append(str(uuid.uuid4()))
            start = end - overlap

        return chunks, chunk_ids

    def generate_embeddings(self, chunks):
        """Generate embeddings using Groq's LLaMA 3 70B"""
        response = self.groq_client.embeddings.create(
            input=chunks,  # Assuming 'input' is the correct keyword argument
            model="llama-3.1-70b-versatile"  # Replace with the correct model name as needed
        )
        embeddings = np.array([result['embedding'] for result in response.embeddings])
        return embeddings

    def process_pdf(self, text, pdf_name):
        """Process PDF text into chunks, generate embeddings, and add to FAISS index"""
        self.pdf_text = text
        self.current_pdf_id = str(uuid.uuid4())
        
        # Create chunks with unique IDs
        chunks, chunk_ids = self.chunk_text(text)
        
        # Generate embeddings for the chunks
        embeddings = self.generate_embeddings(chunks)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings)
        
        # Store the chunks
        for chunk_id, chunk in zip(chunk_ids, chunks):
            self.id_to_chunk[chunk_id] = chunk
        
        return len(chunks)

    def search_chunks(self, query, k=3):
        """Search chunks using FAISS similarity search"""
        if not self.current_pdf_id:
            return []
        
        # Generate embedding for the query
        query_embedding = self.generate_embeddings([query])
        
        # Perform the FAISS search
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve the top-k chunks
        chunk_ids = list(self.id_to_chunk.keys())
        return [self.id_to_chunk[chunk_ids[idx]] for idx in indices[0]]

    def generate_response(self, context, query, model_name="llama3-70b-4096"):
        """Generate response using Groq"""
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. Use the following context to answer the user's question. Context: {context}"},
            {"role": "user", "content": query},
        ]
        response = self.groq_client.chat.completions.create(
            messages=messages,
            model=model_name
        )
        return response.choices[0].message.content

    def process_chat(self, query, k=3):
        """Process chat query using vector database RAG"""
        if not self.current_pdf_id:
            return "Please upload and process a PDF first."
        
        relevant_chunks = self.search_chunks(query, k)
        context = " ".join(relevant_chunks)
        return self.generate_response(context, query)

    def clear_current_pdf(self):
        """Clear chunks for the current PDF"""
        if self.current_pdf_id:
            self.index.reset()  # Clear the FAISS index
            self.current_pdf_id = None
            self.pdf_text = ""
            self.id_to_chunk = {}

def main():
    st.set_page_config(page_title="Vector DB PDF Chat with FAISS", layout="wide")
    
    # Create persist directory if it doesn't exist
    if not os.path.exists("./faiss_index"):
        os.makedirs("./faiss_index")
    
    # Initialize session state
    if 'rag_chat' not in st.session_state:
        st.session_state.rag_chat = VectorDBRAGChat()
    
    st.title("📚 PDF Chat with FAISS Vector Database")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 PDF Upload & Processing")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                # Clear previous PDF data
                st.session_state.rag_chat.clear_current_pdf()
                
                # Read and process PDF text
                pdf_text = st.session_state.rag_chat.read_pdf(uploaded_file)
                chunk_count = st.session_state.rag_chat.process_pdf(pdf_text, uploaded_file.name)
                
                st.success(f"PDF processed successfully! Created {chunk_count} chunks.")
                with st.expander("View Extracted Text"):
                    st.text_area("PDF Content", pdf_text, height=300)

    with col2:
        st.header("💬 Chat Interface")
        
        # Chat interface
        if not st.session_state.rag_chat.current_pdf_id:
            st.info("Please upload a PDF first to start chatting.")
        else:
            query = st.text_input("Ask a question about your PDF:")
            if query:
                with st.spinner("Generating response..."):
                    response = st.session_state.rag_chat.process_chat(query)
                    st.write("Response:", response)

if __name__ == "__main__":
    main()
