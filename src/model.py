import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Initialize Groq client
client = Groq(api_key='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')

# Load FAISS index
@st.cache_resource
def load_faiss_index(index_path):
    return faiss.read_index(index_path)

# Search vectors in FAISS index
def search_vectors(index, query_vector, k=5):
    distances, indices = index.search(query_vector, k)
    return indices, distances

# Convert query to vector
def query_to_vector(query, model):
    return np.array([model.encode(query)])

# Generate response using Groq
def generate_response(context, query, model_name="llama3-8b-8192"):
    messages = [
        {"role": "system", "content": f"Context: {context}"},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(messages=messages, model=model_name)
    return response.choices[0].message.content

# RAG chat function
def rag_chat(index, query, embedding_model, groq_model_name="llama3-8b-8192", k=5):
    query_vector = query_to_vector(query, embedding_model)
    indices, _ = search_vectors(index, query_vector, k)
    context = " ".join([f"Segment {i}" for i in indices])
    response = generate_response(context, query, model_name=groq_model_name)
    return response

# Streamlit app
def main():
    st.title("RAG Chat with FAISS and Groq")
    
    index_path = st.text_input("FAISS Index Path", value="D:/Rag-Vector-DB/Rag-Vector-DB/DB_Storage/vectors.index")
    query = st.text_input("Enter your query:")
    model_name = st.selectbox("Select Model", ["llama3-8b-8192"])

    if st.button("Get Response"):
        with st.spinner("Processing..."):
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            index = load_faiss_index(index_path)
            response = rag_chat(index, query, embedding_model, groq_model_name=model_name)
            st.write("Response:", response)

if __name__ == "__main__":
    main()
