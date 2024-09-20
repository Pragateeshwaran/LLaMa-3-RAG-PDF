import streamlit as st
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import tempfile

# Read PDF file
def read_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Convert text to vectors
def text_to_vector(text, model):
    sentences = text.split("\n")
    embeddings = model.encode(sentences, convert_to_tensor=False)
    return np.array(embeddings)

# Store vectors in FAISS index
def store_in_faiss(vectors, output_index):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, output_index)

# Streamlit app
def main():
    st.title("PDF to FAISS Indexing")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = read_pdf(uploaded_file)
            st.text_area("Extracted Text", pdf_text, height=300)

            model = SentenceTransformer('all-MiniLM-L6-v2')
            vectors = text_to_vector(pdf_text, model)

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                output_index = temp_file.name
                store_in_faiss(vectors, output_index)
                st.write(f"FAISS index saved to {output_index}")

if __name__ == "__main__":
    main()
