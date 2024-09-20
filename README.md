
---

# LLaMa-3-RAG-PDF

This repository contains a Streamlit-based web application for interacting with machine learning models and vector databases.

## Overview

The application is built using Streamlit to provide an easy-to-use web interface for:

- Training and evaluating machine learning models (`model.py`).
- Performing operations with a vector database (`Vector_DB.py`).

## Files in the Repository

### 1. `model.py`
Contains the code for machine learning models, including functions for model creation, training, and evaluation.

### 2. `Vector_DB.py`
Handles operations related to vector databases, such as inserting vectors, querying for similar vectors, and database management.


## Requirements

- Python 3.x
- Streamlit
- Other dependencies (installable via `requirements.txt`)

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sathiesh05/chat-with-pdf.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd chat-with-pdf
   ```
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit application:**
   ```bash
   streamlit run model.py
   ```

## Usage

Once the Streamlit app is running, you can access it in your web browser. The interface will guide you through:

1. **Model Operations:**
   - Train, evaluate, and manage machine learning models directly from the web interface.

2. **Vector Database Operations:**
   - Perform various operations with the vector database, such as inserting new vectors or querying for similar ones.
