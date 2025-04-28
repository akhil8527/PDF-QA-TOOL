# PDF Question Answering with Groq and ChromaDB

This project allows you to load a PDF, generate embeddings, and ask questions about the document using Groq LLM and ChromaDB.

## Features
- Load and process PDFs from terminal
- Store and retrieve document embeddings using ChromaDB
- Answer questions using Groq's LLaMA3-70B model

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. (Recommended) Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Linux/Mac
    venv\Scripts\activate      # On Windows
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file and add your Groq API key:

    ```
    GROQ_API_KEY=your_groq_api_key_here
    ```

5. Run the script:

    ```bash
    python main.py --pdf path_to_your_pdf.pdf
    ```

## Requirements

- Python 3.12+
- Packages: requirements.txt

## Example

```bash
$ python main.py --pdf example.pdf
[1] Loading and processing PDF...
[2] Splitting text and storing embeddings in ChromaDB
✅ Created new collection: pdf_qa

Ask a question (or type 'exit'): What is the document about?
