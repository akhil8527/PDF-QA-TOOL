# ========== IMPORTING LIBRARIES ========== #
import os
import yaspin
import chromadb
import argparse
import pypdfium2
from groq import Groq
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ========== CONFIG SETUP ========== #
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)


chromadb_client = chromadb.Client()

collection_name = "pdf_qa"
existing_collections = chromadb_client.list_collections()

if collection_name in [c.name for c in existing_collections]:
    collection = chromadb_client.get_collection(name=collection_name)
    print(f"✅ Using existing collection: {collection_name}")
else:
    collection = chromadb_client.create_collection(name=collection_name)
    print(f"✅ Created new collection: {collection_name}")

### By default, Chroma uses the Sentence Transformers all-MiniLM-L6-v2 model to compute embeddings for documents.
embedding_function = embedding_functions.DefaultEmbeddingFunction()


# ========== LOAD PDF ========== #
def load_pdf_text(file_path):

    pdf = pypdfium2.PdfDocument(file_path)
    text = ""

    for page in pdf:
        text_page = page.get_textpage()
        page_text = text_page.get_text_bounded()
        text += page_text
    pdf.close()
    return text


# ========== SPLIT TEXT ========== #
def split_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splitted_text = text_splitter.split_text(text)

    return splitted_text


# ========== STORE EMBEDDINGS ========== #
def store_embeddings(chunks):
    embeddings = embedding_function(chunks)  # embed all chunks together
    ids = [str(idx) for idx in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)


# ========== SEARCH CONTEXT ========== #
def search_context(query, top_k=3):
    query_embeddings = embedding_function(query)
    results = collection.query(query_embeddings=query_embeddings, n_results=top_k)

    if results["documents"]:
        context = "\n".join(doc[0] for doc in results["documents"])
        return context
    return ""


# ========== QUERY GROQ MODEL ========== #
def query_groq(context, question):
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"

    with yaspin.yaspin(text="Thinking...", color="cyan") as spinner:
        try:
            response = GROQ_CLIENT.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )

            spinner.ok("✅ ")
            output_text = response.choices[0].message.content.strip()

            return output_text
        except Exception as e:
            spinner.fail("💥 ")
            raise e


# ========== MAIN FUNCITON ========== #
if __name__ == "__main__":

    # taking pdf from terminal
    parser = argparse.ArgumentParser(
        description="PDF-based Question Answering with Groq and ChromaDB"
    )
    parser.add_argument(
        "--pdf", type=str, required=True, help="Path to the PDF document"
    )
    args = parser.parse_args()

    pdf_file_path = args.pdf

    if not os.path.exists(pdf_file_path):
        print(f"File not found: {pdf_file_path}")
        exit(1)

    try:
        print("[1] Loading and processing PDF...")
        pdf_text = load_pdf_text(pdf_file_path)
    except Exception as e:
        print(f"❌ Error while loading PDF: {e}")
        exit(1)

    try:
        print("[2] Splitting text and storing embeddings in ChromaDB...")
        chunks = split_text(pdf_text)
    except Exception as e:
        print(f"❌ Error while splitting text: {e}")
        exit(1)

    try:
        store_embeddings(chunks)
    except Exception as e:
        print(f"❌ Error while storing embeddings to ChromaDB: {e}")
        exit(1)

    while True:
        try:
            user_question = input("\nAsk a question (or type 'exit'): ")
            if user_question.lower() == "exit":
                break

            context = search_context(user_question)
            if not context:
                print("No relevant context found.")
                continue

            answer = query_groq(context, user_question)
            print("\nAnswer:", answer)

        except KeyboardInterrupt:
            print("\n❗ Interrupted by user. Exiting...")
            break

        except Exception as e:
            print(f"❌ Error during QA loop: {e}")

