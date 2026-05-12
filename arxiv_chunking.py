import os
import requests
import xml.etree.ElementTree as ET
import fitz
from tqdm import tqdm
import uuid
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


SAVE_DIR = "llm_papers"
MAX_RESULTS = 100
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

os.makedirs(SAVE_DIR, exist_ok=True)
'''
1. we will fetch LLM related papers from arxiv, published after 2023, download and extract
2. then chunking and build vector DB
3. then create query RAG and citation and QA dataset generation
4. then call all the functions
'''

def fetch_llm_papers():
    url = "https://export.arxiv.org/api/query"

    query = """
    (cat:cs.CL OR cat:cs.LG)
    AND (large language model OR LLM OR transformer OR GPT OR instruction tuning OR RLHF)
    """

    params = {
        "search_query": query,
        "start": 0,
        "max_results": MAX_RESULTS,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    response = requests.get(url, params=params)
    root = ET.fromstring(response.text)

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []

    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.lower()
        published = entry.find("atom:published", ns).text[:4]

        keywords = ["llm", "large language model", "transformer", "gpt"]
        if not any(k in summary for k in keywords):
            continue

        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib["href"]

        if int(published) >= 2023:
            papers.append({
                "title": title,
                "pdf_url": pdf_url,
                "year": int(published),
                "summary": summary
            })

    return papers


def download_and_extract(paper):
    filename = os.path.join(SAVE_DIR, f"{uuid.uuid4().hex}.pdf")

    try:
        r = requests.get(paper["pdf_url"])
        with open(filename, "wb") as f:
            f.write(r.content)

        doc = fitz.open(filename)

        text = ""
        for page in doc:
            text += page.get_text()

        return {
            "text": text,
            "metadata": {
                "title": paper["title"],
                "year": paper["year"],
                "source": paper["pdf_url"]
            }
        }

    except Exception as e:
        print("Error:", e)
        return None

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def chunk_documents(papers):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = []

    for paper in tqdm(papers):
        extracted = download_and_extract(paper)
        if not extracted:
            continue

        chunks = splitter.split_text(extracted["text"])

        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata=extracted["metadata"]
                )
            )

    return docs

def build_vector_db(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)
    return db

def query_rag(db, query):
    results = db.similarity_search(query, k=5)

    context = ""
    citations = []

    for r in results:
        context += r.page_content + "\n\n"
        citations.append(f"{r.metadata['title']} ({r.metadata['year']})")

    return context, list(set(citations))

def generate_qa_dataset(docs, output_file="llm_qa.jsonl"):
    from transformers import pipeline

    generator = pipeline("text-generation", model="gpt2", device=-1)

    with open(output_file, "w") as f:
        for doc in tqdm(docs[:20]): # Reduced to 20 documents for quicker feedback on CPU
            prompt = f"""
            Generate a high-quality research question and answer from this text.

            Text:
            {doc.page_content[:500]}

            Focus on LLM concepts.

            Format:
            Q:
            A:
            """

            try:
                output = generator(prompt, max_length=100)[0]["generated_text"] # Reduced max_length

                q = output.split("A:")[0].replace("Q:", "").strip()
                a = output.split("A:")[-1].strip()

                entry = {
                    "messages": [
                        {"role": "system", "content": "Answer as an AI researcher and cite sources."},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a + f" [source: {doc.metadata['title']}, {doc.metadata['year']}]"}
                    ]
                }

                f.write(json.dumps(entry) + "\n")

            except Exception as e: # Explicitly print the exception
                print(f"Error generating QA for document: {doc.metadata.get('title', 'Unknown Title')}. Error: {e}")
                continue


def main():
    print("Fetching LLM papers...")
    papers = fetch_llm_papers()

    print(f"Found {len(papers)} relevant papers")

    print("Chunking...")
    docs = chunk_documents(papers)

    print("Building vector DB...")
    db = build_vector_db(docs)

    print("Testing query...")
    context, citations = query_rag(
        db, "What is instruction tuning in large language models?"
    )

    print("\nContext sample:\n", context[:1000])
    print("\nCitations:\n", citations)

    print("Generating QA dataset...")
    generate_qa_dataset(docs)



if __name__ == "__main__":
    main()