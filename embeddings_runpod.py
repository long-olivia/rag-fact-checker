import os
import json
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.embeddings.base import Embeddings

id_to_embedding = {}
docs = []

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
INDEX_PATH = "faiss_index"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

device = 0 if torch.cuda.is_available() else -1 

embedding_pipeline = pipeline(
    task="feature-extraction",
    model=EMBEDDING_MODEL_NAME,
    tokenizer=EMBEDDING_MODEL_NAME,
    device=device
)

class IdentityEmbeddings(Embeddings):
    def embed_documents(self, documents):
        return [np.array(doc.embedding, dtype=np.float32) for doc in documents]
    def embed_query(self, text):
        embedding = embedding_pipeline([text], padding=True, truncation=True)
        return np.mean(np.array(embedding[0]), axis=0).astype(np.float32)

def embed_texts(texts):
    embeddings = embedding_pipeline(texts, padding=True, truncation=True)
    return [np.mean(np.array(e), axis=0).tolist() for e in embeddings]

def embed():
    with open("chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    vector_db = FAISS.from_documents(
        docs,
        embedding=IdentityEmbeddings(),
        distance_strategy=DistanceStrategy.COSINE
    )
    chunks = [item["Content"] for item in data]
    batch_size = 64
    all_embeddings = []

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[i:i + batch_size]
        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)

    for i, item in enumerate(data):
        item["embedding"] = all_embeddings[i]

    with open("embedded_doc.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data

def vectorize(data):
    global id_to_embedding
    global docs
    id_to_embedding = {item["ID"]: item["embedding"] for item in data}
    docs = [
        Document(
            page_content=item["Content"],
            metadata={"ID": item["ID"], "Title": item["Title"], "Section": item["Section"]},
            embedding=np.array(item["embedding"],dtype=np.float32)
        )
        for item in data
    ]

    vector_db = FAISS.from_documents(
        docs, embedding=IdentityEmbeddings(), 
        distance_strategy=DistanceStrategy.COSINE
    )
    vector_db.save_local(INDEX_PATH)
    print("saved")
    return vector_db

def query(vector_db, question):
    retrieved_docs = vector_db.similarity_search(question, k=5)

    context = "\n".join(
        [f"Document {i}:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)]
    )

    prompt_template = (
        "System: Using the information contained in the context, "
        "give a comprehensive answer to the question. "
        "Be concise, relevant, and cite the source number when relevant.\n\n"
        "Context:\n{context}\n---\nQuestion: {question}"
    )
    final_prompt = prompt_template.format(context=context, question=question)

    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    model = AutoModel.from_pretrained(READER_MODEL_NAME)

    llm = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500
    )

    answer = llm(final_prompt)[0]["generated_text"]
    print("Answer:", answer)

if __name__ == "__main__":
    if os.path.exists("embedded_doc.json"):
        with open("embedded_doc.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = embed()

    if os.path.exists(f"{INDEX_PATH}/index.faiss"):
        vector_db = FAISS.load_local(INDEX_PATH, embeddings=IdentityEmbeddings(), allow_dangerous_deserialization=True)
        print("Loaded existing FAISS index.")
    else:
        print("Building FAISS index...")
        vector_db = vectorize(data)
    query(vector_db, "Who is Louis Desaix?")
