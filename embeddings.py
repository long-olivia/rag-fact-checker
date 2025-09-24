from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
import json
import numpy as np
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from colabcode import ColabCode

EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2" #smaller model first

def embed():
    with open("chunks.json", 'r', encoding="utf-8") as f:
        data=json.load(f)

    chunks=[item["Content"] for item in data]

    embedding_model=HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cpu"}, #change to colab once we do the whole dataset
        encode_kwargs={"normalize_embeddings": True}
    )

    batch_size=64
    all_embeddings=[]

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[i:i+batch_size]
        embeddings = embedding_model.embed_documents(batch)
        all_embeddings.extend(embeddings)

    for i, item in enumerate(data):
        item["embedding"] = all_embeddings[i]

    with open("embedded_doc.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    vectorize(data, embedding_model)

def vectorize(): #data, embedding_model insert these two arguments later on
    embedding_model=HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cpu"}, #depends on what we ultimately end up using
        encode_kwargs={"normalize_embeddings": True}
    )
    with open("embedded_doc.json", 'r') as f:
        data=json.load(f)

    docs = [
        Document(
            page_content=item["Content"],
            metadata={
                "ID": item["ID"],
                "Title": item["Title"],
                "Section": item["Section"]
            }
        )
        for item in data
    ]
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    return KNOWLEDGE_VECTOR_DATABASE


# KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
#     docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
# )

if __name__=="__main__":
    vectorize()