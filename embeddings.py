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
import pacmap

EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2" #smaller model first
READER_MODEL_NAME="HuggingFaceH4/zephyr-7b-beta"

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
    return embedding_model, KNOWLEDGE_VECTOR_DATABASE

def query(embedding_model, KNOWLEDGE_VECTOR_DATABASE, query):
    query_vector=embedding_model.embed_query(query)
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=query, k=5)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
            give a comprehensive answer to the question.
            Respond only to the question asked, response should be concise and relevant to the question.
            Provide the number of the source document when relevant.
            If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
            {context}
            ---
            Now here is the question you need to answer.

            Question: {question}""",
        },
    ]
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=query, context=context)
    answer = READER_LLM(final_prompt)[0]["generated_text"]
    print(answer)







if __name__=="__main__":
    embedding_model, KNOWLEDGE_VECTOR_DATABASE=vectorize()
    question="Who is Louis Desaix?"
    query(embedding_model, KNOWLEDGE_VECTOR_DATABASE, question)