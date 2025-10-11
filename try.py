import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

class RAGPipeline:
    def __init__(self, documents, embedding_model_name, reader_model_name, k=5):
        self.k = k
        self.documents = documents
        self.doc_contents = [doc['Content'] for doc in documents]
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)
        self.document_embeddings = self._embed_documents()
        print("Document embeddings generated.")

        print(f"Loading reader model and tokenizer: {reader_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(reader_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token 

        self.model = AutoModelForCausalLM.from_pretrained(
            reader_model_name,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16,
            device_map='auto'
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=self.model.dtype,
            device_map='auto'
        )


    def _embed_documents(self):
        return self.embedder.encode(self.doc_contents, convert_to_tensor=True, device=self.device)

    def _retrieve(self, query_embedding):
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        doc_embeddings_np = self.document_embeddings.cpu().numpy()
        
        similarities = cosine_similarity(query_embedding_np, doc_embeddings_np)[0]
        
        top_k_indices = np.argsort(similarities)[::-1][:self.k] #rerank based on similiarity 
        
        retrieved_documents = [self.documents[i] for i in top_k_indices]
        
        return retrieved_documents

    def _format_prompt(self, query, retrieved_docs):
        context_text = "\n\n".join([f"Chunk {i+1}: {doc['Content']}" for i, doc in enumerate(retrieved_docs)])
        
        system_message = (
            "You are an expert Q&A assistant. Use the following pieces of retrieved "
            "context to answer the user's question concisely. "
            "If the answer cannot be found in the context, state that you do not "
            "have enough information, but do not make up an answer."
        )
        
        full_query = (
            f"{system_message}\n\n"
            f"--- CONTEXT ---\n{context_text}\n\n"
            f"--- QUESTION ---\n{query}"
        )

        messages = [
            {"role": "user", "content": full_query}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def query(self, user_query, max_new_tokens=256):
        print("-" * 50)
        print(f"User Query: {user_query}")
        
        start_time = time.time()
        
        query_embedding = self.embedder.encode(user_query, convert_to_tensor=True, device=self.device)
        
        retrieved_docs = self._retrieve(query_embedding)
        retrieval_time = time.time()
        
        print(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time - start_time:.2f}s:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. ID {doc['ID']} | Title: {doc['Title']} | Snippet: \"{doc['Content'][:50]}...\"")
        
        formatted_prompt = self._format_prompt(user_query, retrieved_docs)
        
        generation_output = self.generator(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        full_response = generation_output[0]['generated_text']
        
        try:
            assistant_start_marker = "<|assistant|>"
            start_index = full_response.rfind(assistant_start_marker)
            
            if start_index != -1:
                answer = full_response[start_index + len(assistant_start_marker):].strip()
            else:
                answer = full_response[len(formatted_prompt):].strip()
        except Exception as e:
             answer = f"Error extracting answer: {e}"

        generation_time = time.time()
        print(f"Generation completed in {generation_time - retrieval_time:.2f}s.")
        print("-" * 50)
        
        return answer

def load(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
        

if __name__ == "__main__":
    documents = load("chunks.json")

    rag = RAGPipeline(documents, EMBEDDING_MODEL_NAME, READER_MODEL_NAME, k=5)
    
    sample_query = "What happened to Louis Desaix after he was recalled from Upper Egypt?" 
    print(f"\nSample Query: {sample_query}\n")
    answer = rag.query(sample_query)    
    print(f"\nFinal Answer: {answer}")

    sample_query_2 = "Where was Desaix born and who were his parents?"
    print(f"\nSample Query: {sample_query_2}\n")
    answer_2 = rag.query(sample_query_2)
    print(f"\nFinal Answer: {answer_2}")
