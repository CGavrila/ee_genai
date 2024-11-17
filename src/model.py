import logging
from llama_cpp import Llama

from .rag import query_rag

logging.basicConfig(level=logging.INFO)

MODEL_PATH = "./resources/llama-2-7b-chat.Q4_K_M.gguf"
LLAMA_MAX_TOKENS = 4096
model = Llama(model_path=MODEL_PATH, n_ctx=LLAMA_MAX_TOKENS, n_gpu_layers=-1)

def answer_question(question: str, verbose=True) -> str:
    rag_results = query_rag(question)
    documents = [doc['entity']['text'] for doc in rag_results[0]]
    if verbose:
        logging.info(f"RAG results: {documents}")

    documents_str = "\n\n".join(documents)

    prompt = f"""
        # Goal
        Your goal is to answer the question based on the documents provided.
        Answer using the only the information provided in the documents. If the answer is not in the documents, say "I don't know."
        # Question
        {question}
        # Documents
        {documents_str}
        # Answer:
    """

    max_tokens = LLAMA_MAX_TOKENS - len(model.tokenize(prompt.encode('utf-8')))

    answer = model(prompt, temperature=0.0, max_tokens=max_tokens)

    return answer['choices'][0]['text']

if __name__ == "__main__":
    while True:
        question = input("Enter your question: ")
        answer = answer_question(question, verbose=False)
        print("-" * 100)
        print("Answer:")
        print(answer)
        print("-" * 100)
