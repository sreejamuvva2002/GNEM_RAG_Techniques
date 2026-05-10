from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import answer_relevancy
from datasets import Dataset
from ragas import evaluate

llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:14b", temperature=0, base_url="http://localhost:11434"))
emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434"))
ds = Dataset.from_dict({"question": ["What is AI?"], "answer": ["AI is artificial intelligence."], "contexts": [["AI is a branch of computer science."]], "ground_truths": [["AI is a branch of computer science."]]})
answer_relevancy.llm = llm
if hasattr(answer_relevancy, "embeddings"):
    answer_relevancy.embeddings = emb

print(evaluate(ds, metrics=[answer_relevancy], llm=llm, embeddings=emb))
