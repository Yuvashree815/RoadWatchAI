from langchain_huggingface import HuggingFaceEmbeddings

# Using a local embedding model to avoid API key requirements
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
