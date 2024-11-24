from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import Pinecone
#from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
import os


load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


docs = load_pdf_file(data="./data")
chunks = text_split(docs)
embedding = download_huggingface_embeddings()


#pinecone.init(api_key=PINECONE_API_KEY)
pc = PineconeClient(
    api_key=os.environ.get("PINECONE_API_KEY")
)
#pc = Pinecone(api_key= PINECONE_API_KEY)
index_name = "medi"

# Create the index if it doesn't exist
if index_name not in [index.name for index in pc.list_indexes().names()]:
  pc.create_index(
    name="medi",
    dimension=384, 
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

doc_search = Pinecone.from_documents(
    documents=chunks,
    index_name="medi",
    embedding=embedding
)