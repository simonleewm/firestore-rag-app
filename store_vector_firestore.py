import vertexai
import logging
import google.cloud.logging
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

import pickle
from IPython.display import display, Markdown

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

PROJECT_ID='qwiklabs-gcp-01-02da72d36ae2'
LOCATION='us-central1'

vertexai.init(project=PROJECT_ID, location=LOCATION)

embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")

loader = PyMuPDFLoader("nutrition-and-cancer-booklet.pdf")
data = loader.load()

def clean_page(page):
    return page.page_content.replace("-\\n","").replace("\\n"," ").replace("\\x02","").replace("\\x03","")

# cleaned_pages = [clean_page(page) for page in data]
# text_splitter = SemanticChunker(embedding_model)
# chunked_content = text_splitter.create_documents(cleaned_pages)
# chunked_content = [chunk.page_content for chunk in chunked_content if chunk.page_content]
# chunked_embeddings = embedding_model.embed_documents(chunked_content)


# Save the chunked content and embeddings to pickle files
with open("chunked_content.pkl", "wb") as f:
    pickle.dump(chunked_content, f)

with open("chunked_embeddings.pkl", "wb") as f:
    pickle.dump(chunked_embeddings, f)

# Load the chunked content and embeddings from pickle files
chunked_content = pickle.load(open("chunked_content.pkl", "rb"))
chunked_embeddings = pickle.load(open("chunked_embeddings.pkl", "rb"))


# Logging setup
client = google.cloud.logging.Client()
client.setup_logging()
log_message = f"chunked contents are: {chunked_content[0][:20]}"
logging.info(log_message)



# Initialize Firestore client
db = firestore.Client(project=PROJECT_ID)
collection = db.collection("nutrition-and-cancer")

# Store each embedding and chunk
pairs = zip(chunked_content, chunked_embeddings)

for content, embedding in pairs:
    doc = {
        "chunk": content,
        "embedding": Vector(embedding)
    }
    collection.add(doc)

print("Data loaded into Firestore.")



def search_vector_database(query: str):
  """
  Receives a query, gets its embedding, and compiles a context
  consisting of the text from the 5 documents with the most
  similar embeddings.
  """
  context = ""

  # 1. Generate the embedding for the query.
  emb = embedding_model.embed_query(query)

  # 2. Find the 5 nearest neighbors.
  neighbors = collection.find_nearest(
      vector_field="embedding",
      query_vector=Vector(emb),
      distance_measure=DistanceMeasure.COSINE,
      limit=5
  ).get()

  # 3. Extract the 'chunk' field from each document.
  context = "\n\n".join([neighbor.to_dict()["chunk"] for neighbor in neighbors])

  return context



# Run this for testing
# query = "When eating eggs, what should I be aware of?"
# relevant_text = search_vector_database(query)
# print(f"### Related Content:\n\n{relevant_text}\n\n")

# chat_model = GenerativeModel("gemini-2.5-flash")  # Instantiate GenerativeModel directly
# chat = chat_model.start_chat()

# response = chat.send_message(
#     f"Use the following to answer the question:\n\n{relevant_text}\n\nQuestion: {query}"
# )
# print(f"### Model Response:\n\n{response.text}")