import os
import yaml
import re
import logging
import google.cloud.logging
from flask import Flask, render_template, request

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from langchain_google_vertexai import VertexAIEmbeddings

# Configure Cloud Logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# Read application variables from the config fle
BOTNAME = "QnA Bot"
SUBTITLE = "Your Cancer Nutrition Expert"

app = Flask(__name__)

# Initializing the Firebase client
db = firestore.Client()

# Instantiate a collection reference
collection = db.collection("nutrition-and-cancer")

# Instantiate an embedding model here
embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")

# Instantiate a Generative AI model here
gen_model = GenerativeModel("gemini-2.5-flash", generation_config={"temperature": 0.0})

# Implement this function to return relevant context
# from your vector database
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

# Provide Gemini the context data, generate a response, and return the response text.
def ask_gemini(question):

    # 1. Create a prompt_template with instructions to the model.
    prompt_template = """
    Your name is QnA Bot, you are an expert in nutritition for people living with cancer.
    Use the following context to answer the user's question.
    If the context does not contain the answer, state that you are an expert in nutritition for people living with cancer and the question is outside your expertise.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # 2. Use your search_vector_database function to retrieve context.
    context = search_vector_database(question)

    # 3. Format the prompt template with the question & context.
    prompt = prompt_template.format(context=context, question=question)

    # 4. Configure safety settings and get the response from Gemini.
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    response = gen_model.generate_content(prompt, safety_settings=safety_settings)

    return response.text

# Convert markdown to HTML
def markdown_to_html(text):
    # Convert **text** to <strong>text</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Convert * text to <li>text</li>
    lines = text.split('\n')
    in_list = False
    for i, line in enumerate(lines):
        if line.strip().startswith('* '):
            lines[i] = '<li>' + line.strip()[2:] + '</li>'
            if not in_list:
                lines.insert(i, '<ul>')
                in_list = True
    if in_list:
        lines.append('</ul>')
    return '\n'.join(lines).replace('\n', '<br>')

# The Home page route
@app.route("/", methods=["POST", "GET"])
def main():

    # The user clicked on a link to the Home page
    # They haven't yet submitted the form
    if request.method == "GET":
        question = ""
        answer = "Hello, I'm here to answer any questions about nutritition for people living with cancer. How can I help?"

    # The user asked a question and submitted the form
    # The request.method would equal 'POST'
    else:
        answer = ""
        question = request.form["input"]
        # Do not delete this logging statement.
        logging.info(
            question,
            extra={"labels": {"service": "cymbal-service", "component": "question"}},
        )
        
        # Ask Gemini to answer the question using the data from the database
        answer = ask_gemini(question)

    # Do not delete this logging statement.
    logging.info(
        answer, extra={"labels": {"service": "cymbal-service", "component": "answer"}}
    )
    print("Answer: " + answer)

    # Display the home page with the required variables set
    config = {
        "title": BOTNAME,
        "subtitle": SUBTITLE,
        "botname": BOTNAME,
        "message": markdown_to_html(answer),
        "input": question,
    }

    return render_template("index.html", config=config)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
