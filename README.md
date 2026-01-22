# Deploy a RAG application with vector search in Firestore

This guide demonstrates how to build a real-world Generative AI Q&A solution using a RAG (Retrieval-Augmented Generation) framework. We will use Firestore as a vector database and deploy a Flask app as a user interface to query the nutrition guide for people living with cancer knowledge base.

## Overview

*   Load a text document (PDF) and split it into chunks.
*   Generate embeddings for each chunk using Vertex AI Embeddings API.
*   Store the text chunks and their embeddings in Firestore.
*   Conduct vector search in Firestore to find similar documents to a query.
*   Use Gemini to generate a response based on the context of similar documents.
*   Deploy the Flask application to Cloud Run.

## Prerequisites

*   A Google Cloud project with billing enabled.
*   `gcloud` CLI installed and authenticated.
*   `git` installed.
*   `python3` and `pip` installed.
*   `docker` installed.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/simonleewm/firestore-rag-app.git
    cd rag-app-with-firestore-vector-search
    ```

2.  **Set up your Google Cloud project:**

    ```bash
    export PROJECT_ID="<YOUR_PROJECT_ID>"
    gcloud config set project $PROJECT_ID
    ```

3.  **Enable necessary APIs:**

    ```bash
    gcloud services enable \
        compute.googleapis.com \
        iam.googleapis.com \
        iamcredentials.googleapis.com \
        sts.googleapis.com \
        cloudresourcemanager.googleapis.com \
        aiplatform.googleapis.com \
        firestore.googleapis.com \
        artifactregistry.googleapis.com \
        run.googleapis.com \
        logging.googleapis.com
    ```

4.  **Create a Firestore database:**

    Go to the Firestore console and create a database (ID: default) in "Native Mode". Choose a location (e.g., `us-central1`).

5.  **Create an Artifact Registry repository:**

    ```bash
    gcloud artifacts repositories create nutrition-and-cancer-repo \
        --repository-format=docker \
        --location=us-central1 \
        --description="Nutrition and Cancer repo"
    ```

6.  **Set up the environment and load data into Firestore:**

    a. **Install additional Python packages:**

    ```bash
    pip install --upgrade --quiet google-cloud-logging google-cloud-firestore google-cloud-aiplatform langchain "langchain-experimental==0.3.4" langchain-community langchain-google-vertexai pymupdf "langchain-core<2.0.0" "langchain-text-splitters<2.0.0"
    ```

    b. **Download the knowledge base document:**

    For our knowledge base, we're using the PDF support guide provided by Cancer Council - *Nutrition for People Living with Cancer*.

    ```bash
    wget -O nutrition-and-cancer-booklet.pdf https://www.cancer.org.au/assets/pdf/nutrition-and-cancer-booklet
    ```

    c. **Run a Python script to process the data and populate Firestore.**

    Run the included Python script in a Jupyter or Colab notebook to store the text chunks and embeddings into Firestore. **Remember to replace `<YOUR_PROJECT_ID>` with your actual project ID.**
    
    ```bash
    python3 store_vector_firestore.py 
    ```
    
    Digging into the code - we first initialise an `embedding_model` and use the LangChain class `PyMuPDFLoader` to loads the contents of the PDF to a variable.
    
    ```python
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")
    loader = PyMuPDFLoader("nutrition-and-cancer-booklet.pdf")
    data = loader.load()
    ```
        
    The `clean_page` function performs basic cleaning on artifacts found on the document. We then use LangChain's `SemanticChunker` to split the pages into text chunks. `SemanticChunker` determines when to start a new chunk when it encounters a larger distance between sentence embeddings. The strings of page contents is stored into a list called `chunked_content`. Then we use the embedding model to generate the `chucked_embeddings`.
    
    ```python
    def clean_page(page):
        return page.page_content.replace("-\\n","").replace("\\n"," ").replace("\\x02","").replace("\\x03","")

    cleaned_pages = [clean_page(page) for page in data]
    text_splitter = SemanticChunker(embedding_model)
    chunked_content = text_splitter.create_documents(cleaned_pages)
    chunked_content = [chunk.page_content for chunk in chunked_content if chunk.page_content]
    chunked_embeddings = embedding_model.embed_documents(chunked_content)
    ```
    
    Once the Firestore database is created (in the previous step), store the embeddings and chunks into a collection named `nutrition-and-cancer`.
    
    ```python
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
    ```

    d. **Create the Firestore vector index on the embedding field, which allows us later to query on this index:**

    ```bash
    gcloud firestore indexes composite create --collection-group=nutrition-and-cancer --query-scope=COLLECTION --field-config=vector-config='{\"dimension\":\"768\",\"flat\": \"{}\"}',field-path=embedding
    ```

    e. **Test the Vector Search**

    We call `search_vector_database` with our user query, which performs a vector search in Firestore to find the top k=5 nearest neighbors/chunks. Then use those chunks as context to generate an answer using the Gemini model.

    ```python
    query = "When eating eggs, what should I be aware of?"
    relevant_text = search_vector_database(query)
    print(f"### Related Content:\n\n{relevant_text}\n\n")

    chat_model = GenerativeModel("gemini-2.5-flash")  # Instantiate GenerativeModel directly
    chat = chat_model.start_chat()

    response = chat.send_message(
        f"Use the following to answer the question:\n\n{relevant_text}\n\nQuestion: {query}"
    )
    print(f"### Model Response:\n\n{response.text}")
    ```

    **Expected Output:**

    ```
    ### Related Content:

    23 Treatment side effects and nutrition Food type Safe action Precautions to take salad,  fruits and  vegetables •	 wash thoroughly  before preparing •	 refrigerate leftovers immediately, and  eat within 24 hours •	 avoid ready-to-eat or pre-packaged deli  salads (including pre-cut fruit salads  and roast vegetables) •	 pick unblemished fruits and vegetables eggs •	 keep uncracked,  clean eggs in fridge •	 cook until yolks and  whites are solid •	 avoid cracked, dirty and raw eggs •	 avoid food containing raw eggs  (e.g. homemade mayonnaise, raw cake  mix and biscuit dough) cheese and  other dairy  products •	 eat hard or  processed cheese •	 store cheese and  pasteurised dairy  products in fridge •	 avoid soft, semisoft and surfaceripened cheeses (e.g. camembert,  brie, ricotta, feta, blue) •	 avoid unpasteurised dairy products packaged  food •	 eat within use-by  dates •	 store unused perishable food in fridge  in clean, sealed containers, and use  within 24 hours of opening ice-cream •	 keep frozen •	 avoid soft serve ice-cream General precautions •	 Wash your hands and knives, cutting boards and food preparation  areas thoroughly with hot soapy water before and after cooking. •	 Take extra care when eating out.
    ...
    ...

    ### Model Response:

    When eating eggs, you should be aware of the following:

    *   **Safe actions:**
        *   Keep uncracked, clean eggs in the fridge.
        *   Cook until yolks and whites are solid.
    *   **Precautions to take:**
        *   Avoid cracked, dirty, and raw eggs.
        *   Avoid food containing raw eggs (e.g., homemade mayonnaise, raw cake mix, and biscuit dough).
    ```

## Running the Application Locally

1.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Flask app:**

    ```bash
    python3 main.py
    ```

3.  **Preview the app:**

    The app will be running on `http://localhost:8080`.

    **Expected Output:** You should see a web page with a chat interface. You can ask questions like "When eating eggs, what should I be aware of?" and get an answer from the nutrition for people living with cancer booklet. The expected answer should provide the safe actions and precautions.


## Deploying to Cloud Run

1.  **Build the Docker image:**

    ```bash
    export ARTIFACT_REPO="us-central1-docker.pkg.dev/$PROJECT_ID/nutrition-and-cancer-repo"
    gcloud auth configure-docker us-central1-docker.pkg.dev
    docker build -t $ARTIFACT_REPO/qna-image -f Dockerfile .
    ```

2.  **Push the image to Artifact Registry:**

    ```bash
    docker push $ARTIFACT_REPO/qna-image
    ```

3.  **Deploy to Cloud Run:**

    ```bash
    gcloud run deploy qna-service \
        --image $ARTIFACT_REPO/qna-image \
        --region us-central1 \
        --allow-unauthenticated
    ```

4.  **Test the deployed service:**

    Access the URL provided by the `gcloud run deploy` command. You can ask questions like "How to manage bowel irritation?".
