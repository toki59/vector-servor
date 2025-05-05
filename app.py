from flask import Flask, request, jsonify, abort
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from dotenv import load_dotenv
import uuid
import os

# Load environment variables from .env
load_dotenv()

# Init Flask app
app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Security: protect with API token
API_TOKEN = os.getenv("VECTOR_API_KEY", "default-token")

def check_token():
    auth = request.headers.get("Authorization")
    if not auth or auth != f"Bearer {API_TOKEN}":
        abort(403)

# Default collection if not provided
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "default_agent")

# Init Qdrant
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

@app.route("/embed", methods=["POST"])
def embed():
    check_token()
    text = request.json.get("text")
    vector = model.encode(text).tolist()
    return jsonify({"vector": vector})

@app.route("/push", methods=["POST"])
def push():
    check_token()
    text = request.json.get("text")
    collection = request.json.get("collection", DEFAULT_COLLECTION)

    # Create collection if it doesn't exist
    collections = [c.name for c in qdrant.get_collections().collections]
    if collection not in collections:
        qdrant.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    vector = model.encode(text).tolist()
    point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": text})
    qdrant.upsert(collection_name=collection, points=[point])
    return jsonify({"status": "ok", "collection": collection})

@app.route("/search", methods=["POST"])
def search():
    check_token()
    text = request.json.get("text")
    collection = request.json.get("collection", DEFAULT_COLLECTION)
    vector = model.encode(text).tolist()
    results = qdrant.search(collection_name=collection, query_vector=vector, limit=3)
    return jsonify([r.payload["text"] for r in results])

@app.route("/", methods=["GET"])
def index():
    return "✅ Vector server is running securely."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

