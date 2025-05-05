from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid
import os

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Configuration par défaut si collection non précisée dans les requêtes
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "default_agent")

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

@app.route("/embed", methods=["POST"])
def embed():
    text = request.json.get("text")
    vector = model.encode(text).tolist()
    return jsonify({"vector": vector})

@app.route("/push", methods=["POST"])
def push():
    text = request.json.get("text")
    collection = request.json.get("collection", DEFAULT_COLLECTION)

    # Création de collection si elle n'existe pas
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
    text = request.json.get("text")
    collection = request.json.get("collection", DEFAULT_COLLECTION)
    vector = model.encode(text).tolist()
    results = qdrant.search(collection_name=collection, query_vector=vector, limit=3)
    return jsonify([r.payload["text"] for r in results])

@app.route("/", methods=["GET"])
def index():
    return "✅ Vector server is running."
