from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid
import os

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION = os.getenv("QDRANT_COLLECTION", "agent_voidorder")

@app.before_first_request
def setup():
    if COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

@app.route("/embed", methods=["POST"])
def embed():
    text = request.json.get("text")
    vec = model.encode(text).tolist()
    return jsonify({ "vector": vec })

@app.route("/push", methods=["POST"])
def push():
    text = request.json.get("text")
    vec = model.encode(text).tolist()
    point = PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"text": text})
    qdrant.upsert(collection_name=COLLECTION, points=[point])
    return jsonify({"status": "ok"})

@app.route("/search", methods=["POST"])
def search():
    text = request.json.get("text")
    vec = model.encode(text).tolist()
    results = qdrant.search(collection_name=COLLECTION, query_vector=vec, limit=3)
    return jsonify([r.payload["text"] for r in results])

@app.route("/", methods=["GET"])
def index():
    return "Vector server is running."
