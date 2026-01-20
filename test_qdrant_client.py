#!/usr/bin/env python3
"""Test LatticeDB with Qdrant Python client (v1.16+)."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Connect to LatticeDB (Qdrant-compatible)
client = QdrantClient(host="localhost", port=6333, check_compatibility=False)

COLLECTION_NAME = "test_collection"
VECTOR_DIM = 128

print("=" * 50)
print("Testing LatticeDB with Qdrant Python Client")
print("=" * 50)

# 1. List collections
print("\n1. Listing collections...")
collections = client.get_collections()
print(f"   Collections: {[c.name for c in collections.collections]}")

# 2. Delete test collection if exists
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"   Cleaned up existing '{COLLECTION_NAME}'")
except:
    pass

# 3. Create collection
print(f"\n2. Creating collection '{COLLECTION_NAME}'...")
result = client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
)
print(f"   Result: {result}")

# 4. Get collection info
print(f"\n3. Getting collection info...")
info = client.get_collection(collection_name=COLLECTION_NAME)
print(f"   Status: {info.status}")
print(f"   Points count: {info.points_count}")
print(f"   Vector size: {info.config.params.vectors.size}")

# 5. Upsert points
print("\n4. Upserting 10 points...")
import random
random.seed(42)

points = [
    PointStruct(
        id=i,
        vector=[random.random() for _ in range(VECTOR_DIM)],
        payload={"category": f"cat_{i % 3}", "value": i * 10}
    )
    for i in range(10)
]

client.upsert(collection_name=COLLECTION_NAME, points=points)
print("   Upserted 10 points!")

# 6. Get collection info again
print("\n5. Checking point count...")
info = client.get_collection(collection_name=COLLECTION_NAME)
print(f"   Points count: {info.points_count}")

# 7. Search using query_points (v1.16+ API)
print("\n6. Searching for nearest neighbors...")
query_vector = [random.random() for _ in range(VECTOR_DIM)]
response = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5,
    with_payload=True,
)
print("   Top 5 results:")
for r in response.points:
    print(f"     ID: {r.id}, Score: {r.score:.4f}, Payload: {r.payload}")

# 8. Retrieve specific points
print("\n7. Retrieving points by ID...")
retrieved = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids=[0, 1, 2],
    with_payload=True,
    with_vectors=False,
)
print(f"   Retrieved {len(retrieved)} points:")
for p in retrieved:
    print(f"     ID: {p.id}, Payload: {p.payload}")

# 9. Scroll through all points
print("\n8. Scrolling through points...")
scroll_result = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=5,
    with_payload=True,
    with_vectors=False,
)
points_page, next_offset = scroll_result
print(f"   First page: {len(points_page)} points, next_offset: {next_offset}")

# 10. Delete points
print("\n9. Deleting points [8, 9]...")
client.delete(
    collection_name=COLLECTION_NAME,
    points_selector=[8, 9],
)
info = client.get_collection(collection_name=COLLECTION_NAME)
print(f"   Points count after delete: {info.points_count}")

# 11. Delete collection
print(f"\n10. Deleting collection '{COLLECTION_NAME}'...")
client.delete_collection(collection_name=COLLECTION_NAME)
print("   Collection deleted!")

# Verify deletion
print("\n11. Verifying deletion...")
collections = client.get_collections()
print(f"   Collections: {[c.name for c in collections.collections]}")

print("\n" + "=" * 50)
print("All tests completed successfully!")
print("=" * 50)
