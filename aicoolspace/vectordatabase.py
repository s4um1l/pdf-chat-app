import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aicoolspace.openai_utils.embedding import EmbeddingModel
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    distance = -np.linalg.norm(vector_a - vector_b)
    return distance


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array, metadata: Dict[str, Any] = None) -> None:
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        scores = []
        for key, vector in self.vectors.items():
            # Skip if metadata filter doesn't match
            if metadata_filter and not self._matches_metadata_filter(key, metadata_filter):
                continue
                
            scores.append((key, distance_measure(query_vector, vector)))
            
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    
    def _matches_metadata_filter(self, key: str, metadata_filter: Dict[str, Any]) -> bool:
        """Check if document metadata matches the filter criteria"""
        doc_metadata = self.metadata.get(key, {})
        
        for filter_key, filter_value in metadata_filter.items():
            if filter_key not in doc_metadata or doc_metadata[filter_key] != filter_value:
                return False
        return True

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, metadata_filter)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> Tuple[np.array, Dict[str, Any]]:
        return self.vectors.get(key, None), self.metadata.get(key, {})

    async def abuild_from_list(
        self, 
        list_of_text: List[str], 
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        
        for idx, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            metadata = metadata_list[idx] if metadata_list and idx < len(metadata_list) else None
            self.insert(text, np.array(embedding), metadata)
            
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
