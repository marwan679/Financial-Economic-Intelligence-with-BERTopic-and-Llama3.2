"""
KeyBERT-inspired term reranking for BERTopic representation refinement.
Reranks c-TF-IDF keyword candidates using cosine similarity to cluster centroid embeddings.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class KeyBERTReranker:
    """
    Reranks topic terms using sentence-transformer embeddings to maximize semantic similarity
    between terms and the centroid of representative documents in a cluster.

    Parameters
    ----------
    embedding_model_name : str, default="thenlper/gte-small"
        SentenceTransformer model used for embedding calculation.
    top_n : int, default=10
        Number of top terms to return after reranking.
    """

    def __init__(
        self,
        embedding_model_name: str = "thenlper/gte-small",
        top_n: int = 10,
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.top_n = top_n

    def rerank(
        self,
        candidate_keywords: List[str],
        representative_docs: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidate keywords against cluster representative documents.
        """
        if not candidate_keywords or not representative_docs:
            return [(kw, 1.0) for kw in candidate_keywords[:self.top_n]]

        # Compute centroid embedding of representative docs
        doc_embeddings = self.embedding_model.encode(representative_docs, show_progress_bar=False)
        centroid = np.mean(doc_embeddings, axis=0, keepdims=True)

        # Compute candidate term embeddings
        term_embeddings = self.embedding_model.encode(candidate_keywords, show_progress_bar=False)

        # Cosine similarity between terms and centroid
        similarities = cosine_similarity(term_embeddings, centroid).flatten()

        # Sort terms by similarity score
        ranked_indices = np.argsort(similarities)[::-1]
        reranked = [
            (candidate_keywords[i], float(similarities[i]))
            for i in ranked_indices[:self.top_n]
        ]
        return reranked
