"""
Sovereign Intelligence: Main BERTopic-powered RAG Engine with Topic-Boundary Chunking.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from tqdm import tqdm

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document

from skills.ollama_representation import OllamaRepresentation


class BERTopicRAG:
    """
    Topic-Coherent RAG system powered by BERTopic topic discovery,
    vectorized boundary chunking, FAISS index, and local Ollama LLMs.
    """

    def __init__(
        self,
        ollama_model: str = "mistral:latest",
        embedding_model_name: str = "thenlper/gte-small",
        min_cluster_size: int = 150,
        n_umap_components: int = 5,
        top_k_retrieval: int = 4,
        verbose: bool = True,
    ):
        self.ollama_model = ollama_model
        self.embedding_model_name = embedding_model_name
        self.min_cluster_size = min_cluster_size
        self.n_umap_components = n_umap_components
        self.top_k_retrieval = top_k_retrieval
        self.verbose = verbose

        self.sentence_model = SentenceTransformer(embedding_model_name)
        self.topic_model: Optional[BERTopic] = None
        self.vectorstore: Optional[FAISS] = None
        self.last_retrieved_context: Optional[str] = None

        self.llm = ChatOllama(model=ollama_model, temperature=0.2)
        self.embeddings = OllamaEmbeddings(model=ollama_model)

    def log(self, msg: str):
        if self.verbose:
            print(f"[BERTopicRAG] {msg}")

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load multi-format documents (PDF, CSV, DOCX, TXT) via LangChain loaders.
        """
        raw_docs = []
        for path in file_paths:
            ext = os.path.splitext(path)[-1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext == ".csv":
                loader = CSVLoader(path)
            elif ext in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(path)
            elif ext in [".txt", ".md"]:
                loader = TextLoader(path)
            else:
                self.log(f"Skipping unsupported file extension: {path}")
                continue

            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = path
            raw_docs.extend(loaded)
        return raw_docs

    def load_from_hf_dataset(self, dataset: Any, text_fields: List[str]) -> List[str]:
        """
        Extract text fields from HuggingFace dataset items into raw document strings.
        """
        texts = []
        for sample in dataset:
            combined = []
            for field in text_fields:
                val = sample.get(field)
                if isinstance(val, list):
                    for turn in val:
                        if isinstance(turn, dict) and "value" in turn:
                            combined.append(turn["value"])
                        elif isinstance(turn, str):
                            combined.append(turn)
                elif isinstance(val, str):
                    combined.append(val)
            if combined:
                texts.append("\n".join(combined))
        return texts

    def split_into_sentences(self, raw_texts: List[str], min_length: int = 40) -> Tuple[List[str], List[int]]:
        """
        Regex sentence splitting with min-length filtering.
        Returns extracted sentences and their corresponding parent document index.
        """
        sentences = []
        doc_indices = []

        sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')

        for idx, text in enumerate(raw_texts):
            raw_sents = sentence_pattern.split(text)
            for s in raw_sents:
                clean_s = s.strip()
                if len(clean_s) >= min_length:
                    sentences.append(clean_s)
                    doc_indices.append(idx)

        return sentences, doc_indices

    def fit_bertopic(self, sentences: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Fit BERTopic model over sentences using UMAP + HDBSCAN + OllamaRepresentation.
        """
        self.log(f"Encoding {len(sentences)} sentences using {self.embedding_model_name}...")
        embeddings = self.sentence_model.encode(sentences, show_progress_bar=self.verbose)

        self.log("Initializing UMAP & HDBSCAN...")
        umap_model = UMAP(
            n_components=self.n_umap_components,
            metric="cosine",
            min_dist=0.0,
            random_state=42
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )

        representation_model = OllamaRepresentation(model=self.ollama_model)

        self.log("Fitting BERTopic...")
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            representation_model=representation_model,
            verbose=self.verbose
        )

        topics, probs = self.topic_model.fit_transform(sentences, embeddings)
        return topics, embeddings

    def chunk_by_topic_boundaries(
        self,
        sentences: List[str],
        doc_indices: List[int],
        topics: List[int]
    ) -> List[Document]:
        """
        Vectorized NumPy topic boundary chunking: merges consecutive same-topic sentences
        within the same parent document into topically coherent Document objects.
        """
        self.log("Performing vectorized topic boundary detection...")
        topics_arr = np.array(topics)
        doc_idx = np.array(doc_indices)

        topic_change = np.zeros(len(topics_arr), dtype=bool)
        doc_change = np.zeros(len(doc_idx), dtype=bool)

        topic_change[1:] = (topics_arr[1:] != topics_arr[:-1]) & (topics_arr[1:] != -1)
        doc_change[1:] = doc_idx[1:] != doc_idx[:-1]

        boundary_positions = np.where(topic_change | doc_change)[0]
        splits = np.split(np.arange(len(sentences)), boundary_positions)

        topic_info = self.topic_model.get_topic_info()
        topic_name_map = dict(zip(topic_info['Topic'], topic_info.get('Name', topic_info['Topic'])))

        chunks = []
        for idx_group in splits:
            if len(idx_group) == 0:
                continue
            chunk_sents = [sentences[i] for i in idx_group]
            chunk_topic = topics[idx_group[0]]
            chunk_doc_idx = doc_indices[idx_group[0]]
            chunk_text = " ".join(chunk_sents)

            topic_label = topic_name_map.get(chunk_topic, f"Topic_{chunk_topic}")

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "topic_id": int(chunk_topic),
                    "topic_label": str(topic_label),
                    "doc_index": int(chunk_doc_idx),
                    "n_sentences": len(chunk_sents)
                }
            )
            chunks.append(doc)

        self.log(f"Created {len(chunks)} topic-coherent chunks from {len(sentences)} sentences.")
        return chunks

    def create_vectorstore(self, chunks: List[Document]):
        """
        Build FAISS vector store over topic-coherent chunks.
        """
        self.log("Indexing topic-coherent chunks into FAISS vector store...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.log("FAISS index build complete.")

    def ingest(self, file_paths: List[str]):
        """
        Full ingestion pipeline: load documents -> sentence split -> BERTopic -> boundary chunk -> FAISS.
        """
        raw_docs = self.load_documents(file_paths)
        raw_texts = [d.page_content for d in raw_docs]

        sentences, doc_indices = self.split_into_sentences(raw_texts)
        topics, _ = self.fit_bertopic(sentences)
        chunks = self.chunk_by_topic_boundaries(sentences, doc_indices, topics)
        self.create_vectorstore(chunks)

    def get_topic_summary(self) -> pd.DataFrame:
        """
        Return topic summary table from fitted BERTopic model.
        """
        if self.topic_model is None:
            raise ValueError("BERTopic model has not been fitted yet.")
        return self.topic_model.get_topic_info()

    def query(self, question: str) -> str:
        """
        Retrieve top-k topic-coherent chunks and generate an answer using local Ollama LLM.
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore is not initialized. Run ingest() or load_index() first.")

        retrieved_docs = self.vectorstore.similarity_search(question, k=self.top_k_retrieval)

        context_blocks = []
        for i, doc in enumerate(retrieved_docs, 1):
            t_id = doc.metadata.get("topic_id", "N/A")
            t_label = doc.metadata.get("topic_label", "N/A")
            context_blocks.append(
                f"[Chunk {i} | Topic {t_id}: {t_label}]\n{doc.page_content}"
            )

        context_str = "\n\n".join(context_blocks)
        self.last_retrieved_context = context_str

        prompt = (
            "You are Sovereign Intelligence, a privacy-first document discovery assistant.\n"
            "Use ONLY the topically coherent document context below to answer the user's question.\n"
            "If the context does not contain enough information to answer, state clearly that the document collection does not cover it.\n\n"
            f"--- CONTEXT ---\n{context_str}\n---------------\n\n"
            f"Question: {question}\nAnswer:"
        )

        response = self.llm.invoke(prompt).content
        return response.strip()

    def get_last_retrieved_context(self) -> Optional[str]:
        """
        Retrieve context block passed to LLM in the last query execution.
        """
        return self.last_retrieved_context

    def save_index(self, path: str):
        """
        Save FAISS index to disk.
        """
        if self.vectorstore:
            self.vectorstore.save_local(path)
            self.log(f"Saved FAISS index to {path}")

    def load_index(self, path: str):
        """
        Load FAISS index from disk.
        """
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        self.log(f"Loaded FAISS index from {path}")
