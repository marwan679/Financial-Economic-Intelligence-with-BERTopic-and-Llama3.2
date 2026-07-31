"""
Custom BERTopic Representation module powered by local Ollama LLMs.
Generates concise, human-readable labels for discovered topic clusters.
"""

from typing import List, Dict, Tuple, Any, Optional
import requests
from bertopic.representation._base import BaseRepresentation


class OllamaRepresentation(BaseRepresentation):
    """
    BERTopic representation model using local Ollama instance for topic labeling.
    
    Parameters
    ----------
    model : str, default="mistral:latest"
        Name of the Ollama model running locally (e.g. 'mistral', 'llama3.2', 'llama3.1').
    ollama_url : str, default="http://localhost:11434/api/generate"
        API endpoint of local Ollama server.
    prompt_template : str, optional
        Custom prompt template containing {keywords} and {documents} placeholders.
    temperature : float, default=0.2
        Sampling temperature for LLM generation.
    """

    DEFAULT_PROMPT = (
        "You are an expert topic analyst. Below is a list of top keywords and representative documents for a topic cluster.\n\n"
        "Keywords: {keywords}\n"
        "Representative Documents:\n{documents}\n\n"
        "Based on the keywords and representative documents above, provide a concise, high-level topic label (3-6 words maximum).\n"
        "Return ONLY the topic label without quotes, formatting, preamble, or conversational filler.\n"
        "Topic Label:"
    )

    def __init__(
        self,
        model: str = "mistral:latest",
        ollama_url: str = "http://localhost:11434/api/generate",
        prompt_template: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.temperature = temperature

    def extract_topics(
        self,
        topic_model: Any,
        documents: Dict[int, List[str]],
        c_tf_idf: Any,
        topics: Dict[int, List[Tuple[str, float]]]
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract topics using local Ollama LLM.
        """
        updated_topics = {}

        for topic, keywords_scores in topics.items():
            if topic == -1:
                updated_topics[-1] = [("Outlier / Noise", 1.0)]
                continue

            keywords_str = ", ".join([kw for kw, _ in keywords_scores[:10]])
            rep_docs = documents.get(topic, [])[:3]
            docs_str = "\n".join([f"- {doc}" for doc in rep_docs])

            prompt = self.prompt_template.format(
                keywords=keywords_str,
                documents=docs_str
            )

            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": self.temperature}
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    label = response.json().get("response", "").strip()
                    updated_topics[topic] = [(label, 1.0)]
                else:
                    updated_topics[topic] = keywords_scores
            except Exception as e:
                # Fallback to default keyword representation on error
                updated_topics[topic] = keywords_scores

        return updated_topics
