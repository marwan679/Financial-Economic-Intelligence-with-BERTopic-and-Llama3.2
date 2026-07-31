"""
Abstractive Topic Summarization module using Hugging Face T5 models.
Compares t5-base vs t5-large for summarizing topically grouped passages.
"""

from typing import List, Dict, Union
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5SummarizationPipeline:
    """
    Abstractive summarizer for BERTopic topic clusters using T5.
    
    Parameters
    ----------
    model_name : str, default="t5-base"
        Hugging Face T5 model checkpoint ('t5-small', 't5-base', 't5-large').
    device : str, optional
        Target device ('cuda', 'cpu'). Inferred automatically if None.
    max_length : int, default=150
        Maximum token length of generated abstractive summary.
    min_length : int, default=30
        Minimum token length of summary.
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        device: Optional[str] = None,
        max_length: int = 150,
        min_length: int = 30,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.min_length = min_length

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def summarize(self, text: str) -> str:
        """
        Summarize a single text string using T5.
        """
        input_text = "summarize: " + text.strip()
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs,
            max_length=self.max_length,
            min_length=self.min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize_topics(self, topic_documents: Dict[int, List[str]]) -> Dict[int, str]:
        """
        Generate abstractive summaries for documents belonging to each topic ID.
        """
        topic_summaries = {}
        for topic_id, docs in topic_documents.items():
            if topic_id == -1 or not docs:
                continue
            combined_text = " ".join(docs[:10])
            topic_summaries[topic_id] = self.summarize(combined_text)
        return topic_summaries
