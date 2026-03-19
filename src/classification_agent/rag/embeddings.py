from typing import List
from abc import ABC, abstractmethod
from openai import OpenAI


class BaseEmbeddings(ABC):
    """Base interface for text embeddings"""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string"""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings in batch"""
        pass


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embeddings client for text embedding"""

    def __init__(
        self,
        client: OpenAI,
        model: str = "text-embedding-3-small",
    ):
        self.client = client
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings in batch"""
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        if not response.data:
            return []
        return [data.embedding for data in response.data if data and hasattr(data, 'embedding')]


class BGEEmbeddings(BaseEmbeddings):
    """Local BGE embedding model (BAAI/bge-large-zh-v1.5) via HuggingFace transformers

    Runs locally on your machine, no API calls needed. Good for Chinese text.
    Requires: pip install transformers torch
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cpu",
        use_fp16: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name or path to local model
                Default: "BAAI/bge-large-zh-v1.5"
            device: Device to run on ("cpu", "cuda", "cuda:0", etc.)
            use_fp16: Use half precision for faster inference
        """
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "BGEEmbeddings requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype="auto" if use_fp16 else None,
        ).to(device)

    @staticmethod
    def _average_pool(last_hidden_state, attention_mask):
        """Average pooling for BGE embeddings"""
        import torch
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string with BGE instruction prefix for retrieval"""
        # BGE uses "为这个句子生成表示以用于检索相关文章：" as instruction for retrieval
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings in batch"""
        import torch
        if not texts:
            return []

        # Add BGE instruction prefix for retrieval
        # Reference: https://huggingface.co/BAAI/bge-large-zh-v1.5
        instruction = "为这个句子生成表示以用于检索相关文章："
        texts_with_instruction = [f"{instruction}{text}" for text in texts]

        # Tokenize
        encoded_input = self.tokenizer(
            texts_with_instruction,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)

        # Inference
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self._average_pool(
                model_output.last_hidden_state,
                encoded_input["attention_mask"]
            )
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list[list[float]]
        return embeddings.cpu().tolist()
