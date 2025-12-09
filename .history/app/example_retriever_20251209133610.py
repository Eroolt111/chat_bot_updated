import json
import logging
from pathlib import Path
from typing import List, Dict

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .llm import llm_manager
from .config import config

logger = logging.getLogger(__name__)

class ExampleRetriever:
    """
    Loads, indexes, and retrieves relevant query/SQL example pairs
    to provide dynamic few-shot examples for the text-to-SQL LLM.
    """

    def __init__(self, examples_file: str = "app/sql_examples.json"):
        self.examples: List[Dict[str, str]] = []
        self.index: VectorStoreIndex = None
        self._load_and_index_examples(examples_file)

    def _load_and_index_examples(self, examples_file: str):
        """Loads examples from a JSON file and builds a vector index."""
        try:
            examples_path = Path(examples_file)
            if not examples_path.exists():
                logger.warning(f"Examples file not found at {examples_file}. Dynamic examples will be disabled.")
                return

            with open(examples_path, 'r', encoding='utf-8') as f:
                self.examples = json.load(f)

            if not self.examples:
                logger.warning("No examples found in the file. Dynamic examples will be disabled.")
                return

            # Create documents from the 'user_question' field for indexing
            documents = [Document(text=ex["user_question"], metadata={"sql_query": ex["sql_query"]}) for ex in self.examples]

            # Get the embedding model from our manager
            embed_model = llm_manager.get_embed_model()

            # Create an in-memory vector index
            self.index = VectorStoreIndex(documents, embed_model=embed_model)
            logger.info(f"✅ Successfully indexed {len(self.examples)} SQL examples.")

        except Exception as e:
            logger.error(f"❌ Failed to load or index SQL examples: {e}", exc_info=True)
            self.index = None
        
    def retrieve_examples(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        Retrieves the most similar user_question/sql_query pairs from the index.
        """
        if not self.index:
            return []

        retriever = self.index.as_retriever(similarity_top_k=top_k)
        retrieved_nodes: List[NodeWithScore] = retriever.retrieve(query)

        retrieved_examples = []
        for node_with_score in retrieved_nodes:
            retrieved_examples.append({
                "user_question": node_with_score.node.get_content(),
                "sql_query": node_with_score.node.metadata["sql_query"]
            })

        logger.info(f"Retrieved {len(retrieved_examples)} dynamic examples for the query.")
        print("Question for retrieval:" , query)
        return retrieved_examples

# Create a single instance to be used across the application
example_retriever = ExampleRetriever()