import logging
from llama_index.core.settings import Settings
from .config import config

logger = logging.getLogger(__name__)
try:
    from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS
    
    GPT5_MODELS = {
        "gpt-5-mini" : 128000,
        "gpt-5-mini-2025-08-07" : 256000,
    }
    ALL_AVAILABLE_MODELS.update(GPT5_MODELS)
    logger.info("✅ Added GPT-5 models")
    logger.info(f"Available: {list(GPT5_MODELS.keys())}")
except ImportError as e:
    logger.warning(f"❌ Could not import OpenAI utils to add GPT-5 models: {e}")
except Exception as e:
    logger.error(f"❌ Could not patch GPT-5 models: {e}")

try:
    import tiktoken.model
    tiktoken.model.MODEL_PREFIX_TO_ENCODING["gpt-5-"] = "o200k_base"

    GPT_TIKTOKEN_MODELS = {
        "gpt-5-mini": "o200k_base",
        "gpt-5-mini-2025-08-07": "o200k_base",
    }
    tiktoken.model.MODEL_TO_ENCODING.update(GPT_TIKTOKEN_MODELS)
    logger.info("✅ Added GPT-5 tokenizer mappings to tiktoken")
    logger.info(f"Specific models: {list(GPT_TIKTOKEN_MODELS.keys())}")
except Exception as e:
    logger.error(f"❌ Could not patch tiktoken for GPT-5 models: {e}")

# ========== PATCH 3: Monkey-patch OpenAI class metadata property ==========
try:
    from llama_index.llms.openai import OpenAI as OriginalOpenAI
    from llama_index.llms.openai.utils import openai_modelname_to_contextsize, is_chat_model, is_function_calling_model
    from llama_index.core.base.llms.types import LLMMetadata
    
    # Store the original metadata property
    _original_metadata_fget = OriginalOpenAI.metadata.fget
    
    def patched_metadata(self):
        """Patched metadata that correctly identifies GPT-5 as a chat model."""
        model_name = self._get_model_name()
        
        # Check if it's a GPT-5 model
        if model_name.startswith("gpt-5"):
            # Manually construct metadata for GPT-5
            return LLMMetadata(
                context_window=256000,  # GPT-5 context window
                num_output=4096,
                is_chat_model=True,  # ← Force True
                is_function_calling_model=True,  # ← Force True
                model_name=model_name,
            )
        
        # For other models, use original behavior
        return _original_metadata_fget(self)
    
    # Replace the metadata property
    OriginalOpenAI.metadata = property(patched_metadata)
    
    logger.info("✅ Patched OpenAI.metadata property for GPT-5")
    
except Exception as e:
    logger.error(f"❌ Failed to patch OpenAI class: {e}")
    import traceback
    logger.error(traceback.format_exc())

# ============================================================

class LLMManager:
    """Language Model Manager supporting Ollama or OpenAI based on config.
       Initializes lazily on first use to avoid network calls at import time.
    """

    def __init__(self):
        self.llm = None
        self.aux_llm = None
        self.embed_model = None 
        self._initialized = False

    def _initialize_models(self) -> None:
        """Initialize either Ollama or OpenAI clients per config."""
        if self._initialized:
            return
        try:
            # Initialize LLM based on LLM_BACKEND
            if config.LLM_BACKEND == "openai":
                from llama_index.llms.openai import OpenAI
                logger.info(f"Initializing OpenAI LLM {config.OPENAI_COMPLETION_MODEL}")
                model_name = config.OPENAI_COMPLETION_MODEL

                if "gpt-5" in model_name.lower():
                    logger.info(f"Detected GPT-5 model: {model_name}")
                    # Explicitly set context_window to bypass validation
                    self.llm = OpenAI(
                        api_key=config.OPENAI_API_KEY,
                        model=model_name,
                        request_timeout=config.OPENAI_REQUEST_TIMEOUT,
                        temperature=0.1,
                        #context_window=128000, 
                    )
                else:
                    # Normal initialization for GPT-4.1
                    self.llm = OpenAI(
                        api_key=config.OPENAI_API_KEY,
                        model=model_name,
                        request_timeout=config.OPENAI_REQUEST_TIMEOUT,
                        temperature=0.1,
                    )

            elif config.LLM_BACKEND == "gemini":
                from .custom_gemini import GeminiLLM
                logger.info("Initializing Gemini LLM")
                self.llm = GeminiLLM()
            else:
                from llama_index.llms.ollama import Ollama
                logger.info("Initializing Ollama LLM")
                self.llm = Ollama(
                    model=config.OLLAMA_LLM_MODEL,
                    request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
                    base_url=config.OLLAMA_HOST,
                    context_window=8192, 
                )
            logger.info(f"✅ Main LLM configured to use: {self.llm.model}")
            # --- Initialize Auxiliary LLM (for retrieval/selection) ---
            # CRITICAL FIX: Check if the specific models are the same, not just the backend.
            is_same_openai_model = (
                config.LLM_BACKEND == "openai" and 
                config.AUX_LLM_BACKEND == "openai" and 
                config.OPENAI_COMPLETION_MODEL == config.OPENAI_AUX_MODEL
            )
            is_same_ollama_model = (
                config.LLM_BACKEND == "ollama" and 
                config.AUX_LLM_BACKEND == "ollama" and 
                config.OLLAMA_LLM_MODEL == config.OLLAMA_AUX_LLM_MODEL
            )

            if is_same_openai_model or is_same_ollama_model:
                logger.info("Auxiliary LLM is the same as Main LLM. Re-using the object.")
                self.aux_llm = self.llm
            else:
                # If models or backends are different, initialize a new LLM object.
                if config.AUX_LLM_BACKEND == "openai":
                    from llama_index.llms.openai import OpenAI
                    logger.info(f"Initializing Auxiliary OpenAI LLM: {config.OPENAI_AUX_MODEL}")
                    self.aux_llm = OpenAI(
                        api_key=config.OPENAI_API_KEY,
                        model=config.OPENAI_AUX_MODEL,
                        request_timeout=config.OPENAI_REQUEST_TIMEOUT,
                        temperature=0.5,
                    )
                else: # Fallback to Ollama for aux
                    from llama_index.llms.ollama import Ollama
                    logger.info(f"Initializing Auxiliary Ollama LLM: {config.OLLAMA_AUX_LLM_MODEL}")
                    self.aux_llm = Ollama(
                        model=config.OLLAMA_AUX_LLM_MODEL,
                        request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
                        base_url=config.OLLAMA_HOST,
                        context_window=8192,
                    )
            logger.info(f"✅ Auxiliary LLM configured to use: {self.aux_llm.model}")

            # Initialize embedding model based on EMBEDDING_BACKEND
            if config.EMBEDDING_BACKEND == "openai":
                from llama_index.embeddings.openai import OpenAIEmbedding
                logger.info("Initializing OpenAI Embedding")
                self.embed_model = OpenAIEmbedding(
                    api_key=config.OPENAI_API_KEY,
                    model=config.OPENAI_EMBED_MODEL,
                    request_timeout=config.OPENAI_REQUEST_TIMEOUT,
                )
            else:
                from llama_index.embeddings.ollama import OllamaEmbedding
                logger.info("Initializing Ollama Embedding")
                self.embed_model = OllamaEmbedding(
                    model_name=config.OLLAMA_EMBED_MODEL,
                    base_url=config.OLLAMA_HOST,
                )

            # Register for llama_index
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model

            self._initialized = True
            logger.info("✅ LLM & Embedding initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing LLM backends: {e}")
            raise

    def _test_connection(self) -> None:
        """Simple sanity-check call to ensure LLM is responsive."""
        if not self._initialized:
            self._initialize_models()
        try:
            resp = self.llm.complete("Hello, are you there?")
            logger.debug(f"LLM healthcheck response: {resp}")
        except Exception as e:
            logger.error(f"LLM healthcheck failed: {e}")
            raise
            
    def get_llm(self):
        """Return the raw LLM instance for llama_index pipelines."""
        if not self._initialized:
            self._initialize_models()
        return self.llm

    def get_aux_llm(self):
        """Returns the auxiliary LLM instance (for table/column selection)."""
        if not self._initialized:
            self._initialize_models()
        return self.aux_llm
    
    def get_embed_model(self):
        """Return the embedding model instance."""
        if not self._initialized:
            self._initialize_models()
        return self.embed_model
    
llm_manager = LLMManager()
