import os
from typing import Any, Sequence, Generator

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    ChatResponseGen,
    CompletionResponseGen,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
import google.generativeai as genai
from pydantic import PrivateAttr # ADD THIS IMPORT

# Import the global config object
from .config import config

class GeminiLLM(CustomLLM):
    """
    A robust, custom LlamaIndex LLM wrapper for Google's Gemini.
    """
    # --- DECLARE PRIVATE ATTRIBUTES FOR PYDANTIC ---
    _model: Any = PrivateAttr()
    _request_options: dict = PrivateAttr()
    _model_name: str = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        # We call super() first, then set our private attributes
        super().__init__(**kwargs)
        
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in your environment or config.")
        
        genai.configure(api_key=config.GEMINI_API_KEY)

        generation_config = {
            "temperature": 0.7,
        }

        # Use the private attributes with a leading underscore
        self._request_options = {
            "timeout": config.GEMINI_REQUEST_TIMEOUT,
        }
        self._model_name = config.GEMINI_LLM_MODEL
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=generation_config,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            is_chat_model=True,
            model_name=self._model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        try:
            # Use the private attributes here
            response = self._model.generate_content(
                prompt, request_options=self._request_options
            )
            return CompletionResponse(text=response.text, raw=response)
        except Exception as e:
            return CompletionResponse(text=f"Error: Gemini API call failed. {e}")

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        google_messages = [{"role": m.role.value, "parts": [m.content]} for m in messages]
        
        if google_messages and google_messages[-1]['role'] != 'user':
             google_messages.append({"role": "user", "parts": ["Continue."]})
        
        try:
            # Use the private attributes here
            response = self._model.generate_content(
                google_messages, request_options=self._request_options
            )
            return ChatResponse(
                message=ChatMessage(role="assistant", content=response.text), raw=response
            )
        except Exception as e:
            return ChatResponse(
                message=ChatMessage(role="assistant", content=f"Error: Gemini API call failed. {e}")
            )

    # --- STREAMING METHODS (UNCHANGED) ---
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response_stream = self._model.generate_content(
            prompt, stream=True, request_options=self._request_options
        )
        def gen() -> Generator[CompletionResponse, None, None]:
            text = ""
            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    text += chunk.text
                    yield CompletionResponse(text=text, delta=chunk.text)
        return gen()

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        google_messages = [{"role": m.role.value, "parts": [m.content]} for m in messages]
        if google_messages and google_messages[-1]['role'] != 'user':
             google_messages.append({"role": "user", "parts": ["Continue."]})

        response_stream = self._model.generate_content(
            google_messages, stream=True, request_options=self._request_options
        )
        def gen() -> Generator[ChatResponse, None, None]:
            content = ""
            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    content += chunk.text
                    yield ChatResponse(
                        message=ChatMessage(role="assistant", content=content),
                        delta=chunk.text,
                    )
        return gen()

    # --- ASYNC METHODS (UNCHANGED) ---
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)