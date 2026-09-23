"""
InteractMD — LLM Provider Base Interface.
Abstract base class defining the contract for AI text and chat generation providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class LLMProvider(ABC):

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the identifier name of this LLM provider."""
        pass

    @abstractmethod
    async def generate_chat(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Asynchronously generates a conversational patient dialogue response.
        Returns the clean response string or None if generation failed.
        """
        pass
