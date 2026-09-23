"""
InteractMD — Hugging Face LLM Provider.
Integrates with Hugging Face Inference API / Serverless Router using multi-provider inference.
"""

import os
import asyncio
import httpx
from typing import Dict, Any, List, Optional
from huggingface_hub import InferenceClient

from config import settings
from providers.llm_provider import LLMProvider


class HuggingFaceProvider(LLMProvider):

    def __init__(self):
        self.token = settings.HF_TOKEN or settings.HUGGINGFACE_API_KEY or os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_API_KEY", "")
        self.model = settings.HF_MODEL or os.getenv("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
        self.custom_api_url = settings.HF_API_URL or os.getenv("HF_API_URL", "")
        self.temperature = settings.HF_TEMPERATURE
        self.max_new_tokens = settings.HF_MAX_NEW_TOKENS
        self.top_p = settings.HF_TOP_P
        self.repetition_penalty = settings.HF_REPETITION_PENALTY

    @property
    def provider_name(self) -> str:
        return f"HuggingFace ({self.model})"

    @property
    def is_configured(self) -> bool:
        return bool(self.token and len(self.token.strip()) > 0)

    def _generate_sync(
        self,
        messages: List[Dict[str, str]],
        temp: float,
        max_tokens: int
    ) -> Optional[str]:
        """Synchronous generation using multi-provider InferenceClient."""
        clean_token = self.token.strip()
        providers = ["featherless-ai", "novita", "nscale", "together"]
        
        for provider in providers:
            try:
                client = InferenceClient(provider=provider, token=clean_token, timeout=12.0)
                res = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_p=self.top_p
                )
                if res and res.choices and len(res.choices) > 0:
                    text = res.choices[0].message.content
                    if text and len(text.strip()) > 0:
                        return text.strip()
            except Exception as e:
                # Try next provider
                continue

        # Fallback to direct HTTP inference router
        try:
            url = f"https://router.huggingface.co/hf-inference/models/{self.model}/v1/chat/completions"
            headers = {"Authorization": f"Bearer {clean_token}", "Content-Type": "application/json"}
            payload = {"model": self.model, "messages": messages, "temperature": temp, "max_tokens": max_tokens}
            with httpx.Client(timeout=8.0) as http_client:
                r = http_client.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    data = r.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        return data["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

        return None

    async def generate_chat(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> Optional[str]:
        if not self.is_configured:
            return None

        temp = temperature if temperature is not None else self.temperature
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        # Construct Chat Messages payload
        messages = [{"role": "system", "content": system_prompt}]

        # Include recent conversation turns (up to last 4)
        for turn in conversation_history[-4:]:
            sender = str(turn.get("sender") or turn.get("role", "")).lower()
            role = "user" if sender in ["learner", "student", "doctor", "user"] else "assistant"
            content = turn.get("text") or turn.get("content") or turn.get("message", "")
            if content:
                messages.append({"role": role, "content": str(content)})

        # Append current user question
        messages.append({"role": "user", "content": user_message})

        # Run non-blocking in thread pool
        try:
            return await asyncio.to_thread(self._generate_sync, messages, temp, max_tokens)
        except Exception as e:
            print(f"[HuggingFace LLM Error] {e}")
            return None

