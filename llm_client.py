"""
InteractMD — Multi-Provider LLM & RAG Generation Client.
Supports OpenAI, Google Gemini, Groq, Anthropic Claude, and Local Clinical RAG Engine.
"""

import os
import httpx
from typing import Dict, Any, List, Optional
from config import settings


class LLMClient:
    def __init__(self):
        self.openai_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.gemini_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.groq_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        self.anthropic_key = settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "")
        self.hf_token = settings.HUGGINGFACE_API_KEY or settings.HF_TOKEN or os.getenv("HUGGINGFACE_API_KEY", "") or os.getenv("HF_TOKEN", "")

    async def generate_response(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        retrieved_context: str
    ) -> Optional[str]:
        """
        Attempts generation using active cloud LLM providers in priority order.
        Returns None if no external API key is configured, allowing instant local RAG fallback.
        """
        # 1. HuggingFace Serverless Inference (Meta-Llama-3 / Mistral)
        if self.hf_token:
            try:
                res = await self._call_huggingface(system_prompt, user_message, conversation_history, retrieved_context)
                if res:
                    return res
            except Exception as e:
                print(f"[HuggingFace LLM Warning] {e}")

        # 2. Groq (Ultra-fast Llama 3)
        if self.groq_key:
            try:
                res = await self._call_groq(system_prompt, user_message, conversation_history, retrieved_context)
                if res:
                    return res
            except Exception as e:
                print(f"[Groq LLM Warning] {e}")

        # 3. OpenAI (GPT-4o / GPT-4o-mini)
        if self.openai_key:
            try:
                res = await self._call_openai(system_prompt, user_message, conversation_history, retrieved_context)
                if res:
                    return res
            except Exception as e:
                print(f"[OpenAI LLM Warning] {e}")

        # 4. Google Gemini
        if self.gemini_key:
            try:
                res = await self._call_gemini(system_prompt, user_message, conversation_history, retrieved_context)
                if res:
                    return res
            except Exception as e:
                print(f"[Gemini LLM Warning] {e}")

        return None

    async def _call_huggingface(self, system_prompt: str, user_message: str, history: List[Dict[str, Any]], context: str) -> Optional[str]:
        messages = [{"role": "system", "content": f"{system_prompt}\n\nRETRIEVED CLINICAL GROUNDING EVIDENCE:\n{context}"}]
        for h in history[-4:]:
            role = "user" if h.get("sender") in ["student", "doctor", "LEARNER", "user"] else "assistant"
            messages.append({"role": role, "content": h.get("text", "")})
        messages.append({"role": "user", "content": user_message})

        endpoints = [
            "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-8B-Instruct/v1/chat/completions",
            "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions",
            "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct/v1/chat/completions"
        ]

        async with httpx.AsyncClient(timeout=8.0) as client:
            for ep in endpoints:
                try:
                    res = await client.post(
                        ep,
                        headers={"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"},
                        json={
                            "messages": messages,
                            "temperature": 0.2,
                            "max_tokens": 120
                        }
                    )
                    if res.status_code == 200:
                        data = res.json()
                        if "choices" in data and len(data["choices"]) > 0:
                            return data["choices"][0]["message"]["content"].strip()
                except Exception:
                    continue
        return None

    async def _call_openai(self, system_prompt: str, user_message: str, history: List[Dict[str, Any]], context: str) -> Optional[str]:
        messages = [{"role": "system", "content": f"{system_prompt}\n\nRETRIEVED CLINICAL GROUNDING EVIDENCE:\n{context}"}]
        for h in history[-6:]:
            role = "user" if h.get("sender") in ["student", "doctor", "LEARNER", "user"] else "assistant"
            messages.append({"role": role, "content": h.get("text", "")})
        messages.append({"role": "user", "content": user_message})

        async with httpx.AsyncClient(timeout=6.0) as client:
            res = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 120
                }
            )
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()
        return None

    async def _call_groq(self, system_prompt: str, user_message: str, history: List[Dict[str, Any]], context: str) -> Optional[str]:
        messages = [{"role": "system", "content": f"{system_prompt}\n\nRETRIEVED CLINICAL GROUNDING EVIDENCE:\n{context}"}]
        for h in history[-6:]:
            role = "user" if h.get("sender") in ["student", "doctor", "LEARNER", "user"] else "assistant"
            messages.append({"role": role, "content": h.get("text", "")})
        messages.append({"role": "user", "content": user_message})

        async with httpx.AsyncClient(timeout=4.0) as client:
            res = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 120
                }
            )
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()
        return None

    async def _call_gemini(self, system_prompt: str, user_message: str, history: List[Dict[str, Any]], context: str) -> Optional[str]:
        prompt = f"{system_prompt}\n\nCLINICAL CONTEXT:\n{context}\n\nDOCTOR'S QUESTION:\n{user_message}"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 100}
                }
            )
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    return candidates[0]["content"]["parts"][0]["text"].strip()
        return None


llm_client = LLMClient()
