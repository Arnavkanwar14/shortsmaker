"""Swappable LLM client: Ollama (local), Groq free tier, Gemini free tier,
or none. Provider "auto" picks the first reachable one in that order.

Every caller must tolerate complete() returning "" -- the pipeline has
heuristic/template fallbacks so no LLM is strictly required.
"""
from __future__ import annotations

import json
import logging
import re

import requests

from .config import Config, gemini_api_key, groq_api_key

log = logging.getLogger("shortsmaker")


def _ollama_reachable(cfg: Config) -> bool:
    try:
        return requests.get(f"{cfg.ollama_url}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


def provider_available(cfg: Config) -> str:
    """Return the usable provider name, or '' if none is reachable."""
    p = cfg.llm_provider
    if p == "none":
        return ""
    if p == "groq":
        return "groq" if groq_api_key() else ""
    if p == "gemini":
        return "gemini" if gemini_api_key() else ""
    if p == "ollama":
        return "ollama" if _ollama_reachable(cfg) else ""
    if p == "auto":
        if _ollama_reachable(cfg):
            return "ollama"
        if groq_api_key():
            return "groq"
        if gemini_api_key():
            return "gemini"
        return ""
    return ""


def complete(cfg: Config, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    provider = provider_available(cfg)
    if not provider:
        return ""
    try:
        fn = {"ollama": _ollama, "groq": _groq, "gemini": _gemini}[provider]
        return fn(cfg, prompt, system, max_tokens)
    except Exception as e:  # LLM failure must never kill the pipeline
        log.warning("LLM call failed (%s): %s", provider, e)
        return ""


def _ollama(cfg: Config, prompt: str, system: str, max_tokens: int) -> str:
    r = requests.post(
        f"{cfg.ollama_url}/api/generate",
        json={
            "model": cfg.ollama_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.8},
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _groq(cfg: Config, prompt: str, system: str, max_tokens: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {groq_api_key()}"},
        json={"model": cfg.groq_model, "messages": messages,
              "max_tokens": max_tokens, "temperature": 0.8},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _gemini(cfg: Config, prompt: str, system: str, max_tokens: int) -> str:
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.8},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{cfg.gemini_model}:generateContent",
        headers={"x-goog-api-key": gemini_api_key()},
        json=body, timeout=120,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def extract_json_array(text: str):
    """Pull the first JSON array out of an LLM reply, tolerating prose/fences."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
