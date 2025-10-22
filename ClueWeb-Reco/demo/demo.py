"""Minimal API demo service for generating ClueWeb recommendations."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, List

import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field, validator

from get_queries import generate_prompt, query_openai
from demo.auth.auth_db import init_auth, verify_api_key_exists


logger = logging.getLogger("clueweb-demo")
logging.basicConfig(level=os.getenv("CLUEWEB_DEMO_LOG", "INFO"))

CLUEWEB_API_URL = os.getenv("CLUEWEB_API_URL", "https://clueweb22.us/search")


class TitlesRequest(BaseModel):
    titles: List[str] = Field(..., description="Ordered browsing history titles")
    top_k: int = Field(10, ge=1, le=100, description="Number of documents to retrieve")

    @validator("titles")
    def validate_titles(cls, value: List[str]) -> List[str]:  # noqa: D417
        if not value:
            raise ValueError("titles must contain at least one entry")
        cleaned = [title.strip() for title in value if title and title.strip()]
        if not cleaned:
            raise ValueError("titles must contain non-empty strings")
        return cleaned


class RecommendationResponse(BaseModel):
    query: str = Field(..., description="Search query generated from the provided titles")
    recommended_pages: List[Dict[str, Any]] = Field(..., description="Documents returned by the ClueWeb API")


AUTH_DB_PATH = os.getenv("DEMO_AUTH_DB")
AUTH_CHECK_INTERVAL = int(os.getenv("DEMO_AUTH_CHECK_INTERVAL", "60"))
if AUTH_DB_PATH:
    init_auth(auth_file=AUTH_DB_PATH, check_interval=AUTH_CHECK_INTERVAL)
else:
    init_auth(check_interval=AUTH_CHECK_INTERVAL)


app = FastAPI(
    title="ClueWeb Recommendation Demo",
    version="0.1.0",
    description="Demo service that generates a query from browsing history and retrieves ClueWeb pages.",
)


def _get_clueweb_api_key() -> str:
    api_key = os.getenv("CLUEWEB_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set the CLUEWEB_API_KEY environment variable with your ClueWeb search API key before calling this endpoint."
        )
    return api_key


def _retrieve_documents(query: str, top_k: int) -> List[Dict[str, Any]]:
    params = {"query": query, "k": top_k}
    headers = {"X-API-Key": _get_clueweb_api_key()}
    try:
        response = requests.get(CLUEWEB_API_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.exception("Failed to call ClueWeb API")
        raise HTTPException(status_code=502, detail=f"ClueWeb API request failed: {exc}") from exc

    payload = response.json()
    raw_results = payload.get("results", [])
    documents: List[Dict[str, Any]] = []
    for raw_doc in raw_results:
        try:
            decoded = base64.b64decode(raw_doc).decode("utf-8")
            documents.append(json.loads(decoded))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed document: %s", exc)
            continue

    if len(documents) < top_k:
        logger.warning(
            "Requested %s documents but received %s from ClueWeb API.",
            top_k,
            len(documents),
        )
    return documents


@app.post("/recommend", response_model=RecommendationResponse)
def recommend_titles(
    request: TitlesRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> RecommendationResponse:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if not verify_api_key_exists(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")

    prompt = generate_prompt(request.titles)
    query = query_openai(prompt)
    if not query:
        raise HTTPException(status_code=502, detail="Failed to generate query from titles.")

    documents = _retrieve_documents(query, request.top_k)
    if not documents:
        raise HTTPException(status_code=502, detail="No documents returned by ClueWeb API.")

    return RecommendationResponse(query=query, recommended_pages=documents)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
