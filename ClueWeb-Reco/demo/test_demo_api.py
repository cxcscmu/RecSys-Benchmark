"""CLI script for manually testing the ClueWeb demo API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import requests


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the ClueWeb demo API with sample titles.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the running FastAPI demo service (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of documents to request from the /recommend endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--title",
        action="append",
        dest="titles",
        help="Browsing history title to include. Can be repeated. If omitted, built-in examples are used.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for each request (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip calling the /health endpoint before /recommend.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DEMO_CLIENT_API_KEY"),
        help="API key for the demo service (default: value from DEMO_CLIENT_API_KEY environment variable)",
    )
    return parser.parse_args(argv)


def _get_titles(args: argparse.Namespace) -> List[str]:
    if args.titles:
        return [title.strip() for title in args.titles if title.strip()]
    return [
        "iPhone 15 Pro Max review",
        "best phone cameras 2024",
        "Samsung Galaxy S24 specs",
    ]


def _call_health(base_url: str, timeout: float) -> None:
    url = f"{base_url.rstrip('/')}/health"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    print(f"Health check OK: {json.dumps(payload)}")


def _call_recommend(base_url: str, titles: List[str], top_k: int, timeout: float, api_key: str) -> None:
    url = f"{base_url.rstrip('/')}/recommend"
    payload = {"titles": titles, "top_k": top_k}
    headers = {"X-API-Key": api_key}
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()

    recommendation = response.json()
    query = recommendation.get("query")
    print("\nRequest payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if query is not None:
        print("\nGenerated query:")
        print(query)
    print("\nResponse body:")
    print(json.dumps(recommendation, indent=2, ensure_ascii=False))


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    api_key = args.api_key
    if not api_key:
        print("An API key is required. Provide via --api-key or DEMO_CLIENT_API_KEY.")
        return 1

    titles = _get_titles(args)
    if not titles:
        print("No titles provided after stripping empty values.")
        return 1

    try:
        if not args.skip_health:
            _call_health(args.base_url, args.timeout)
        _call_recommend(args.base_url, titles, args.top_k, args.timeout, api_key)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
