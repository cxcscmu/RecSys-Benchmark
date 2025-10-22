# ClueWeb-Reco Demo

- **Purpose:** Transform a user’s recent browsing titles into a semantic search intent and fetch the most relevant documents from the ClueWeb service.  
- **Value:** Provides an end-to-end validation path for generative recommendation ideas, covering prompt construction, LLM-driven query generation, and live retrieval.

## Quick Start

### Setup Environment
The demo server has been tested with Python 3.13.7. It is recommended to create a virtual environment for the demo.
```bash
cd ClueWeb-Reco/demo
conda create -n orbit-demo python=3.13.7 -y
conda activate orbit-demo
pip install -r requirements.txt
```

### Starting the Demo Server
```bash
cd ..  # You should be in the ClueWeb-Reco directory
python -m demo.demo
```

### Manage API Keys
See [auth/auth_key_manager.py](./auth/auth_key_manager.py) for instructions on how to add and manage API keys for accessing the demo server.

## Processing Workflow
1. **Request Intake:** Receive an ordered list of titles along with the desired document count (`top_k`).
2. **Query Generation:**
   - `generate_prompt` composes the browsing history into a query-generation prompt.
   - `query_openai` sends the prompt to ChatCompletion and returns the generated search query.
3. **Document Retrieval:**
   - The service calls `https://clueweb22.us/search` with the generated query and `top_k` value.
   - Authentication uses `CLUEWEB_API_KEY`; the request timeout is 30 seconds.
4. **Result Handling:**
   - Each base64-encoded document string in the response is decoded and parsed as JSON.
   - Malformed entries are skipped with warnings.
5. **Response Assembly:** Send back the generated query and the list of decoded document dictionaries.

## API Contract

### Endpoint
`POST /recommend`

### Request Payload
| Field    | Type        | Description                                                                 |
|----------|-------------|-----------------------------------------------------------------------------|
| `titles` | `List[str]` | Ordered browsing history titles; must contain at least one non-empty value. |
| `top_k`  | `int`       | Number of documents to retrieve (1–100, defaults to 10).                    |

### Response Payload
| Field              | Type                | Description                                                        |
|--------------------|---------------------|--------------------------------------------------------------------|
| `query`            | `str`               | Search query generated from the provided titles.                   |
| `recommended_pages` | `List[Dict[str, Any]]` | Documents returned by the ClueWeb API (IDs, titles, snippets, scores, etc.). |

### Sample Request
```json
POST /recommend
{
  "titles": [
    "iPhone 15 Pro Max review",
    "best phone cameras 2024",
    "Samsung Galaxy S24 specs"
  ],
  "top_k": 5
}
```

### Sample Response
```json
HTTP/1.1 200 OK
{
  "query": "flagship phone camera comparison",
  "recommended_pages": [
    {
      "docid": "...",
      "title": "...",
      "snippet": "...",
      "score": 0.91
    },
    {
      "docid": "...",
      "title": "...",
      "snippet": "...",
      "score": 0.87
    }
  ]
}
```

## Supporting Script
- [`ClueWeb-Reco/test_demo_api.py`](./test_demo_api.py): CLI helper with flags for `--title`, `--top-k`, `--base-url`, `--timeout`, `--skip-health`, and `--api-key`. Prints the request payload, generated query, and full response for quick validation while the API service is running.
