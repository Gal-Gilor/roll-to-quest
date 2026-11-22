"""Vertex AI client and embedding calls for the GraphRAG chatbot.

Authentication uses Application Default Credentials:
    gcloud auth application-default login
"""

from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.api_core.retry import Retry
from google.genai import types
from google.genai.errors import APIError
from tqdm import tqdm

PROJECT = "agent-dnd"
LOCATION = "us-central1"
GENERATION_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768  # The model's full 3,072 dimensions aren't needed for 2,400 nodes
MAX_EMBEDDING_CHARS = 8000  # About the model's 2,048-token input limit
RETRYABLE_STATUS_CODES = (429, 500, 503)


def create_client() -> genai.Client:
    return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)


def _is_retryable(exception: Exception) -> bool:
    return isinstance(exception, APIError) and exception.code in RETRYABLE_STATUS_CODES


@Retry(predicate=_is_retryable, initial=1.0, maximum=60.0, timeout=300.0)
def _embed(client: genai.Client, text: str, task_type: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text[:MAX_EMBEDDING_CHARS],
        config=types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS
        ),
    )

    return response.embeddings[0].values


def embed_query(client: genai.Client, query: str) -> list[float]:
    """Embed a question for retrieval."""
    return _embed(client, query, "RETRIEVAL_QUERY")


def embed_documents(
    client: genai.Client, texts: list[str], max_workers: int = 8
) -> list[list[float]]:
    """Embed documents for retrieval, in input order.

    `gemini-embedding-001` on Vertex AI takes one text per request, so the requests
    run in a thread pool.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        embeddings = pool.map(lambda text: _embed(client, text, "RETRIEVAL_DOCUMENT"), texts)

        return list(tqdm(embeddings, total=len(texts), desc="Embedding nodes"))
