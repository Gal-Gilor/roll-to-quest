"""Gradio chatbot that answers D&D rules questions with GraphRAG over the SRD.

On startup the SRD Markdown is split with `markdown-chunkify` and linked into an
in-memory graph of D&D entities (see `src.graphrag.graph`). Each question retrieves
seed nodes and their graph neighbors, and Gemini on Vertex AI answers from their text
and relationships.

Authentication uses Application Default Credentials:
    gcloud auth application-default login

Example Usage:
    python -m src.scripts.chat
    python -m src.scripts.chat --markdown srd_extraction.md --share
"""

import argparse
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
from google import genai
from google.genai import types

from src.graphrag.graph import build_graph
from src.graphrag.retrieval import GraphRetriever

PROJECT = "agent-dnd"
LOCATION = "us-central1"
MODEL = "gemini-3.5-flash-lite"
DEFAULT_MARKDOWN = Path(__file__).parents[2] / "srd_extraction.md"

SYSTEM_INSTRUCTION = (
    "You are a Dungeons & Dragons 5E rules assistant. Answer using only the SRD "
    "sections provided with the question. Each section may end with a Relationships "
    "list taken from a knowledge graph of the SRD; a line starting with '<-' lists the "
    "entities that have that relationship to the section. Name the sections you relied "
    "on. If the sections don't contain the answer, say so."
)
SOURCES_TITLE = "Sources"


def _text(content: str | list) -> str:
    """Return the text of a Gradio message, whose content is a string or a list of parts."""
    if isinstance(content, str):
        return content

    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def create_chat_fn(retriever: GraphRetriever, client: genai.Client):
    """Create the Gradio chat callback bound to a retriever and a GenAI client."""

    def chat(message: str, history: list[dict]) -> Iterator[list[gr.ChatMessage]]:
        # Drop the source lists shown in earlier turns; they aren't part of the dialogue.
        turns = [
            (turn["role"], _text(turn["content"]))
            for turn in history
            if not (turn.get("metadata") or {}).get("title")
        ]

        # Include the previous question so follow-ups like "and at level 5?" retrieve well.
        previous_questions = [text for role, text in turns if role == "user"]
        node_ids = retriever.retrieve(" ".join(previous_questions[-1:] + [message]))

        contents = [
            types.Content(
                role="user" if role == "user" else "model", parts=[types.Part(text=text)]
            )
            for role, text in turns
        ]
        prompt = (
            f"SRD sections:\n\n{retriever.format_context(node_ids)}\n\nQuestion: {message}"
        )
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        nodes = [retriever.graph.nodes[node_id] for node_id in node_ids]
        sources = gr.ChatMessage(
            content="\n".join(f"- {node['label']}: {node['breadcrumb']}" for node in nodes)
            or "No matching sections.",
            metadata={"title": SOURCES_TITLE, "status": "done"},
        )

        answer = ""
        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
        ):
            answer += chunk.text or ""
            yield [sources, gr.ChatMessage(content=answer)]

    return chat


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the SRD using GraphRAG.")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    args = parser.parse_args()

    retriever = GraphRetriever(build_graph(args.markdown))
    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

    gr.ChatInterface(
        fn=create_chat_fn(retriever, client),
        type="messages",
        title="Roll-to-Quest",
        description="Ask about the D&D 5E rules. Answers come from the SRD 5.2.1.",
        examples=[
            "What does Fireball do?",
            "What happens if a ghoul hits me with its claws?",
            "How does grappling work?",
        ],
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
