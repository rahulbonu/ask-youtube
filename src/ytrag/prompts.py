"""Prompt construction."""

from __future__ import annotations

from .index import Retrieved
from .video import format_timestamp

ANSWER_SYSTEM = """\
You answer questions about a single YouTube video, using excerpts from its \
transcript.

Rules:
1. Answer only from the excerpts provided. They are the entire source of truth.
2. Never invent details. If the excerpts do not contain the answer, reply exactly:
   "I couldn't find the answer to that in this video."
3. Cite every factual claim with the excerpt number it came from, like [2]. Cite
   more than one where a claim draws on several, like [1][3]. Put the marker at
   the end of the sentence it supports.
4. Never cite an excerpt number that does not appear below.
5. Transcripts are often auto-generated and may contain mistakes, and may be in a
   language other than the question. Read past obvious transcription errors.
6. Answer in the same language the question was asked in.
7. Be direct and concise. Do not restate the question or describe what you are
   about to do. Do not mention "excerpts" or "transcript" in the answer itself -
   the citations already make the source clear.
"""

CONDENSE_SYSTEM = """\
You rewrite a follow-up question into a standalone one.

Given a conversation and a follow-up, output a single question that can be
understood with no prior context, resolving pronouns and implicit references
against the conversation.

Output only the rewritten question - no preamble, no quotes, no explanation. If
the follow-up already stands alone, output it unchanged.
"""


def build_context(retrieved: list[Retrieved]) -> str:
    """Render retrieved chunks as numbered, timestamped excerpts.

    The numbering here is what the model cites, and it is 1-based to match how
    citations read to a person. Timestamps are included so the model can refer to
    when something was said.
    """
    blocks = []
    for position, item in enumerate(retrieved, start=1):
        span = f"{format_timestamp(item.chunk.start)}-{format_timestamp(item.chunk.end)}"
        blocks.append(f"[{position}] ({span})\n{item.chunk.text}")
    return "\n\n".join(blocks)


def build_answer_prompt(question: str, retrieved: list[Retrieved]) -> str:
    """Build the user turn: excerpts first, then the question."""
    return (
        "Transcript excerpts:\n"
        "--------------------\n"
        f"{build_context(retrieved)}\n"
        "--------------------\n\n"
        f"Question: {question}"
    )


def build_condense_prompt(question: str, history: list[tuple[str, str]]) -> str:
    """Build the user turn for question rewriting."""
    lines = []
    for user_turn, assistant_turn in history:
        lines.append(f"User: {user_turn}")
        lines.append(f"Assistant: {assistant_turn}")
    conversation = "\n".join(lines) if lines else "(no previous turns)"
    return f"Conversation:\n{conversation}\n\nFollow-up: {question}\n\nStandalone question:"
