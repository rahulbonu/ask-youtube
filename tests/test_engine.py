from __future__ import annotations

from ytrag.config import Settings
from ytrag.engine import NO_ANSWER, ChatEngine, resolve_citations
from ytrag.index import Retrieved, build_index
from ytrag.llm.fake import FakeProvider


def make_engine(transcript, embedder, replies, **overrides) -> ChatEngine:
    index = build_index(transcript, embedder, chunk_chars=60, overlap_chars=0)
    settings = Settings(top_k=3, **overrides)
    return ChatEngine(index, embedder, FakeProvider(replies), settings)


def test_answer_streams_and_resolves_citations(transcript, embedder) -> None:
    engine = make_engine(
        transcript,
        embedder,
        ["Plants convert sunlight into energy [1]."],
        condense_question=False,
    )
    stream = engine.stream("What do plants do?")
    streamed = "".join(stream)

    assert "Plants convert sunlight" in streamed
    answer = stream.result
    assert answer.text.strip().endswith("[1].")
    assert len(answer.citations) == 1
    assert answer.citations[0].marker == 1
    assert "watch?v=dQw4w9WgXcQ&t=" in answer.citations[0].url


def test_answer_without_iterating_the_stream(transcript, embedder) -> None:
    engine = make_engine(transcript, embedder, ["Sunlight [1]."], condense_question=False)
    answer = engine.answer("What do plants do?")
    assert "Sunlight" in answer.text
    assert answer.citations


def test_follow_up_is_rewritten_before_retrieval(transcript, embedder) -> None:
    engine = make_engine(
        transcript,
        embedder,
        ["What does chlorophyll absorb?", "It absorbs red and blue light [1]."],
    )
    history = [("What is photosynthesis?", "It converts sunlight to energy.")]
    answer = engine.answer("what about chlorophyll?", history)

    assert answer.standalone_question == "What does chlorophyll absorb?"
    assert answer.was_rewritten is True


def test_no_rewrite_without_history(transcript, embedder) -> None:
    engine = make_engine(transcript, embedder, ["Answer [1]."])
    answer = engine.answer("What is photosynthesis?")
    assert answer.standalone_question == "What is photosynthesis?"
    assert answer.was_rewritten is False


def test_rewriting_can_be_disabled(transcript, embedder) -> None:
    engine = make_engine(transcript, embedder, ["Answer [1]."], condense_question=False)
    answer = engine.answer("what about it?", [("q", "a")])
    assert answer.standalone_question == "what about it?"


def test_empty_index_declines_to_answer(transcript, embedder) -> None:
    engine = make_engine(transcript, embedder, ["should not be used"], condense_question=False)
    engine.index.chunks = []
    engine.index.vectors = engine.index.vectors[:0]

    answer = engine.answer("anything?")
    assert answer.text == NO_ANSWER
    assert answer.citations == []
    # The provider must not be called when there is nothing to ground an answer in.
    assert engine.provider.calls == []


def test_context_sent_to_the_model_is_numbered_and_timestamped(transcript, embedder) -> None:
    engine = make_engine(transcript, embedder, ["Answer [1]."], condense_question=False)
    engine.answer("photosynthesis?")

    _system, messages = engine.provider.calls[0]
    prompt = messages[0].content
    assert "[1] (" in prompt
    assert "Question: photosynthesis?" in prompt


# -- citation resolution ------------------------------------------------------


def _retrieved(transcript, embedder, count=3):
    index = build_index(transcript, embedder, chunk_chars=60, overlap_chars=0)
    return [Retrieved(chunk=chunk, score=1.0) for chunk in index.chunks[:count]]


def test_resolves_multiple_and_grouped_markers(transcript, embedder) -> None:
    retrieved = _retrieved(transcript, embedder, 3)
    citations = resolve_citations("First [1][3]. Second [2, 3].", retrieved, "vid00000001")
    assert sorted(c.marker for c in citations) == [1, 2, 3]


def test_ignores_out_of_range_markers(transcript, embedder) -> None:
    retrieved = _retrieved(transcript, embedder, 2)
    citations = resolve_citations("Hallucinated [9] but real [1].", retrieved, "vid00000001")
    assert [c.marker for c in citations] == [1]


def test_ignores_zero_and_non_numeric_markers(transcript, embedder) -> None:
    retrieved = _retrieved(transcript, embedder, 2)
    assert resolve_citations("See [0] and [abc].", retrieved, "vid00000001") == []


def test_citations_are_ordered_by_position_in_the_video(transcript, embedder) -> None:
    retrieved = _retrieved(transcript, embedder, 3)
    citations = resolve_citations("Late [3]. Early [1].", retrieved, "vid00000001")
    starts = [c.chunk.start for c in citations]
    assert starts == sorted(starts)


def test_uncited_answer_yields_no_citations(transcript, embedder) -> None:
    retrieved = _retrieved(transcript, embedder, 2)
    assert resolve_citations("No markers here at all.", retrieved, "vid00000001") == []
