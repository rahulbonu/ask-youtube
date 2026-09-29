"""Transcript retrieval: data model, per-source adapters, and the fallback chain."""

from .fetcher import fetch_transcript
from .models import Segment, Transcript

__all__ = ["Segment", "Transcript", "fetch_transcript"]
