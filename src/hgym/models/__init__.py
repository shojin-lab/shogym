"""The inference seam (RFC 001 Section 6)."""

from hgym.models._client import (
    CompletionRequest,
    ModelClient,
    OpenAICompatClient,
    build_extra_kwargs,
    split_model,
)

__all__ = [
    "CompletionRequest",
    "ModelClient",
    "OpenAICompatClient",
    "build_extra_kwargs",
    "split_model",
]
