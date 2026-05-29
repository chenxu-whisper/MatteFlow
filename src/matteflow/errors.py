"""Typed exceptions exposed by the MatteFlow service layer."""

from __future__ import annotations


class MatteFlowError(Exception):
    """Base class for user-facing MatteFlow errors."""


class InputValidationError(MatteFlowError):
    """Raised when an input path or option is invalid before processing."""


class ModelLoadError(MatteFlowError):
    """Raised when an AI model cannot be loaded."""


class JobCancelledError(MatteFlowError):
    """Raised when a user-requested cancellation stops processing."""


class ProgressCallbackError(MatteFlowError):
    """Raised when a progress callback fails and processing must stop."""


class ProcessingError(MatteFlowError):
    """Raised when media processing fails after a job has started."""
