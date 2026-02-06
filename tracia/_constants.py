"""Constants and configuration for the Tracia SDK."""

from __future__ import annotations

# SDK Version (defined here to avoid circular imports)
SDK_VERSION = "0.2.0"

# API Configuration
BASE_URL = "https://app.tracia.io"

# Timeout Configuration (in milliseconds)
DEFAULT_TIMEOUT_MS = 120_000  # 2 minutes

# Span Management
MAX_PENDING_SPANS = 1000
SPAN_RETRY_ATTEMPTS = 2
SPAN_RETRY_DELAY_MS = 500

# Span Status
SPAN_STATUS_SUCCESS = "SUCCESS"
SPAN_STATUS_ERROR = "ERROR"

# ID Prefixes
SPAN_ID_PREFIX = "sp_"
TRACE_ID_PREFIX = "tr_"

# Environment Variable Names for Provider API Keys
ENV_VAR_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

# Evaluation Constants
class Eval:
    """Evaluation value constants."""

    POSITIVE = 1
    NEGATIVE = 0
