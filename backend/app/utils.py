from __future__ import annotations

import uuid


def try_parse_uuid(value: str) -> uuid.UUID:
    return uuid.UUID(value)
