from __future__ import annotations

import re
from typing import Optional


_ROOT_RE = re.compile(r"(<root\b[\s\S]*?</root>)", re.IGNORECASE)


def extract_root_xml(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    m = _ROOT_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()

