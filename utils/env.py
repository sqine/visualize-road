"""비밀값(API 키) 읽기 — 코드에 키를 적지 않는다.

우선순위: 환경변수 → streamlit_app/.env → road-analysis/.env
.env 는 .gitignore 에 들어 있으며 절대 커밋하지 않는다. 형식은 .env.example 참고.
"""

from __future__ import annotations

import os
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent
_ENV_FILES = (_APP / ".env", _APP.parent / ".env")


def get_secret(name: str, required: bool = True) -> str | None:
    val = os.environ.get(name)
    if not val:
        for p in _ENV_FILES:
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{name}="):
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    break
            if val:
                break
    if required and not val:
        raise SystemExit(
            f"{name} 가 없습니다. `export {name}=...` 또는 .env 에 적어주세요 (.env.example 참고)."
        )
    return val
