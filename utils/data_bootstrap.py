"""런타임 데이터 부트스트랩.

앱 첫 실행 시 필수 데이터 파일이 있는지 검사하고, 없으면 두 가지 경로로
보충:
    A. Google Drive 공개 zip 다운로드 (온라인 환경)
    B. 사용자가 로컬 zip/geojson 파일 직접 업로드 (폐쇄망 환경)

표준 라이브러리만 사용 (urllib, zipfile) — 추가 의존성 없음.
"""

from __future__ import annotations

import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

# 앱이 동작하려면 반드시 있어야 하는 파일들
REQUIRED_FILES = (
    "seoul_roads.geojson",
    "seoul_gu.geojson",
    "heating_2026.csv",
    "icing_zones.csv",
    "flpop.csv",
)

# 있으면 추가 기능 활성화되는 파일들 (없어도 앱은 동작)
OPTIONAL_FILES = (
    "road_links.geojson",  # 유동인구 × 경사도 도로 시각화
)

# 파일별 설명/용도 (UI에서 안내용)
FILE_INFO: dict[str, dict] = {
    "heating_2026.csv": {
        "size": "50KB",
        "desc": "열선 설치 현황 (자치구별)",
        "required": True,
    },
    "icing_zones.csv": {
        "size": "564KB",
        "desc": "행정안전부 상습 결빙구간",
        "required": True,
    },
    "seoul_gu.geojson": {
        "size": "36KB",
        "desc": "서울 자치구 25개 경계",
        "required": True,
    },
    "seoul_roads.geojson": {
        "size": "~18MB",
        "desc": "OSM 도로 라인 + 경사도",
        "required": True,
    },
    "flpop.csv": {
        "size": "~750KB",
        "desc": "도로구간별 유동인구 (기본 샘플)",
        "required": True,
    },
    "road_links.geojson": {
        "size": "~55MB",
        "desc": "TBGIS 도로링크 좌표 + 경사도 (유동인구 시각화용)",
        "required": False,
    },
}

# Drive 다운로드 기본 URL.
# road_data.zip 안에 seoul_roads.geojson + road_links.geojson 두 파일.
# 사용자가 UI 입력란에서 덮어쓸 수 있고, Streamlit secrets.toml 의
# DATA_DRIVE_URL 키로도 설정 가능.
DEFAULT_DRIVE_URL = (
    "https://drive.google.com/file/d/1Mn8bvZF5UIpCqg1WYBGekx1Gkdc1NX3h/view"
)


def missing_files(data_dir: Path) -> list[str]:
    """필수 파일 중 누락된 것 목록."""
    return [name for name in REQUIRED_FILES if not (data_dir / name).exists()]


def missing_optional(data_dir: Path) -> list[str]:
    """선택 파일 중 누락된 것 목록 (없어도 앱은 동작)."""
    return [name for name in OPTIONAL_FILES if not (data_dir / name).exists()]


# ──────────────────────────────────────────────────────────────────────
# Google Drive 다운로드
# ──────────────────────────────────────────────────────────────────────

_DRIVE_ID_RE = re.compile(r"(?:/d/|id=)([A-Za-z0-9_-]{20,})")


def extract_drive_id(url_or_id: str) -> str:
    """URL 또는 raw ID에서 file ID만 추출."""
    s = (url_or_id or "").strip()
    if not s:
        return ""
    m = _DRIVE_ID_RE.search(s)
    if m:
        return m.group(1)
    qs = urllib.parse.urlparse(s).query
    qs_dict = urllib.parse.parse_qs(qs)
    if "id" in qs_dict:
        return qs_dict["id"][0]
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", s):
        return s
    return ""


def download_from_drive(
    url_or_id: str, dest: Path, progress=None
) -> None:
    """Google Drive 공개 파일을 dest 경로에 저장.

    25MB 초과 파일은 confirm token이 필요할 수 있지만 우리 데이터는 그보다
    작아 단순 GET으로 충분. 만약 confirm 페이지가 응답으로 오면 한 번 더
    토큰 추출해서 재요청.

    progress: callable(downloaded_bytes, total_bytes) — Streamlit progress bar 등.
    """
    file_id = extract_drive_id(url_or_id)
    if not file_id:
        raise ValueError("유효한 Google Drive URL 또는 ID가 아닙니다.")

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    headers = {"User-Agent": "Mozilla/5.0 (road-analysis)"}

    def _fetch(target_url: str) -> tuple[bytes, dict]:
        req = urllib.request.Request(target_url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read(), dict(resp.headers)

    # 첫 요청 — content가 HTML(confirm page)이면 토큰 추출 후 재요청
    body, hdrs = _fetch(url)
    ctype = hdrs.get("Content-Type", "")
    if ctype.startswith("text/html") and len(body) < 200_000:
        m = re.search(rb'name="confirm"\s+value="([^"]+)"', body) or re.search(
            rb"confirm=([0-9A-Za-z_-]+)", body
        )
        if m:
            token = m.group(1).decode()
            url2 = (
                f"https://drive.google.com/uc?export=download"
                f"&confirm={token}&id={file_id}"
            )
            body, hdrs = _fetch(url2)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    if progress is not None:
        total = int(hdrs.get("Content-Length", 0) or len(body))
        progress(len(body), max(total, len(body)))


# ──────────────────────────────────────────────────────────────────────
# zip / 단일 파일 추출
# ──────────────────────────────────────────────────────────────────────

def extract_archive_to(src: Path, data_dir: Path) -> list[str]:
    """zip 파일이면 풀어서 data_dir 에 배치. 일반 파일이면 그대로 복사.

    반환: 배치된 파일 이름 리스트.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    placed: list[str] = []

    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # zip 내부 경로의 마지막 이름만 사용 (depth 무시)
                name = Path(info.filename).name
                if not name:
                    continue
                dest = data_dir / name
                with zf.open(info) as f_in, dest.open("wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                placed.append(name)
    else:
        dest = data_dir / src.name
        shutil.copyfile(src, dest)
        placed.append(src.name)

    return placed


def save_uploaded_to(file_obj, filename: str, data_dir: Path) -> Path:
    """Streamlit UploadedFile 객체를 임시 파일로 저장."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = data_dir / f"_upload_{filename}"
    with tmp.open("wb") as f:
        f.write(file_obj.getbuffer())
    return tmp
