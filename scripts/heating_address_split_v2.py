import re

ROAD_RE = re.compile(r'[가-힣0-9]+(?:로|길|가)\d*(?:가|길)?')
BARE_NUM = re.compile(r'^\d+(-\d+)?\s*$')

def strip_trailing_paren(s):
    return re.sub(r'\s*\([^()]*\)\s*$', '', s).strip()

def extract_prefix(s):
    m = re.match(r'^([가-힣]+동)\s', s)
    if m:
        return m.group(1)
    m = re.match(r'^([가-힣0-9]+(?:로|길|가))(?:\s|\d)', s + ' ')
    if m:
        return m.group(1)
    return None

def split_sides_v2(raw):
    raw = str(raw).strip()
    m = re.match(r'^(.*?)\(([^()]*[~∼][^()]*)\)\s*$', raw)
    if m:
        raw_for_split = m.group(2)
    else:
        raw_for_split = raw

    # 쉼표로 이어진 여러 구간(도화2길 케이스)은 첫 구간만 사용(보수적으로)
    parts = [p.strip() for p in re.split(r'[~∼]', raw_for_split) if p.strip()]
    if len(parts) == 0:
        return []

    cleaned = []
    for p in parts:
        p = p.split(',')[0].strip()  # 쉼표 이후는 버림(다음 구간 시작점이라 중복 처리 방지)
        inner_parens = re.findall(r'\(([^()]+)\)', p)
        road_inside = next((ip for ip in inner_parens if ROAD_RE.search(ip)), None)
        if road_inside:
            p2 = road_inside.strip()
        else:
            p2 = strip_trailing_paren(p)
            rm = ROAD_RE.search(p2)
            if rm and rm.start() > 0 and not BARE_NUM.match(p2):
                lead = p2[:rm.start()]
                if len(lead) > 1 and not re.search(r'\d', lead):
                    p2 = p2[rm.start():]
        cleaned.append(p2)

    if len(cleaned) >= 2:
        prefix = extract_prefix(cleaned[0])
        for i in range(1, len(cleaned)):
            if BARE_NUM.match(cleaned[i]) and prefix:
                cleaned[i] = f"{prefix} {cleaned[i]}"
    return cleaned


def gen_variants(s):
    """추가 후보: 공백삽입, 오탈자 치환"""
    out = [s]
    # 도로명+숫자 붙어있으면 공백삽입 (예: 동작대로29길118 -> 동작대로29길 118)
    m = re.match(r'^([가-힣0-9]+(?:로|길|가))(\d[\d-]*)$', s)
    if m:
        out.append(f"{m.group(1)} {m.group(2)}")
    if '뢰' in s:
        out.append(s.replace('뢰','로'))
    if '가리' in s:
        out.append(s.replace('가리','길'))
    seen=set(); res=[]
    for x in out:
        if x not in seen:
            seen.add(x); res.append(x)
    return res
