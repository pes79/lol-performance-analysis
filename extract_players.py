"""
extract_players.py — 경기별로 어느 소환사(puuid)가 어느 라인·진영에서 뛰었는지 뽑는 스크립트
(match_N.json 만 읽기 때문에 extract_all.py 보다 훨씬 빠름. 타임라인은 안 읽음)

사용법 (LOL_project 폴더에서):
    python extract_players.py             # match_data 폴더 자동 탐색
    python extract_players.py C:/경로     # 다른 폴더

결과: out/lane_players.csv  (경기 × 10명, 약 15만 줄)
"""
import glob
import json
import os
import re
import sys
import pandas as pd

POS2LANE = {"TOP": ("TOP", "TOP"), "JUNGLE": ("JUNGLE", "JUNGLE"), "MIDDLE": ("MIDDLE", "MIDDLE"),
            "BOTTOM": ("BOTTOM", "ADC"), "UTILITY": ("BOTTOM", "SUP")}


def match_files(data_dir):
    files = []
    for path in glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True):
        m = re.match(r"match[_-]?(\d+)\.json$", os.path.basename(path).lower())
        if m:
            files.append((int(m.group(1)), path))
    return sorted(files)


def main(data_dir):
    os.makedirs("out", exist_ok=True)
    files = match_files(data_dir)
    print(f"match 파일 {len(files)}개 발견")
    rows, seen, broken = [], set(), []
    for n, path in files:
        try:
            with open(path, encoding="utf-8") as f:
                matches = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            broken.append(os.path.basename(path))
            print(f"  ⚠ {os.path.basename(path)} 깨져서 건너뜀 ({type(e).__name__})")
            continue
        for m in matches:
            mid = m["metadata"]["matchId"]
            if mid in seen:
                continue
            seen.add(mid)
            for p in m["info"]["participants"]:
                if p.get("teamPosition") not in POS2LANE:
                    continue
                lane, role = POS2LANE[p["teamPosition"]]
                rows.append({"matchId": mid, "lane": lane, "role": role,
                             "side": "blue" if p["teamId"] == 100 else "red",
                             "puuid": p["puuid"], "championName": p["championName"], "win": p["win"]})
        print(f"  {os.path.basename(path)} 처리 완료 (누적 {len(seen)}경기)")
    pd.DataFrame(rows).to_csv("out/lane_players.csv", index=False, encoding="utf-8-sig")
    print(f"\n완료: {len(seen)}경기, {len(rows)}줄 → out/lane_players.csv 를 올려 주세요.")
    if broken:
        print("깨진 파일:", broken)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ("match_data" if os.path.isdir("match_data") else "."))
