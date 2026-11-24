"""
extract_all.py — 내 컴퓨터에 있는 여러 개의 match_N.json / timeline_N.json 을 한 번에 읽어서
                  분석에 필요한 작은 CSV 3개로 줄여 주는 스크립트

사용법:
    python extract_all.py match_data      # match_data 폴더 (하위 폴더까지 찾음)
    python extract_all.py                 # 현재 폴더

결과 (out/ 폴더, 1,000경기 기준 합계 약 15MB):
    out/lane_tci_dataset.csv   라인별 피처 + TCI (5~20분)      → 모델 학습용
    out/match_summary.csv      경기별 팀 합계 (골드·경험치·딜·킬 등) → 그림 01용
    out/lane_minute_gold.csv   라인별 3~30분 골드·경험치 격차   → 그림 03·04·05용
"""
import glob
import json
import os
import re
import sys
import pandas as pd


LANES = ["TOP", "MIDDLE", "JUNGLE", "BOTTOM"]
MINUTES = range(5, 21)          # 5~20분 (30경기 모두 20분 이상 진행)
LANE_TYPE = {"TOP": "TOP_LANE", "MIDDLE": "MID_LANE", "BOTTOM": "BOT_LANE"}


# ------------------------------------------------------------------ #
# 1. 기본 정보 (data_process_sample.py 와 동일한 틀)
# ------------------------------------------------------------------ #
def get_default_match_data(match, timeline):
    data = {"matchId": match["metadata"]["matchId"],
            "numFrames": len(timeline["info"]["frames"])}
    p0 = match["info"]["participants"][0]
    data["blueWin"] = p0["win"] if p0["teamId"] == 100 else (not p0["win"])
    for lane in LANES:
        data[lane] = {"blue": {"feature": {}, "pids": []}, "red": {"feature": {}, "pids": []}}
    for p in match["info"]["participants"]:
        side = "blue" if p["teamId"] == 100 else "red"
        pos = "BOTTOM" if p["teamPosition"] == "UTILITY" else p["teamPosition"]
        data[pos][side]["pids"].append(p["participantId"])   # BOTTOM 은 [원딜, 서폿] 2명
    return data


# ------------------------------------------------------------------ #
# 2. 타겟 변수 y : TCI = (우리 라인 골드 - 상대 라인 골드) / 게임 전체 골드
# ------------------------------------------------------------------ #
def calculate_tci(pf, blue_pids, red_pids):
    tot_g = sum(pf[str(i)]["totalGold"] for i in range(1, 11))
    g = sum(pf[str(i)]["totalGold"] for i in blue_pids) - sum(pf[str(i)]["totalGold"] for i in red_pids)
    return g / tot_g


# ------------------------------------------------------------------ #
# 3. 피처 추출 : 모든 라인 공통 + 라인 특화 (전부 누적 격차)
# ------------------------------------------------------------------ #
def _frame_stat(pf, pids, key):
    if key == "cs":
        return sum(pf[str(i)]["minionsKilled"] + pf[str(i)]["jungleMinionsKilled"] for i in pids)
    if key == "jgCs":
        return sum(pf[str(i)]["jungleMinionsKilled"] for i in pids)
    if key == "dmgDealt":
        return sum(pf[str(i)]["damageStats"]["totalDamageDoneToChampions"] for i in pids)
    if key == "dmgTaken":
        return sum(pf[str(i)]["damageStats"]["totalDamageTaken"] for i in pids)
    if key == "cc":
        return sum(pf[str(i)]["timeEnemySpentControlled"] for i in pids)
    raise KeyError(key)


def check_effective_followup(all_events, base_ts, team_id):
    """킬 이후 60초 이내 같은 팀이 에픽 몬스터/건물을 처치했는지"""
    team_pids = range(1, 6) if team_id == 100 else range(6, 11)
    for e in all_events:
        dt = e.get("timestamp", 0) - base_ts
        if dt <= 0:
            continue
        if dt > 60000:
            return False
        if e["type"] in ("ELITE_MONSTER_KILL", "BUILDING_KILL") and e.get("killerId", 0) in team_pids:
            return True
    return False


def get_lane_feature(lane, timeline, data):
    """
    반환: {피처명: [0분, 1분, ..., 마지막분] 누적 격차 리스트} (블루 - 레드)
    """
    bp, rp = data[lane]["blue"]["pids"], data[lane]["red"]["pids"]
    frames = timeline["info"]["frames"]
    all_events = [e for f in frames for e in f.get("events", [])]

    # 이벤트 누적 카운터 (blue, red)
    cnt = {k: [0, 0] for k in ["kill", "death", "assist", "soloKill", "plate",
                               "wardPlaced", "wardKill", "epic", "effGank", "roamKA"]}
    feat = {k: [] for k in ["csDiff", "dmgDealtDiff", "dmgTakenDiff", "ccDiff",
                            "killDiff", "deathDiff"]}
    extra = {"TOP": ["soloKillDiff", "plateDiff"],
             "MIDDLE": ["soloKillDiff", "plateDiff", "roamKADiff"],
             "JUNGLE": ["jgCsDiff", "assistDiff", "epicDiff", "effGankDiff"],
             "BOTTOM": ["plateDiff", "assistDiff", "wardPlacedDiff", "wardKillDiff"]}[lane]
    for k in extra:
        feat[k] = []

    for frame in frames:
        pf = frame["participantFrames"]
        for e in frame.get("events", []):
            t = e["type"]
            if t == "CHAMPION_KILL":
                killer, victim = e.get("killerId", 0), e.get("victimId", 0)
                assists = e.get("assistingParticipantIds", [])
                for s, (mine, enemy) in enumerate([(bp, rp), (rp, bp)]):
                    if killer in mine:
                        cnt["kill"][s] += 1
                        if not assists and victim in enemy:
                            cnt["soloKill"][s] += 1
                    if victim in mine:
                        cnt["death"][s] += 1
                    if any(a in mine for a in assists):
                        cnt["assist"][s] += 1
                    # 미드 로밍: 상대 미드가 아닌 챔피언 처치에 킬/어시 관여
                    if (killer in mine or any(a in mine for a in assists)) and victim not in enemy:
                        cnt["roamKA"][s] += 1
                    # 정글 유효 갱킹
                    if lane == "JUNGLE" and (killer in mine or any(a in mine for a in assists)):
                        if check_effective_followup(all_events, e["timestamp"], 100 if s == 0 else 200):
                            cnt["effGank"][s] += 1
            elif t == "TURRET_PLATE_DESTROYED" and lane in LANE_TYPE and e.get("laneType") == LANE_TYPE[lane]:
                # teamId = 포탑 주인. 레드(200) 포탑의 골드가 깎이면 블루 이득
                cnt["plate"][0 if e.get("teamId") == 200 else 1] += 1
            elif t == "WARD_PLACED":
                for s, mine in enumerate([bp, rp]):
                    if e.get("creatorId") in mine:
                        cnt["wardPlaced"][s] += 1
            elif t == "WARD_KILL":
                for s, mine in enumerate([bp, rp]):
                    if e.get("killerId") in mine:
                        cnt["wardKill"][s] += 1
            elif t == "ELITE_MONSTER_KILL":
                cnt["epic"][0 if e.get("killerTeamId") == 100 else 1] += 1

        # 프레임 값 -> 누적 격차
        feat["csDiff"].append(_frame_stat(pf, bp, "cs") - _frame_stat(pf, rp, "cs"))
        feat["dmgDealtDiff"].append(_frame_stat(pf, bp, "dmgDealt") - _frame_stat(pf, rp, "dmgDealt"))
        feat["dmgTakenDiff"].append(_frame_stat(pf, bp, "dmgTaken") - _frame_stat(pf, rp, "dmgTaken"))
        feat["ccDiff"].append(_frame_stat(pf, bp, "cc") - _frame_stat(pf, rp, "cc"))
        feat["killDiff"].append(cnt["kill"][0] - cnt["kill"][1])
        feat["deathDiff"].append(cnt["death"][0] - cnt["death"][1])
        for k in extra:
            if k == "jgCsDiff":
                feat[k].append(_frame_stat(pf, bp, "jgCs") - _frame_stat(pf, rp, "jgCs"))
            else:
                base = k.replace("Diff", "")
                feat[k].append(cnt[base][0] - cnt[base][1])

    data[lane]["blue"]["feature"].update(feat)
    data[lane]["red"]["feature"].update({k: [-v for v in vs] for k, vs in feat.items()})
    return feat



# ------------------------------------------------------------------ #
# 여러 파일 한꺼번에 처리
# ------------------------------------------------------------------ #
POSITIONS = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}


def is_valid(match, timeline):
    """분석에 쓸 수 있는 경기인지 확인 (다시하기·조기 항복·포지션 누락 제외)"""
    ps = match["info"]["participants"]
    if len(ps) != 10:
        return "참가자 10명 아님"
    if match["metadata"]["matchId"] != timeline["metadata"]["matchId"]:
        return "match/timeline 순서 불일치"
    if len(timeline["info"]["frames"]) < 21:
        return "20분 미만"
    for side in (100, 200):
        pos = [p["teamPosition"] for p in ps if p["teamId"] == side]
        if set(pos) != POSITIONS:
            return "포지션 누락/중복"
    return None


def file_pairs(data_dir):
    """폴더(하위 폴더 포함)에서 match_N.json 과 timeline_N.json 을 번호로 짝지음"""
    found = {"match": {}, "timeline": {}}
    for path in glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True):
        name = os.path.basename(path).lower()
        m = re.match(r"(match|timeline)[_-]?(\d+)\.json$", name)
        if m:
            found[m.group(1)].setdefault(int(m.group(2)), path)
    nums = sorted(set(found["match"]) & set(found["timeline"]))
    only_m = sorted(set(found["match"]) - set(found["timeline"]))
    only_t = sorted(set(found["timeline"]) - set(found["match"]))
    print(f"짝이 맞는 파일 {len(nums)}쌍 발견 (번호 {nums[:5]}{' …' if len(nums) > 5 else ''})")
    if only_m or only_t:
        print(f"  짝이 없는 파일 — match만: {only_m[:10]} / timeline만: {only_t[:10]}")
    return [(n, found["match"][n], found["timeline"][n]) for n in nums]


def main(data_dir="."):
    os.makedirs("out", exist_ok=True)
    pairs = file_pairs(data_dir)
    if not pairs:
        print("match_N.json / timeline_N.json 짝을 찾지 못했습니다:", os.path.abspath(data_dir))
        return
    lane_rows, summary_rows, minute_rows = [], [], []
    seen, skipped, broken = set(), {}, []
    for n, mf, tf in pairs:
        try:
            with open(mf, encoding="utf-8") as f:
                matches = json.load(f)
            with open(tf, encoding="utf-8") as f:
                timelines = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            # 저장 도중 끊긴 파일 등 깨진 파일은 건너뜀
            broken.append(f"{os.path.basename(mf)} / {os.path.basename(tf)}: {type(e).__name__}")
            print(f"  ⚠ {n}번 파일이 깨져서 건너뜀 ({type(e).__name__})")
            continue
        tl_by_id = {t["metadata"]["matchId"]: t for t in timelines}
        for m in matches:
            mid = m["metadata"]["matchId"]
            if mid in seen:
                skipped["중복 경기"] = skipped.get("중복 경기", 0) + 1
                continue
            tl = tl_by_id.get(mid)
            reason = "타임라인 없음" if tl is None else is_valid(m, tl)
            if reason:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            seen.add(mid)
            data = get_default_match_data(m, tl)

            # (1) 모델 학습용
            for lane in LANES:
                feat = get_lane_feature(lane, tl, data)
                for t in MINUTES:
                    pf = tl["info"]["frames"][t]["participantFrames"]
                    row = {"matchId": mid, "lane": lane, "minute": t, "blueWin": data["blueWin"],
                           "tci": calculate_tci(pf, data[lane]["blue"]["pids"], data[lane]["red"]["pids"])}
                    row.update({k: v[t] for k, v in feat.items()})
                    lane_rows.append(row)

            # (2) 경기 요약 (그림 01)
            srow = {"matchId": mid, "blueWin": data["blueWin"], "durationSec": m["info"]["gameDuration"]}
            for key in ["goldEarned", "champExperience", "totalDamageDealtToChampions", "kills", "totalDamageTaken"]:
                srow[f"blue_{key}"] = sum(p[key] for p in m["info"]["participants"] if p["teamId"] == 100)
                srow[f"red_{key}"] = sum(p[key] for p in m["info"]["participants"] if p["teamId"] != 100)
            summary_rows.append(srow)

            # (3) 분 단위 라인 골드·경험치 (그림 03·04·05)
            frames = tl["info"]["frames"]
            for t in range(3, min(31, len(frames))):
                pf = frames[t]["participantFrames"]
                tg = sum(pf[str(i)]["totalGold"] for i in range(1, 11))
                tx = sum(pf[str(i)]["xp"] for i in range(1, 11))
                for lane in LANES:
                    b, r = data[lane]["blue"]["pids"], data[lane]["red"]["pids"]
                    minute_rows.append({
                        "matchId": mid, "lane": lane, "minute": t, "blueWin": data["blueWin"],
                        "goldDiff": sum(pf[str(i)]["totalGold"] for i in b) - sum(pf[str(i)]["totalGold"] for i in r),
                        "xpDiff": sum(pf[str(i)]["xp"] for i in b) - sum(pf[str(i)]["xp"] for i in r),
                        "totalGold": tg, "totalXp": tx})
        print(f"  {os.path.basename(mf)} 처리 완료 (누적 {len(seen)}경기)")

    pd.DataFrame(lane_rows).to_csv("out/lane_tci_dataset.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv("out/match_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(minute_rows).to_csv("out/lane_minute_gold.csv", index=False, encoding="utf-8-sig")
    print(f"\n완료: 사용 {len(seen)}경기 / 제외 {sum(skipped.values())}경기 {skipped}")
    if broken:
        print(f"깨져서 건너뛴 파일 {len(broken)}쌍:")
        for b in broken:
            print("  -", b)
    print("out/ 폴더의 CSV 3개를 올려 주세요.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ("match_data" if os.path.isdir("match_data") else "."))
