import json
import pandas as pd


def get_default_match_data(match, timeline, challengers):
    data = {}
    data["matchId"] = match["metadata"]["matchId"]

    for lane in ["TOP", "MIDDLE", "JUNGLE", "BOTTOM"]:
        data[lane] = {"blue": {"feature": {}}, "red": {"feature": {}}}
    data["BOTTOM"]["blue"]["participantId"] = {}
    data["BOTTOM"]["red"]["participantId"] = {}

    for p in match["info"]["participants"]:
        team, side = p["teamId"], "blue" if p["teamId"] == 100 else "red"
        pos, pid = p["teamPosition"], p["participantId"]

        if pos in ["TOP", "MIDDLE", "JUNGLE"]:
            data[pos][side]["participantId"] = pid
        elif pos == "BOTTOM":
            data["BOTTOM"][side]["participantId"]["ADC"] = pid
        elif pos == "UTILITY":
            data["BOTTOM"][side]["participantId"]["SPT"] = pid
    return data


def calculate_tci_target(timeline, data, target_lane, minute):
    if minute >= len(timeline["info"]["frames"]): return None, None
    pf = timeline["info"]["frames"][minute]["participantFrames"]

    blue_gold, red_gold = sum(pf[str(i)]["totalGold"] for i in range(1, 6)), sum(
        pf[str(i)]["totalGold"] for i in range(6, 11))
    blue_xp, red_xp = sum(pf[str(i)]["xp"] for i in range(1, 6)), sum(pf[str(i)]["xp"] for i in range(6, 11))
    tot_gold, tot_xp = blue_gold + red_gold, blue_xp + red_xp

    if target_lane == "BOTTOM":
        b_adc, b_spt = data["BOTTOM"]["blue"]["participantId"]["ADC"], data["BOTTOM"]["blue"]["participantId"]["SPT"]
        r_adc, r_spt = data["BOTTOM"]["red"]["participantId"]["ADC"], data["BOTTOM"]["red"]["participantId"]["SPT"]
        b_l_gold = pf[str(b_adc)]["totalGold"] + pf[str(b_spt)]["totalGold"]
        b_l_xp = pf[str(b_adc)]["xp"] + pf[str(b_spt)]["xp"]
        r_l_gold = pf[str(r_adc)]["totalGold"] + pf[str(r_spt)]["totalGold"]
        r_l_xp = pf[str(r_adc)]["xp"] + pf[str(r_spt)]["xp"]
    else:
        b_pid, r_pid = data[target_lane]["blue"]["participantId"], data[target_lane]["red"]["participantId"]
        b_l_gold, b_l_xp = pf[str(b_pid)]["totalGold"], pf[str(b_pid)]["xp"]
        r_l_gold, r_l_xp = pf[str(r_pid)]["totalGold"], pf[str(r_pid)]["xp"]

    tci_b = (((b_l_gold - r_l_gold) / tot_gold) * 0.6) + (((b_l_xp - r_l_xp) / tot_xp) * 0.4)
    return tci_b, -tci_b


def check_effective_followup(all_events, base_timestamp, team_side):
    team_pids = [1, 2, 3, 4, 5] if team_side == "blue" else [6, 7, 8, 9, 10]
    for event in all_events:
        t_diff = event.get("timestamp", 0) - base_timestamp
        if 0 < t_diff <= 60000 and event["type"] in ["ELITE_MONSTER_KILL", "BUILDING_KILL"]:
            if event.get("killerId", 0) in team_pids: return True
        elif t_diff > 60000:
            break
    return False


def get_jungle_feature(timeline, data):
    b_pid = data["JUNGLE"]["blue"]["participantId"]
    r_pid = data["JUNGLE"]["red"]["participantId"]
    jgMinionDiff, effectiveGankRate, dmgDiff = [], [], []
    b_gank, r_gank, b_eff, r_eff = 0, 0, 0, 0
    all_events = [e for f in timeline["info"]["frames"] for e in f.get("events", [])]

    for frame in timeline["info"]["frames"]:
        pf = frame["participantFrames"]
        # 피처 1: 정글 몹 처치 격차
        jgMinionDiff.append(pf[str(b_pid)].get("jungleMinionsKilled", 0) - pf[str(r_pid)].get("jungleMinionsKilled", 0))
        # 피처 2: 챔피언에게 가한 누적 딜량 격차 (새로 추가됨!)
        b_dmg = pf[str(b_pid)].get("damageStats", {}).get("totalDamageDoneToChampions", 0)
        r_dmg = pf[str(r_pid)].get("damageStats", {}).get("totalDamageDoneToChampions", 0)
        dmgDiff.append(b_dmg - r_dmg)

        # 피처 3: 유효 갱킹률
        for e in frame.get("events", []):
            if e["type"] == "CHAMPION_KILL":
                if e.get("killerId") == b_pid or b_pid in e.get("assistingParticipantIds", []):
                    b_gank += 1
                    if check_effective_followup(all_events, e["timestamp"], "blue"): b_eff += 1
                if e.get("killerId") == r_pid or r_pid in e.get("assistingParticipantIds", []):
                    r_gank += 1
                    if check_effective_followup(all_events, e["timestamp"], "red"): r_eff += 1

        effectiveGankRate.append(b_eff / b_gank if b_gank > 0 else 0)

    # 파이썬 리스트 슬라이싱 편의성을 위해 블루팀 관점 피처만 반환 (레드팀은 부호 반대)
    return {"jgMinionDiffCum": jgMinionDiff, "dmgDiffCum": dmgDiff, "effectiveGankRateCum": effectiveGankRate}


def build_dataset():
    with open("match_0.json", "r", encoding="utf-8") as f:
        matches = json.load(f)
    with open("timeline_0.json", "r", encoding="utf-8") as f:
        timelines = json.load(f)

    dataset_rows = []
    target_minutes = [10, 15]  # 10분, 15분 데이터 추출

    for m, tl in zip(matches, timelines):
        try:
            match_id = m["metadata"]["matchId"]
            data = get_default_match_data(m, tl, [])
            jg_features = get_jungle_feature(tl, data)

            for min_idx in target_minutes:
                tci_b, tci_r = calculate_tci_target(tl, data, "JUNGLE", min_idx)
                if tci_b is None: continue

                # 블루 정글
                dataset_rows.append({
                    "matchId": match_id, "team": "blue", "minute": min_idx,
                    "jgMinionDiffCum": jg_features["jgMinionDiffCum"][min_idx],
                    "dmgDiffCum": jg_features["dmgDiffCum"][min_idx],
                    "effectiveGankRateCum": jg_features["effectiveGankRateCum"][min_idx],
                    "tci_target": tci_b
                })
                # 레드 정글 (격차 지표들은 부호 반대로, 비율 지표는 독립 계산이 맞으나 예시상 축약)
                dataset_rows.append({
                    "matchId": match_id, "team": "red", "minute": min_idx,
                    "jgMinionDiffCum": -jg_features["jgMinionDiffCum"][min_idx],
                    "dmgDiffCum": -jg_features["dmgDiffCum"][min_idx],
                    "effectiveGankRateCum": 0,  # 레드는 구조상 생략 (실제 적용시 별도 할당 필요)
                    "tci_target": tci_r
                })
        except Exception:
            continue

    df = pd.DataFrame(dataset_rows)
    df = df[df['team'] == 'blue']  # 정확도 검증을 위해 블루팀(정방향) 데이터만 학습에 사용
    df.to_csv("jungle_tci_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"전체 피처 데이터셋 생성 완료! (행: {len(df)})")


if __name__ == "__main__":
    build_dataset()