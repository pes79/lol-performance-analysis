import json


def get_default_match_data(match, timeline, challengers):
    """매치 기본 정보 및 포지션별 참가자 ID 세팅 (원본 코드와 동일)"""
    data = {}
    data["matchId"] = match["metadata"]["matchId"]
    data["gameDurationSec"] = match["info"]["gameDuration"]
    data["numFrames"] = len(timeline["info"]["frames"])
    tmp_teamId = match["info"]["participants"][0]["teamId"]
    tmp_win = match["info"]["participants"][0]["win"]
    data["blueWin"] = tmp_win if tmp_teamId == 100 else (not tmp_win)

    lanes = ["TOP", "MIDDLE", "JUNGLE", "BOTTOM"]
    for lane in lanes:
        data[lane] = {"blue": {"feature": {}}, "red": {"feature": {}}}
    data["BOTTOM"]["blue"]["participantId"] = {}
    data["BOTTOM"]["blue"]["challengerPuuid"] = {}
    data["BOTTOM"]["red"]["participantId"] = {}
    data["BOTTOM"]["red"]["challengerPuuid"] = {}

    for p in match["info"]["participants"]:
        team = p["teamId"]
        side = "blue" if team == 100 else "red"
        pos = p["teamPosition"]
        pid = p["participantId"]
        puid = p["puuid"]
        is_chall = puid in challengers
        chall = puid if is_chall else None

        if pos in ["TOP", "MIDDLE", "JUNGLE"]:
            data[pos][side]["teamId"] = team
            data[pos][side]["participantId"] = pid
            data[pos][side]["challengerPuuid"] = chall
        elif pos == "BOTTOM":
            data["BOTTOM"][side]["teamId"] = team
            data["BOTTOM"][side]["participantId"]["ADC"] = pid
            data["BOTTOM"][side]["challengerPuuid"]["ADC"] = chall
        elif pos == "UTILITY":
            data["BOTTOM"][side]["teamId"] = team
            data["BOTTOM"][side]["participantId"]["SPT"] = pid
            data["BOTTOM"][side]["challengerPuuid"]["SPT"] = chall

    return data


def calculate_tci_target(timeline, data, target_lane, minute):
    """
    [TCI 타겟 변수 산출] 특정 시간대(minute)의 시간 가중 상대적 자원 지배력 계산
    반환: (Blue 진영 TCI 값, Red 진영 TCI 값)
    """
    if minute >= len(timeline["info"]["frames"]):
        return None, None

    frame = timeline["info"]["frames"][minute]
    pf = frame["participantFrames"]

    blue_team_gold = sum([pf[str(i)]["totalGold"] for i in range(1, 6)])
    red_team_gold = sum([pf[str(i)]["totalGold"] for i in range(6, 11)])
    blue_team_xp = sum([pf[str(i)]["xp"] for i in range(1, 6)])
    red_team_xp = sum([pf[str(i)]["xp"] for i in range(6, 11)])

    total_game_gold = blue_team_gold + red_team_gold
    total_game_xp = blue_team_xp + red_team_xp

    if target_lane == "BOTTOM":
        blue_pid_adc = data[target_lane]["blue"]["participantId"]["ADC"]
        blue_pid_spt = data[target_lane]["blue"]["participantId"]["SPT"]
        red_pid_adc = data[target_lane]["red"]["participantId"]["ADC"]
        red_pid_spt = data[target_lane]["red"]["participantId"]["SPT"]

        blue_lane_gold = pf[str(blue_pid_adc)]["totalGold"] + pf[str(blue_pid_spt)]["totalGold"]
        blue_lane_xp = pf[str(blue_pid_adc)]["xp"] + pf[str(blue_pid_spt)]["xp"]
        red_lane_gold = pf[str(red_pid_adc)]["totalGold"] + pf[str(red_pid_spt)]["totalGold"]
        red_lane_xp = pf[str(red_pid_adc)]["xp"] + pf[str(red_pid_spt)]["xp"]
    else:
        blue_pid = data[target_lane]["blue"]["participantId"]
        red_pid = data[target_lane]["red"]["participantId"]

        blue_lane_gold = pf[str(blue_pid)]["totalGold"]
        blue_lane_xp = pf[str(blue_pid)]["xp"]
        red_lane_gold = pf[str(red_pid)]["totalGold"]
        red_lane_xp = pf[str(red_pid)]["xp"]

    blue_gold_diff = blue_lane_gold - red_lane_gold
    blue_xp_diff = blue_lane_xp - red_lane_xp

    # 0.6 (골드) + 0.4 (경험치) 가중치 적용 및 전체 자원 대비 비율
    tci_blue = ((blue_gold_diff / total_game_gold) * 0.6) + ((blue_xp_diff / total_game_xp) * 0.4)
    tci_red = -tci_blue  # 상대평가이므로 레드는 블루의 반대

    return tci_blue, tci_red


def get_jungle_feature(timeline, data):
    """
    [JUNGLE 특화 피처 추출] 격차 지표 및 유효 갱킹 이벤트 누적 추출
    """
    bluePid = data["JUNGLE"]["blue"]["participantId"]
    redPid = data["JUNGLE"]["red"]["participantId"]

    # 1. 격차(Gap) 지표: 정글 몬스터 처치 격차
    jgMinionDiffCumBlue, jgMinionDiffCumRed = [], []

    # 2. 이벤트(Event) 지표: 유효 갱킹률
    # 갱킹 시도(킬/어시 관여) 횟수 및 1분 내 오브젝트 획득 횟수 추적
    effectiveGankRateBlue, effectiveGankRateRed = [], []
    blue_gank_attempts, red_gank_attempts = 0, 0
    blue_effective_ganks, red_effective_ganks = 0, 0

    # 향후 프레임을 룩어헤드(Look-ahead)하기 위해 전체 이벤트를 평탄화(Flatten)
    all_events = []
    for frame in timeline["info"]["frames"]:
        all_events.extend(frame.get("events", []))

    for frame_idx, frame in enumerate(timeline["info"]["frames"]):
        pf = frame["participantFrames"]

        # 격차 지표 업데이트
        blue_jg = pf[str(bluePid)].get("jungleMinionsKilled", 0)
        red_jg = pf[str(redPid)].get("jungleMinionsKilled", 0)
        jgMinionDiffCumBlue.append(blue_jg - red_jg)
        jgMinionDiffCumRed.append(red_jg - blue_jg)

        # 해당 프레임(1분 내)에서 발생한 이벤트 확인
        events = frame.get("events", [])
        for i, event in enumerate(events):
            if event["type"] == "CHAMPION_KILL":
                killer = event.get("killerId")
                assists = event.get("assistingParticipantIds", [])
                timestamp = event.get("timestamp")

                # Blue 정글 관여
                if killer == bluePid or bluePid in assists:
                    blue_gank_attempts += 1
                    # 앞으로 60초(60000ms) 내 오브젝트 획득 확인
                    if check_effective_followup(all_events, timestamp, data, team_side="blue"):
                        blue_effective_ganks += 1

                # Red 정글 관여
                if killer == redPid or redPid in assists:
                    red_gank_attempts += 1
                    if check_effective_followup(all_events, timestamp, data, team_side="red"):
                        red_effective_ganks += 1

        # 프레임 단위 누적 유효 갱킹률 산출
        rate_blue = blue_effective_ganks / blue_gank_attempts if blue_gank_attempts > 0 else 0
        rate_red = red_effective_ganks / red_gank_attempts if red_gank_attempts > 0 else 0
        effectiveGankRateBlue.append(rate_blue)
        effectiveGankRateRed.append(rate_red)

    JUNGLE_blue = {
        "jgMinionDiffCum": jgMinionDiffCumBlue,
        "effectiveGankRateCum": effectiveGankRateBlue
    }
    JUNGLE_red = {
        "jgMinionDiffCum": jgMinionDiffCumRed,
        "effectiveGankRateCum": effectiveGankRateRed
    }

    data["JUNGLE"]["blue"]["feature"].update(JUNGLE_blue)
    data["JUNGLE"]["red"]["feature"].update(JUNGLE_red)

    return JUNGLE_blue, JUNGLE_red


def check_effective_followup(all_events, base_timestamp, data, team_side):
    """갱킹(킬) 발생 후 60초 내 같은 팀이 에픽 몬스터나 건물을 처치했는지 확인"""
    team_pids = [1, 2, 3, 4, 5] if team_side == "blue" else [6, 7, 8, 9, 10]

    for event in all_events:
        time_diff = event.get("timestamp", 0) - base_timestamp
        # 아직 base_timestamp 이전이거나 같은 이벤트면 패스
        if time_diff <= 0:
            continue
        # 60초 초과 시 실패
        if time_diff > 60000:
            return False

        if event["type"] in ["ELITE_MONSTER_KILL", "BUILDING_KILL"]:
            killer = event.get("killerId", 0)
            if killer in team_pids:
                return True
    return False


def main():
    # 파일이 존재하는 경우를 가정하고 실행
    # (주의: match_0.json 등의 샘플 파일이 동일 경로에 있어야 동작합니다)
    try:
        with open("match_0.json", "r", encoding="utf-8") as f:
            match = json.load(f)
        with open("timeline_0.json", "r", encoding="utf-8") as f:
            timeline = json.load(f)
        with open("puuid_challengers.txt", "r", encoding="utf-8") as f:
            challengers = f.read().splitlines()

        m = match[0]
        tl = timeline[0]

        # 1. 기본 데이터 틀 초기화
        data = get_default_match_data(m, tl, challengers)

        # 2. 정글 피처(유효 갱킹 등) 추출 테스트
        get_jungle_feature(tl, data)
        print("\n\n [JUNGLE Feature 추출 완료]")
        print(
            f"Blue Effective Gank Rates: {data['JUNGLE']['blue']['feature']['effectiveGankRateCum'][:15]}")  # 15분까지 확인

        # 3. 타겟 변수(TCI) 추출 테스트 (15분 기준)
        tci_b, tci_r = calculate_tci_target(tl, data, target_lane="JUNGLE", minute=15)
        print(f"\n\n [15분 기준 JUNGLE TCI Target 산출]")
        print(f"Blue TCI: {tci_b:.6f} | Red TCI: {tci_r:.6f}")

    except Exception as e:
        print(f"파일 로드 또는 파싱 실패. 오류 메시지: {e}")


if __name__ == "__main__":
    main()