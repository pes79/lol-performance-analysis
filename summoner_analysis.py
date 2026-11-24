"""
summoner_analysis.py — TCI로 '소환사 실력'을 설명하는 분석
필요: lane_tci_dataset.csv, out/lane_players.csv (extract_players.py), results/ (train_lane_models.py)

결과 ① 승패와 라인전 실력은 다르다: 진 팀 라이너의 몇 %가 라인전을 이기고 있었나 (make_figures.py 그림 11)
결과 ② 소환사 단위: 여러 판의 평균 TCI가 승률보다 그 사람의 실력을 '꾸준히' 보여 주는가
   - 판을 홀수·짝수 번째로 반씩 나눠, 한쪽 평균이 다른 쪽 평균을 얼마나 잘 맞히는지(반분 신뢰도) 비교
   - 평균 TCI × 승률 지도로 "라인전은 강한데 승률은 낮은" 소환사 찾기

사용법: python summoner_analysis.py [최소판수=20]
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import platform

MIN_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
KFONT = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system(), "Noto Sans CJK JP")
plt.rcParams.update({"font.family": KFONT, "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.titlesize": 13, "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"
LANES = ["TOP", "MIDDLE", "JUNGLE", "BOTTOM"]
LANE_KO = {"TOP": "탑", "MIDDLE": "미드", "JUNGLE": "정글", "BOTTOM": "바텀"}

# 1) 경기 × 라인별 라인전(5~20분) 평균 TCI와 기대 TCI(교차검증 예측) — 블루 관점
parts = []
for lane in LANES:
    keys = pd.read_csv(f"results/keys_{lane}.csv")
    keys["y"] = np.load(f"results/y_{lane}.npy")
    keys["pred"] = np.load(f"results/oof_{lane}_LightGBM.npy")
    g = keys.groupby("matchId")[["y", "pred"]].mean().reset_index()
    g["lane"] = lane
    parts.append(g)
lane_match = pd.concat(parts)

# 2) 소환사와 연결 (진영에 맞게 부호 뒤집기)
players = pd.read_csv("out/lane_players.csv")
df = players.merge(lane_match, on=["matchId", "lane"])
sign = np.where(df.side == "blue", 1, -1)
df["tci"] = df.y * sign                  # 내 라인의 라인전 평균 TCI
df["expected"] = df.pred * sign          # 내 기록으로 기대되는 TCI
df["gap"] = df.tci - df.expected         # 기록 대비 골드를 더 챙겼나
df["win"] = df.win.astype(float)

# 3) 소환사 × 주 라인 단위 집계 (MIN_GAMES 판 이상)
df = df.sort_values(["puuid", "lane", "matchId"])
df["half"] = df.groupby(["puuid", "lane"]).cumcount() % 2          # 홀수·짝수 번째 판
agg = df.groupby(["puuid", "lane"]).agg(games=("tci", "size"), tci=("tci", "mean"),
                                        winrate=("win", "mean"), gap=("gap", "mean")).reset_index()
agg = agg[agg.games >= MIN_GAMES]
halves = df.merge(agg[["puuid", "lane"]]).groupby(["puuid", "lane", "half"])[["tci", "win", "gap"]].mean().unstack("half")

stats = {"min_games": MIN_GAMES, "summoner_lanes": int(len(agg)), "summoners": int(agg.puuid.nunique()),
         "games_median": float(agg.games.median()),
         "corr_tci_winrate": float(agg[["tci", "winrate"]].corr().iloc[0, 1]),
         "split_half": {"tci": float(halves["tci"].corr().iloc[0, 1]),
                        "winrate": float(halves["win"].corr().iloc[0, 1]),
                        "gap": float(halves["gap"].corr().iloc[0, 1])}}
q_hi = agg.tci.quantile(0.75)
hidden = agg[(agg.tci >= q_hi) & (agg.winrate < 0.5)]
stats["hidden_gems"] = {"count": int(len(hidden)), "share_of_top25": float(len(hidden) / (agg.tci >= q_hi).sum()),
                        "tci_top25_cut": float(q_hi)}
stats["by_lane"] = {l: {"n": int((agg.lane == l).sum()),
                        "corr_tci_winrate": float(agg[agg.lane == l][["tci", "winrate"]].corr().iloc[0, 1])}
                    for l in LANES if (agg.lane == l).sum() > 5}

# 그림 12: 반분 신뢰도 — TCI vs 승률
fig, ax = plt.subplots(figsize=(8, 3.8))
labels = ["승률", "평균 TCI\n(라인전 골드 우위)", "기대 대비 실제\n(골드 효율)"]
vals = [stats["split_half"]["winrate"], stats["split_half"]["tci"], stats["split_half"]["gap"]]
ax.barh(labels, vals, color=["#9fc0e8", BLUE, "#9fc0e8"], height=0.55)
for i, v in enumerate(vals):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=11)
ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
ax.set_xlabel("판을 반씩 나눴을 때 두 쪽 평균의 상관 (1에 가까울수록 꾸준한 실력 지표)")
ax.set_title(f"어떤 지표가 소환사 실력을 꾸준히 보여 주나 ({stats['summoner_lanes']:,}명·라인, 각 {MIN_GAMES}판 이상)")
fig.savefig(f"{OUT}/12_split_half.png"); plt.close(fig)

# 그림 13: 소환사 지도 — 평균 TCI × 승률
fig, ax = plt.subplots(figsize=(8.5, 5.6))
ax.scatter(agg.tci * 100, agg.winrate * 100, s=14, color=BLUE, alpha=0.35, edgecolor="none")
ax.scatter(hidden.tci * 100, hidden.winrate * 100, s=22, color=ORANGE, alpha=0.9, edgecolor="none",
           label=f"라인전 상위 25%인데 승률 50% 미만 ({len(hidden)}명·라인)")
ax.axvline(q_hi * 100, color=INK2, ls="--", lw=1); ax.axhline(50, color=INK2, ls="--", lw=1)
z = np.polyfit(agg.tci * 100, agg.winrate * 100, 1); xs = np.linspace(agg.tci.min() * 100, agg.tci.max() * 100, 50)
ax.plot(xs, np.polyval(z, xs), color="#0b0b0b", lw=1.5)
ax.set_xlabel("소환사의 평균 라인전 TCI (%)"); ax.set_ylabel("승률 (%)")
ax.set_title(f"소환사 실력 지도: 라인전 TCI × 승률 (상관 {stats['corr_tci_winrate']:.2f})")
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.savefig(f"{OUT}/13_summoner_map.png"); plt.close(fig)

agg.to_csv("results/summoner_stats.csv", index=False, encoding="utf-8-sig")
json.dump(stats, open("results/summoner_stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1))
