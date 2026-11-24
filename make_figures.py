"""README용 그래프 생성 (15,276경기 CSV 버전)
먼저 extract_all.py, train_lane_models.py 를 실행한 뒤 사용
사용법: python make_figures.py  ->  figures/ 폴더에 그래프 생성 (02번 한 경기 예시는 JSON 필요)
"""
import json, os, sys, platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

# 한글 폰트: Windows=Malgun Gothic, macOS=AppleGothic, Linux=Noto Sans CJK
import platform
KFONT = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system(), "Noto Sans CJK JP")
plt.rcParams.update({
    "font.family": KFONT, "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": "#0b0b0b",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED_TEAM = "#e34948"
INK2 = "#52514e"


LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]
lane_col = {"TOP": BLUE, "JUNGLE": ORANGE, "MIDDLE": AQUA, "BOTTOM": YELLOW}
lane_ko = {"TOP": "탑", "JUNGLE": "정글", "MIDDLE": "미드", "BOTTOM": "바텀(원딜+서폿)"}
stats = {}
summ = pd.read_csv(f"{DATA}/match_summary.csv")
lm = pd.read_csv(f"{DATA}/lane_minute_gold.csv")
lm["tci"] = lm.goldDiff / lm.totalGold
N = len(summ)

# ---------- Fig 1: 라벨 후보 ----------
cands = {"총 골드\n(goldEarned)": "goldEarned", "총 경험치\n(champExperience)": "champExperience",
         "챔피언 딜량\n(DamageToChampions)": "totalDamageDealtToChampions", "킬 수\n(kills)": "kills",
         "받은 피해량\n(DamageTaken)": "totalDamageTaken"}
acc = {k: float(((summ[f"blue_{v}"] > summ[f"red_{v}"]) == summ.blueWin).mean()) for k, v in cands.items()}
stats["label_acc"] = {k.replace("\n", " "): v for k, v in acc.items()}
items = sorted(acc.items(), key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(8, 4.2))
cols = [BLUE if v == max(acc.values()) else "#9fc0e8" for _, v in items]
ax.barh([k for k, _ in items], [v * 100 for _, v in items], color=cols, height=0.55)
for i, (_, v) in enumerate(items):
    ax.text(v * 100 + 1, i, f"{v*100:.1f}%", va="center", color="#0b0b0b", fontsize=10)
ax.axvline(50, color=INK2, ls="--", lw=1)
ax.text(50.8, -0.45, "동전 던지기(50%)", color=INK2, fontsize=9)
ax.set_xlim(0, 108); ax.set_xlabel("종료 시점 팀 합계가 큰 쪽 = 승리 팀으로 맞힌 비율 (%)")
ax.set_title(f"라벨(y) 후보 비교: 어떤 지표가 승패를 가장 잘 반영하나 ({N:,}경기)")
ax.grid(axis="y", visible=False)
fig.savefig(f"{OUT}/01_label_candidates.png"); plt.close(fig)

# ---------- Fig 3: 라인별 |TCI| 추이 ----------
agg = lm[lm.minute.between(3, 30)].assign(a=lambda d: d.tci.abs()).groupby(["lane", "minute"]).a.mean().unstack(0)
stats["lane_gap"] = {l: {int(m): float(v) for m, v in agg[l].items() if m in (5, 10, 15, 20, 25, 30)} for l in LANES}
fig, ax = plt.subplots(figsize=(9, 4.6))
for lane in LANES:
    ax.plot(agg.index, agg[lane], color=lane_col[lane], lw=2, label=lane_ko[lane])
    ax.text(agg.index[-1] + 0.3, agg[lane].iloc[-1], lane_ko[lane], color="#0b0b0b", fontsize=9, va="center")
ax.set_xlabel("게임 시간 (분)"); ax.set_ylabel("평균 |TCI| (라인 간 골드 격차 크기)")
ax.set_title(f"라인별 격차는 언제 벌어지나 ({N:,}경기 평균, 3~30분)")
ax.legend(frameon=False, loc="lower right", ncol=2); ax.set_xlim(3, 33)
fig.savefig(f"{OUT}/03_lane_gap_over_time.png"); plt.close(fig)

# ---------- Fig 4: TCI 부호 vs 승패 ----------
res = []
for lane in LANES:
    for minute in [10, 15, 20]:
        s = lm[(lm.lane == lane) & (lm.minute == minute) & (lm.goldDiff != 0)]
        res.append({"lane": lane, "minute": minute, "acc": float(((s.tci > 0) == s.blueWin).mean()), "n": len(s)})
res = pd.DataFrame(res)
stats["tci_sign_vs_win"] = res.to_dict("records")
fig, ax = plt.subplots(figsize=(9, 4.4))
w = 0.26
mcol = {10: "#9fc0e8", 15: "#5f9be0", 20: BLUE}
for j, minute in enumerate([10, 15, 20]):
    s = res[res.minute == minute]
    x = np.arange(len(LANES)) + (j - 1) * w
    ax.bar(x, s.acc * 100, width=w - 0.03, color=mcol[minute], label=f"{minute}분")
    for xi, v in zip(x, s.acc):
        ax.text(xi, v * 100 + 1.2, f"{v*100:.0f}", ha="center", fontsize=8.5, color="#0b0b0b")
ax.axhline(50, color=INK2, ls="--", lw=1)
ax.set_xticks(np.arange(len(LANES)), [lane_ko[l] for l in LANES])
ax.set_ylim(0, 100); ax.set_ylabel("골드 앞선 쪽이 이긴 비율 (%)")
ax.set_title(f"특정 시점 라인 골드 우위 → 최종 승리로 이어졌나 ({N:,}경기)")
ax.legend(frameon=False, ncol=3, loc="upper left"); ax.grid(axis="x", visible=False)
fig.savefig(f"{OUT}/04_tci_sign_vs_win.png"); plt.close(fig)

# ---------- Fig 5: 골드 vs 경험치 섞기 ----------
s = lm[lm.minute.isin([10, 15, 20])]
g, x = s.goldDiff / s.totalGold, s.xpDiff / s.totalXp
ws = np.round(np.arange(0, 1.01, 0.1), 1)
wacc = [float((((w * g + (1 - w) * x) > 0) == s.blueWin).mean()) for w in ws]
stats["weight_scan"] = dict(zip(map(float, ws), wacc))
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(ws, np.array(wacc) * 100, color=BLUE, lw=2, marker="o", ms=7, mec="#fcfcfb", mew=2)
ax.scatter([1.0], [wacc[10] * 100], s=140, facecolor="none", edgecolor="#0b0b0b", lw=1.5, zorder=4)
ax.annotate("채택: 골드만 (w = 1.0)", (1.0, wacc[10] * 100), xytext=(0.62, wacc[10] * 100 - 0.4),
            fontsize=10, color="#0b0b0b", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xticks(ws, [f"{w:.1f}" for w in ws])
ax.set_xlabel("골드 가중치 w  (TCI = w·골드격차비율 + (1−w)·경험치격차비율)")
ax.set_ylabel("TCI 우위 팀 = 승리 팀 (%)")
ax.set_title(f"골드에 경험치를 섞으면 더 나을까? (4라인 × 10·15·20분 × {N:,}경기)")
fig.savefig(f"{OUT}/05_target_weight_scan.png"); plt.close(fig)

# ---------- 모델 결과 불러오기 ----------
LANE_ORDER = ["TOP", "MIDDLE", "JUNGLE", "BOTTOM"]
lane_df = pd.read_csv(f"{DATA}/lane_tci_dataset.csv")
met = pd.read_csv(f"{DATA}/results/model_metrics.csv")
best = pd.read_csv(f"{DATA}/results/best_models.csv").set_index("lane")
imp = pd.read_csv(f"{DATA}/results/feature_importance.csv")
sys.path.insert(0, DATA)
from train_lane_models import LANE_FEATURES


def oof(lane, model):
    keys = pd.read_csv(f"{DATA}/results/keys_{lane}.csv")
    return keys.assign(y=np.load(f"{DATA}/results/y_{lane}.npy"), pred=np.load(f"{DATA}/results/oof_{lane}_{model}.npy"))


# ---------- Fig 6: 라인별 피처-TCI 상관계수 ----------
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
for ax, lane in zip(axes, LANE_ORDER):
    d = lane_df[lane_df.lane == lane]
    cor = d[LANE_FEATURES[lane]].corrwith(d.tci).sort_values()
    ax.barh(cor.index, cor.values, color=[BLUE if v > 0 else RED_TEAM for v in cor.values], height=0.6)
    for i, v in enumerate(cor.values):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color=INK2, lw=1); ax.set_xlim(-1.05, 1.05)
    ax.set_title(lane_ko[lane]); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=9)
fig.suptitle(f"라인별 피처와 TCI의 피어슨 상관계수 (5~20분, 라인당 {len(lane_df)//4:,}줄)", fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/06_feature_correlation.png"); plt.close(fig)

# ---------- Fig 7: 모델 비교 (RMSE) ----------
models = ["LinearRegression", "Ridge", "RandomForest", "LightGBM", "MLP"]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
for ax, lane in zip(axes, LANE_ORDER):
    s = met[met.lane == lane].set_index("model")
    vals = s.loc[models, "RMSE"].values * 1000; errs = np.zeros(len(models))
    cols = [BLUE if mdl == best.loc[lane, "model"] else "#9fc0e8" for mdl in models]
    ax.barh(models, vals, color=cols, height=0.6)
    base = s.loc["Baseline(평균)", "RMSE"] * 1000
    for i, v in enumerate(vals):
        ax.text(v + errs[i] + 0.3, i, f"{v:.2f}", va="center", fontsize=8.5)
    ax.set_title(f"{lane_ko[lane]}  (베이스라인 {base:.1f})", fontsize=11)
    ax.set_xlim(0, max(vals) * 1.35); ax.grid(axis="y", visible=False)
    ax.set_xlabel("RMSE (×0.001, 낮을수록 좋음)", fontsize=9)
fig.suptitle("라인별 모델 비교 — 경기 단위 5-fold 교차검증, 진한 막대 = 최종 선정",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(f"{OUT}/07_model_comparison.png"); plt.close(fig)

# ---------- Fig 8: 최종 모델 실제 vs 예측 ----------
fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
for ax, lane in zip(axes, LANE_ORDER):
    b = best.loc[lane]
    s = oof(lane, b.model)
    lim = 0.10
    ax.hexbin(s.y, s.pred, gridsize=70, extent=(-lim, lim, -lim, lim), bins="log", cmap="Blues", mincnt=1)
    ax.plot([-lim, lim], [-lim, lim], color=INK2, ls="--", lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(f"{lane_ko[lane]} — {b.model}\nR² {b.R2:.2f} · 부호 일치 {b.SignAcc*100:.0f}%", fontsize=11)
    ax.grid(False)
    ax.set_xlabel("실제 TCI", fontsize=9)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("예측 TCI (교차검증)")
fig.suptitle("최종 모델 실제 vs 예측 (진할수록 경기가 많은 구간)", fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(f"{OUT}/08_pred_vs_actual.png"); plt.close(fig)

# ---------- Fig 9: 최종 모델 Permutation Importance ----------
fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
for ax, lane in zip(axes, LANE_ORDER):
    s = imp[imp.lane == lane].set_index("feature").perm_importance
    s = (s / s.sum() * 100).sort_values()
    ax.barh(s.index, s.values, color=[BLUE if v == s.max() else "#9fc0e8" for v in s.values], height=0.6)
    for i, v in enumerate(s.values):
        ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=8.5)
    ax.set_xlim(0, s.max() * 1.3)
    ax.set_title(f"{lane_ko[lane]} — {best.loc[lane, 'model']}", fontsize=11)
    ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("중요도 비중 (%)", fontsize=9)
fig.suptitle("최종 모델의 변수 중요도 (Permutation Importance: 피처를 섞었을 때 MSE 증가량)",
             fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/09_feature_importance.png"); plt.close(fig)

# ---------- Fig 10: 시간대별 오차 ----------
fig, ax = plt.subplots(figsize=(9, 4.2))
for lane in LANE_ORDER:
    s = oof(lane, best.loc[lane, "model"])
    e = s.assign(se=(s.y - s.pred) ** 2).groupby("minute").se.mean() ** 0.5 * 1000
    ax.plot(e.index, e.values, color=lane_col[lane], lw=2, label=lane_ko[lane])
ax.set_xlabel("게임 시간 (분)"); ax.set_ylabel("RMSE (×0.001)")
ax.set_title("최종 모델의 시간대별 예측 오차")
ax.legend(frameon=False, loc="upper left", ncol=2); ax.set_xlim(5, 20)
fig.savefig(f"{OUT}/10_error_by_minute.png"); plt.close(fig)


# ---------- Fig 11: 진 팀 라이너도 라인전은 이겼다 ----------
g = lane_df.groupby(["matchId", "lane"]).agg(tci=("tci", "mean"), blueWin=("blueWin", "first")).reset_index()
g["loser"] = np.where(g.blueWin, -g.tci, g.tci)          # 진 팀 관점의 라인전(5~20분) 평균 TCI
ahead = g.groupby("lane").loser.apply(lambda x: (x > 0).mean())
big = g.groupby("lane").loser.apply(lambda x: (x >= 0.02).mean())
per_match = g.assign(a=g.loser > 0).groupby("matchId").a.sum()
stats["loser_ahead"] = {"all": float((g.loser > 0).mean()), "big_all": float((g.loser >= 0.02).mean()),
                        "by_lane": ahead.to_dict(), "big_by_lane": big.to_dict(),
                        "match_any": float((per_match >= 1).mean()), "match_2plus": float((per_match >= 2).mean())}
fig, ax = plt.subplots(figsize=(9, 4.4))
x = np.arange(len(LANE_ORDER))
ax.bar(x - 0.18, [ahead[l] * 100 for l in LANE_ORDER], width=0.34, color="#9fc0e8", label="라인전 평균 TCI > 0 (앞섬)")
ax.bar(x + 0.18, [big[l] * 100 for l in LANE_ORDER], width=0.34, color=BLUE, label="+2% 이상 (확실히 앞섬)")
for i, l in enumerate(LANE_ORDER):
    ax.text(i - 0.18, ahead[l] * 100 + 1, f"{ahead[l]*100:.0f}%", ha="center", fontsize=10)
    ax.text(i + 0.18, big[l] * 100 + 1, f"{big[l]*100:.0f}%", ha="center", fontsize=10)
ax.set_xticks(x, [lane_ko[l] for l in LANE_ORDER]); ax.set_ylim(0, 55)
ax.set_ylabel("진 팀 라이너 중 비율 (%)")
ax.set_title(f"진 팀 라이너도 라인전은 이기고 있었다 ({N:,}경기, 5~20분 평균)")
ax.legend(frameon=False, loc="upper right"); ax.grid(axis="x", visible=False)
fig.savefig(f"{OUT}/11_loser_lane_ahead.png"); plt.close(fig)

json.dump(stats, open(f"{OUT}/stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
print(json.dumps(stats, ensure_ascii=False, indent=1, default=float)[:3000])
