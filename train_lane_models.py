"""
4대 라인별 TCI 예측 모델 비교 (15,276경기 버전)
- 입력: lane_tci_dataset.csv (extract_all.py 결과)
- 모델: 평균 예측(베이스라인) / 선형회귀 / Ridge / RandomForest / LightGBM / MLP 신경망
- 손실함수: MSE (평가지표 RMSE, MAE, R², 부호 일치율)
- 검증: 경기 단위 GroupKFold(5) — 같은 경기의 다른 분이 학습/테스트에 섞이지 않도록 함

사용법: python train_lane_models.py            # 4개 라인 전부
        python train_lane_models.py TOP        # 한 라인만
결과: results/ 폴더
"""
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)
SEED = 42

LANE_FEATURES = {
    "TOP":    ["csDiff", "dmgDealtDiff", "dmgTakenDiff", "ccDiff", "killDiff", "deathDiff",
               "soloKillDiff", "plateDiff"],
    "MIDDLE": ["csDiff", "dmgDealtDiff", "dmgTakenDiff", "ccDiff", "killDiff", "deathDiff",
               "soloKillDiff", "plateDiff", "roamKADiff"],
    "JUNGLE": ["csDiff", "dmgDealtDiff", "dmgTakenDiff", "ccDiff", "killDiff", "deathDiff",
               "jgCsDiff", "assistDiff", "epicDiff", "effGankDiff"],
    "BOTTOM": ["csDiff", "dmgDealtDiff", "dmgTakenDiff", "ccDiff", "killDiff", "deathDiff",
               "plateDiff", "assistDiff", "wardPlacedDiff", "wardKillDiff"],
}
MODEL_NAMES = ["Baseline(평균)", "LinearRegression", "Ridge", "RandomForest", "LightGBM", "MLP"]


def get_models(seed=SEED):
    return {
        "Baseline(평균)": DummyRegressor(strategy="mean"),
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=20,
                                              max_samples=0.3, random_state=seed, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                      min_child_samples=50, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=-1),
        # 신경망은 입력(z-score)과 타겟(y)을 모두 표준화한 뒤 학습
        "MLP": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(),
                                    MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=512,
                                                 learning_rate_init=1e-3, max_iter=200, early_stopping=True,
                                                 n_iter_no_change=8, random_state=seed)),
            transformer=StandardScaler()),
    }


def evaluate(y, p):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p)),
            "SignAcc": float(np.mean(np.sign(y) == np.sign(p)))}


def run_lane(df, lane, models=MODEL_NAMES):
    feats = LANE_FEATURES[lane]
    d = df[df.lane == lane].reset_index(drop=True)
    X = d[feats + ["minute"]].values.astype(np.float32)
    y = d.tci.values
    groups = d.matchId.values
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    out = {}
    for name in models:
        t0 = time.time()
        pred = np.zeros(len(y))
        for tr, te in folds:
            model = get_models()[name]
            model.fit(X[tr], y[tr])
            pred[te] = model.predict(X[te])
        row = {"lane": lane, "model": name, **evaluate(y, pred), "sec": round(time.time() - t0, 1)}
        out[name] = row
        np.save(f"results/oof_{lane}_{name}.npy", pred.astype(np.float32))
        print(f"{lane:7s} {name:17s} RMSE={row['RMSE']*1000:.2f}  R2={row['R2']:.3f}  "
              f"Sign={row['SignAcc']:.3f}  ({row['sec']}s)", flush=True)
    path = f"results/metrics_{lane}.json"
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(out)                                   # 라인을 나눠 돌려도 결과가 합쳐지도록
    json.dump(prev, open(path, "w"), ensure_ascii=False, indent=1)
    np.save(f"results/y_{lane}.npy", y.astype(np.float32))
    d[["matchId", "minute"]].to_csv(f"results/keys_{lane}.csv", index=False)


def finalize(df):
    """라인별 결과를 모아 최종 모델 선정 + 변수 중요도"""
    rows = []
    for lane in LANE_FEATURES:
        rows += list(json.load(open(f"results/metrics_{lane}.json")).values())
    met = pd.DataFrame(rows)
    met.to_csv("results/model_metrics.csv", index=False, encoding="utf-8-sig")
    # 최종 모델: RMSE 최저. 단, LightGBM이 1등과 1% 이내면 더 빠르고 안정적인 LightGBM 선택
    picks = []
    for lane, g in met[met.model != "Baseline(평균)"].groupby("lane"):
        top = g.sort_values("RMSE").iloc[0]
        lgbm = g[g.model == "LightGBM"].iloc[0]
        picks.append(lgbm if lgbm.RMSE <= top.RMSE * 1.01 else top)
    best = pd.DataFrame(picks)
    best.to_csv("results/best_models.csv", index=False, encoding="utf-8-sig")
    print(best[["lane", "model", "RMSE", "R2", "SignAcc"]].to_string(index=False))

    imp_rows = []
    for _, b in best.iterrows():
        lane, feats = b.lane, LANE_FEATURES[b.lane]
        d = df[df.lane == lane]
        X, y = d[feats + ["minute"]].values.astype(np.float32), d.tci.values
        mdl = get_models()[b.model].fit(X, y)
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(y), size=min(30000, len(y)), replace=False)   # 3만 줄로 측정
        pi = permutation_importance(mdl, X[idx], y[idx], n_repeats=5, random_state=SEED,
                                    scoring="neg_mean_squared_error")
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, y)
        for f, v, c in zip(feats + ["minute"], pi.importances_mean, ridge[-1].coef_):
            imp_rows.append({"lane": lane, "model": b.model, "feature": f, "perm_importance": v, "ridge_coef": c})
        print(f"{lane} 중요도 계산 완료", flush=True)
    pd.DataFrame(imp_rows).to_csv("results/feature_importance.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    df = pd.read_csv("lane_tci_dataset.csv")
    args = sys.argv[1:]
    if args and args[0] == "finalize":
        finalize(df)
    elif args:
        run_lane(df, args[0], args[1:] or MODEL_NAMES)
    else:
        for lane in LANE_FEATURES:
            run_lane(df, lane)
        finalize(df)
