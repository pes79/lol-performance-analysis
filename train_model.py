import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Windows 한글 폰트 설정 (폰트 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def train_tci_model():
    try:
        df = pd.read_csv("jungle_tci_dataset.csv")
    except FileNotFoundError:
        print("CSV 파일을 찾을 수 없습니다.")
        return

    # 타겟과 무관한 식별자 컬럼 제외하고 모든 피처 자동 인식
    features = [col for col in df.columns if col not in ['matchId', 'team', 'minute', 'tci_target']]
    target = "tci_target"

    X = df[features]
    y = df[target]

    # 데이터가 60개로 적으므로 test_size를 조금 줄여서 학습에 집중
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n[최종 TCI 예측 모델 평가 결과]")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R^2  : {r2:.4f} (설명력 {r2 * 100:.1f}%)")

    # 1. 변수 중요도 시각화 및 저장
    importance_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("\n[변수 중요도]")
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('TCI 변수 중요도 (Feature Importance)', fontsize=16)
    plt.xlabel('중요도 점수', fontsize=12)
    plt.ylabel('입력 변수', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)

    # 2. 예측값 vs 실제값 산점도 시각화 및 저장
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue', edgecolor='k')

    # 추세선 (대각선)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.title('실제 TCI vs 인공지능 예측 TCI', fontsize=16)
    plt.xlabel('실제 TCI (수학적 계산값)', fontsize=12)
    plt.ylabel('예측 TCI (LightGBM 모델)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('tci_scatter_plot.png', dpi=300)

    print("\n그래프 이미지(2장)가 폴더에 저장되었습니다!")
    plt.show()


if __name__ == "__main__":
    train_tci_model()