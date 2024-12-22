import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA  
from sklearn.linear_model import Ridge  
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.metrics import mean_squared_error, mean_absolute_error  

# 데이터 로드
train_data = pd.read_csv('./training.csv')  
test_ori_data = pd.read_csv('./test_ori.csv')  
user_predicted_data = pd.read_csv('./유승인_test.csv')  # 예측한 결과 파일 불러오기

# 독립 변수(X)와 종속 변수(y) 분리
X = train_data.drop(columns=['Value'])  
y = train_data['Value'] 

X_test = test_ori_data.drop(columns=['Value'])  
y_test = test_ori_data['Value']  # 테스트 데이터의 종속 변수

# 훈련 및 검증 데이터 분리 (훈련:검증 = 8:2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=90)

# 변수 간 상관관계 분석 및 시각화
plt.figure(figsize=(12, 10))  # 그래프 크기 설정
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')  # 상관관계 행렬 시각화하기
plt.title('Feature Correlation Matrix')  # 제목 설정
plt.show()  # 그래프 시각화


# 데이터 표준화
scaler = StandardScaler()  # 표준화 객체 생성
X_train_scaled = scaler.fit_transform(X_train)  # training data 표준화
X_val_scaled = scaler.transform(X_val)  # validation data 표준화
X_test_scaled = scaler.transform(X_test)  # test data 표준화

# PCA 적용 전후 분산 분석
pca = PCA(n_components=0.95, random_state=90)  # 분산이 95%가 되도록 PCA 설정
X_train_pca = pca.fit_transform(X_train_scaled)  # 훈련 데이터를 사용하여 PCA 모델 학습
X_val_pca = pca.transform(X_val_scaled)   # 검증 데이터를 학습된 PCA 모델로 변환
X_test_pca = pca.transform(X_test_scaled)  # 테스트 데이터를 학습된 PCA 모델로 변환
explained_variance = pca.explained_variance_ratio_  # 각 주성분이 데이터의 전체 분산에서 차지하는 비율 계산

# 누적 설명 분산 비율 그래프
plt.figure()
plt.plot(np.cumsum(explained_variance), marker='o')  # 분산의 누적 값 시각화
plt.title('Cumulative Explained Variance by PCA Components')  # 제목 설정
plt.xlabel('Number of Components')       # x축 레이블 설정 
plt.ylabel('Explained Variance Ratio')   # y축 레이블 설정 
plt.show()  # 그래프 시각화

# 성능 지표 함수 정의
def calculate_metrics(y_true, y_pred):
    """
    실제 값(y_true)과 예측 값(y_pred) 간의 평가 지표를 계산하는 함수.
    """
    me = np.mean(y_pred - y_true)  # 평균 오차
    mae = mean_absolute_error(y_true, y_pred)  # 평균 절대 오차
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # RMSE
    cc = np.corrcoef(y_true, y_pred)[0, 1]  # 상관계수
    return me, mae, rmse, cc

# Ridge 회귀 (alpha 튜닝)
ridge = Ridge(random_state=90)  # Ridge 모델 생성
ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}  # 하이퍼파라미터 알파 설정
ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='neg_mean_squared_error')  # GridSearchCV 사용
ridge_grid.fit(X_train_pca, y_train)   # 교차 검증을 통해 최적의 하이퍼파라미터 탐색
ridge_best = ridge_grid.best_estimator_  # 최적의 하이퍼파라미터로 학습된 Ridge 회귀 모델 반환
ridge_preds = ridge_best.predict(X_test_pca)  # test data를 사용하여 최적 Ridge 모델이 예측한 종속 변수 값 반환
ridge_me, ridge_mae, ridge_rmse, ridge_cc = calculate_metrics(y_test, ridge_preds)  # 평균 오차, 평균 절대 오차, RMSE, 상관계수 계산

# Ridge 회귀 계수 시각화
ridge_coefs = ridge_best.coef_  # 최적의 Ridge 모델 회귀 계수 
plt.figure(figsize=(10, 6))
plt.bar(range(len(ridge_coefs)), ridge_coefs)  # 회귀 계수 시각화 진행
plt.title('Ridge Regression Coefficients')  
plt.xlabel('PCA Components')  
plt.ylabel('Coefficient Value')  
plt.show()

# RandomForest 회귀 (하이퍼파라미터 튜닝)
rf = RandomForestRegressor(random_state=90)  # RandomForest 모델 생성
rf_params = {
    'n_estimators': [100, 200, 300],  # 트리 개수
    'max_depth': [None, 10, 20, 30],  # 트리 최대 깊이
    'min_samples_split': [2, 5, 10]  # 최소 샘플 분할 기준
}
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='neg_mean_squared_error')  
rf_grid.fit(X_train_pca, y_train)  # 최적의 하이퍼파라미터 탐색
rf_best = rf_grid.best_estimator_  # 가장 성능이 좋았던 하이퍼파라미터로 학습된 RandomForestRegressor 모델을 반환
rf_preds = rf_best.predict(X_test_pca)  # 최적 모델을 사용하여 test data의 종속 변수 예측
rf_me, rf_mae, rf_rmse, rf_cc = calculate_metrics(y_test, rf_preds)  # 평균 오차, 평균 절대 오차, RMSE, 상관계수 계산
 
# RandomForest 중요도 시각화
plt.figure(figsize=(10, 6))
plt.bar(range(len(rf_best.feature_importances_)), rf_best.feature_importances_)
plt.title('RandomForest Feature Importances')  # 제목 설정
plt.xlabel('Features')    # x축 레이블 설정
plt.ylabel('Importance')  # y축 레이블 설정
plt.show()   # 그래프 시각화

# Gradient Boosting 회귀 (하이퍼파라미터 튜닝)
gb = GradientBoostingRegressor(random_state=90)
gb_params = {
    'n_estimators': [100, 200, 300],    # 트리 개수
    'learning_rate': [0.01, 0.1, 0.2],  # 학습률
    'max_depth': [3, 5, 7]   # 트리 최대 깊이
}
gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='neg_mean_squared_error') # GridSearchCV 사용
gb_grid.fit(X_train_pca, y_train)  # 최적의 하이퍼파라미터 탐색
gb_best = gb_grid.best_estimator_  # 가장 성능이 좋았던 하이퍼파라미터로 학습된 Gradient Boosting 모델을 반환
gb_preds = gb_best.predict(X_test_pca) # 최적 모델을 사용하여 test data의 종속 변수 예측
gb_me, gb_mae, gb_rmse, gb_cc = calculate_metrics(y_test, gb_preds) # 평균 오차, 평균 절대 오차, RMSE, 상관계수 계산

plt.figure(figsize=(10, 6))
plt.bar(range(len(gb_best.feature_importances_)), gb_best.feature_importances_)
plt.title('Gradient Boosting Feature Importances')  # 제목 설정
plt.xlabel('Features')    # x축 레이블 설정
plt.ylabel('Importance')  # y축 레이블 설정
plt.show()   # 그래프 시각화


# 변수 중요도 시각화
rf_importances = rf_best.feature_importances_   # Random Forest 모델에서 각 특징의 중요도를 나타내는 배열
gb_importances = gb_best.feature_importances_   # Gradient Boosting 모델에서 각 특징의 중요도를 나타내는 배열

# PCA의 각 주성분에 대한 중요도
rf_pca_importances = np.dot(pca.components_.T, rf_importances)  # PCA와 모델의 feature_importances_ 결합
gb_pca_importances = np.dot(pca.components_.T, gb_importances)  # PCA와 모델의 feature_importances_ 결합

plt.figure(figsize=(12, 5))

# RandomForest Feature Importance 시각화
plt.subplot(1, 2, 1)
plt.bar(range(len(rf_pca_importances)), rf_pca_importances)
plt.title('RandomForest Feature Importance after PCA')   # 제목 설정
plt.xlabel('PCA Components')     # x축 레이블 설정 
plt.ylabel('Importance Score')   # y축 레이블 설정

# Gradient Boosting Feature Importance 시각화
plt.subplot(1, 2, 2)
plt.bar(range(len(gb_pca_importances)), gb_pca_importances)
plt.title('Gradient Boosting Feature Importance after PCA')  # 제목 설정
plt.xlabel('PCA Components')     # x축 레이블 설정 
plt.ylabel('Importance Score')    # y축 레이블 설정

plt.tight_layout()  # 그래프 간격 조정
plt.show()   # 그래프 시각화 

# 성능 비교 시각화
metrics_df = pd.DataFrame({
    'Model': ['Ridge', 'RandomForest', 'GradientBoosting'],
    'ME': [ridge_me, rf_me, gb_me],
    'MAE': [ridge_mae, rf_mae, gb_mae],
    'RMSE': [ridge_rmse, rf_rmse, gb_rmse],
    'CC': [ridge_cc, rf_cc, gb_cc]
})

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
metrics_df.set_index('Model')[['MAE', 'RMSE']].plot(kind='bar', ax=plt.gca()) # MAE와 'RMSE' 값을 비교하기 위한 막대 그래프를 생성
plt.title('MAE and RMSE Comparison')  # 제목 설정
plt.ylabel('Error')   # y축 레이블 설정

plt.subplot(1, 2, 2)
metrics_df.set_index('Model')['CC'].plot(kind='bar', color='green', ax=plt.gca())  # CC 값을 비교하기 위한 막대 그래프를 생성
plt.title('Correlation Coefficient (CC) Comparison')  # 제목 설정
plt.ylabel('CC')   # y축 레이블 설정

plt.tight_layout()  # 그래프 간격 조정
plt.show()   # 그래프 시각화


# 가중치 기반 앙상블 예측
total_rmse = ridge_rmse + rf_rmse + gb_rmse  # 전체 RMSE 합계
ridge_weight = (1 / ridge_rmse) / (1 / ridge_rmse + 1 / rf_rmse + 1 / gb_rmse)  # Ridge 가중치
rf_weight = (1 / rf_rmse) / (1 / ridge_rmse + 1 / rf_rmse + 1 / gb_rmse)  # RandomForest 가중치
gb_weight = (1 / gb_rmse) / (1 / ridge_rmse + 1 / rf_rmse + 1 / gb_rmse)  # Gradient Boosting 가중치

# 앙상블 예측 수행
final_preds = (
    ridge_weight * ridge_preds +
    rf_weight * rf_preds +
    gb_weight * gb_preds
)

# 앙상블 모델 성능 평가 진행
ensemble_me, ensemble_mae, ensemble_rmse, ensemble_cc = calculate_metrics(y_test, final_preds)  # 평균 오차, 평균 절대 오차, RMSE, 상관계수 계산
print("\nEnsemble Model Performance:")
print(f"ME: {ensemble_me:.4f}, MAE: {ensemble_mae:.4f}, RMSE: {ensemble_rmse:.4f}, CC: {ensemble_cc:.4f}")

# 예측값과 실제값 비교
if 'Prediction' in user_predicted_data.columns:
    user_predicted_data['Actual'] = y_test.values  # 실제값 추가
    user_predicted_data['Difference'] = user_predicted_data['Prediction'] - y_test.values  # 예측 오차 계산

    # 예측 오차 히스토그램 시각화
    plt.figure(figsize=(12, 6))
    plt.hist(user_predicted_data['Difference'], bins=30, color='purple', alpha=0.7)
    plt.title('Distribution of Prediction Errors (User vs Actual)')  # 제목 설정
    plt.xlabel('Prediction Error')   # x축 레이블 설정 
    plt.ylabel('Frequency')   # y축 레이블 설정
    plt.show()   # 그래프 시각화

    # 예측값과 실제값 비교 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(user_predicted_data['Prediction'], label='User Predicted', color='orange')
    plt.plot(y_test.values, label='Actual', color='blue')
    plt.legend()
    plt.title('User Predicted vs Actual Values')   # 제목 설정
    plt.xlabel('Index')   # x축 레이블 설정 
    plt.ylabel('Value')   # y축 레이블 설정
    plt.show()   # 그래프 시각화

    # 최종 결과 저장 후 csv 파일로 저장
    user_predicted_data.to_csv('./retest_result.csv', index=False)
    print("retest_result.csv 파일로 저장")
else:
    print("'Prediction' 값이 존재하지 않습니다. 파일을 다시 확인해주세요.")