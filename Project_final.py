import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# 데이터 로드
train_data = pd.read_csv('./training.csv')  # 훈련 데이터 로드
test_data = pd.read_csv('./test.csv')  # 테스트 데이터 로드

# 훈련 데이터에서 특징(X)과 목표 변수(y) 분리
X = train_data.drop(columns=['Value'])  # 'Value' 열 제거 후 X에 저장 (독립 변수)
y = train_data['Value']  # 'Value' 열을 y에 저장 (종속 변수)

# 훈련 세트와 검증 세트로 분리 (80% 훈련, 20% 검증)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=90)

# 데이터 표준화
scaler = StandardScaler()  # StandardScaler 초기화
X_train_scaled = scaler.fit_transform(X_train)  # 훈련 데이터를 기준으로 표준화 수행
X_val_scaled = scaler.transform(X_val)  # 검증 데이터를 훈련 데이터 기준으로 표준화
X_test_scaled = scaler.transform(test_data) # 테스트 데이터를 훈련 데이터 기준으로 표준화

# 차원 축소 (PCA)
pca = PCA(n_components=0.95, random_state=90)  # PCA 초기화 (95% 이상의 분산 유지)
X_train_pca = pca.fit_transform(X_train_scaled)  # 훈련 데이터에 PCA 적용
X_val_pca = pca.transform(X_val_scaled)  # 검증 데이터에 PCA 적용
X_test_pca = pca.transform(X_test_scaled)  # 테스트 데이터에 PCA 적용

# Ridge Regression 모델 학습 및 하이퍼파라미터 튜닝
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}  # Ridge 하이퍼파라미터 그리드 정의
ridge = Ridge(random_state=90)  # Ridge 회귀 초기화
ridge_grid_search = GridSearchCV(ridge, ridge_params, scoring='neg_mean_squared_error', cv=5)  # Ridge 모델의 교차 검증
ridge_grid_search.fit(X_train_pca, y_train)  # 훈련 데이터를 사용해 하이퍼파라미터 최적화
best_ridge = ridge_grid_search.best_estimator_  # 최적의 Ridge 모델 선택

# Ridge 모델 검증 성능 평가
ridge_val_predictions = best_ridge.predict(X_val_pca)  # 검증 데이터로 예측 수행
ridge_val_rmse = np.sqrt(mean_squared_error(y_val, ridge_val_predictions))  # RMSE 계산
print("Ridge Validation RMSE:", ridge_val_rmse)  # RMSE 출력

# RandomForestRegressor 모델 학습 및 하이퍼파라미터 튜닝
rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10, 20],
             'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}  # Random Forest 하이퍼파라미터 그리드 정의
rf = RandomForestRegressor(random_state=90)  # Random Forest 초기화
rf_grid_search = GridSearchCV(rf, rf_params, scoring='neg_mean_squared_error', cv=5)  # Random Forest 모델 교차 검증
rf_grid_search.fit(X_train_pca, y_train)  # 훈련 데이터를 사용해 하이퍼파라미터 최적화
best_rf = rf_grid_search.best_estimator_  # 최적의 Random Forest 모델 선택

# RandomForest 모델 검증 성능 평가
rf_val_predictions = best_rf.predict(X_val_pca)  # 검증 데이터로 예측 수행
rf_val_rmse = np.sqrt(mean_squared_error(y_val, rf_val_predictions))  # RMSE 계산
print("RandomForest Validation RMSE:", rf_val_rmse)  # RMSE 출력

# Gradient Boosting 모델 학습 및 하이퍼파라미터 튜닝
gb_params = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
             'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}  # Gradient Boosting 하이퍼파라미터 그리드 정의
gb = GradientBoostingRegressor(random_state=90)  # Gradient Boosting 초기화
gb_grid_search = GridSearchCV(gb, gb_params, scoring='neg_mean_squared_error', cv=5)  # Gradient Boosting 모델 교차 검증
gb_grid_search.fit(X_train_pca, y_train)  # 훈련 데이터를 사용해 하이퍼파라미터 최적화
best_gb = gb_grid_search.best_estimator_  # 최적의 Gradient Boosting 모델 선택

# Gradient Boosting 모델 검증 성능 평가
gb_val_predictions = best_gb.predict(X_val_pca)  # 검증 데이터로 예측 수행
gb_val_rmse = np.sqrt(mean_squared_error(y_val, gb_val_predictions))  # RMSE 계산
print("Gradient Boosting Validation RMSE:", gb_val_rmse)  # RMSE 출력

# 테스트 데이터에 대한 예측값 생성 (앙상블 기법 사용)
test_predictions_ridge = best_ridge.predict(X_test_pca)  # Ridge 모델의 테스트 데이터 예측
test_predictions_rf = best_rf.predict(X_test_pca)  # Random Forest 모델의 테스트 데이터 예측
test_predictions_gb = best_gb.predict(X_test_pca)  # Gradient Boosting 모델의 테스트 데이터 예측

# 앙상블 예측 (세 모델의 예측값 평균)
test_predictions_final = (test_predictions_ridge + test_predictions_rf + test_predictions_gb) / 3

# 예측값을 포함한 최종 결과 저장
test_results = test_data.copy()  # 테스트 데이터 복사
test_results['Prediction'] = test_predictions_final  # 예측값 열 추가

# csv 파일로 저장
test_results.to_csv('./유승인_test.csv', index=False)  # 최종 결과를 CSV 파일로 저장
print("예측 결과를 유승인_test.csv 파일에 저장.")  # 저장 완료 메시지 출력