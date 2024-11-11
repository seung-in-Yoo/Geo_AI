# 1번째 임시 코드 
# Ridge Regression과 RandomForestRegressor 모델중 후자를 선택 (이유는 데이터의 비선형성 가능성과 RandomForest의 일반화 능력을 고려)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np


train_data = pd.read_csv('./training.csv') 
test_data = pd.read_csv('./test.csv')      

# 훈련 데이터에서 특징(X)과 목표 변수(y) 분리
X_train = train_data.drop(columns=['Value'])
y_train = train_data['Value']

# 데이터 표준화 시키기
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data)

# 차원 축소 (PCA)
pca = PCA(n_components=0.95, random_state=99)  # 95% 이상의 분산 유지
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Ridge Regression 모델 학습 및 하이퍼파라미터 튜닝
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge = Ridge(random_state=99)
ridge_grid_search = GridSearchCV(ridge, ridge_params, scoring='neg_mean_squared_error', cv=5)
ridge_grid_search.fit(X_train_pca, y_train)
best_ridge = ridge_grid_search.best_estimator_
ridge_best_score = np.sqrt(-ridge_grid_search.best_score_)   # Ridge Regression의 교차 검증 RMSE (약 4.46)

# RandomForestRegressor 모델 학습 및 하이퍼파라미터 튜닝
rf_params = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 5, 10, 20]}
rf = RandomForestRegressor(random_state=99)
rf_grid_search = GridSearchCV(rf, rf_params, scoring='neg_mean_squared_error', cv=5)
rf_grid_search.fit(X_train_pca, y_train)
best_rf = rf_grid_search.best_estimator_
rf_best_score = np.sqrt(-rf_grid_search.best_score_)   # RandomForestRegressor의 교차 검증 RMSE (약 4.55)

# 테스트 데이터에 대한 예측값 생성 (RandomForest 사용)
test_predictions = best_rf.predict(X_test_pca)

# 예측값을 포함한 최종 결과 생성
test_results = test_data.copy()
test_results['Predicted_Value'] = test_predictions

# 결과를 csv파일로 저장
test_results.to_csv('./test_predictions.csv', index=False)  # 결과를 CSV 파일로 저장
print("예측 완료: test_predictions.csv 파일에 저장.")