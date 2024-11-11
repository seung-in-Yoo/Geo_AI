# 2번째 임시코드 
# 이 코드는 Ridge Regression, RandomForestRegressor 모델 예측값에 Gradient Boosting 모델을 추가하여 각각의 예측값을 구하고,
# 3개의 예측값의 평균을 내서 (앙상블 기법) 더 높은 정확도를 기대할수는 있지만 현재 test data의 value값이 없기때문에 
# 두개의 코드는 어떤게 확실하게 더 정확도가 높다고 장담할수 없음
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np


train_data = pd.read_csv('./training.csv')  
test_data = pd.read_csv('./test.csv')       

# 훈련 데이터에서 특징(X)과 목표 변수(y) 분리
X_train = train_data.drop(columns=['Value'])
y_train = train_data['Value']

# 데이터 표준화
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

# RandomForestRegressor 모델 학습 및 하이퍼파라미터 튜닝
rf_params = {'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 5, 10, 20],  
             'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}  # 하이퍼파라미터 추가
rf = RandomForestRegressor(random_state=99)
rf_grid_search = GridSearchCV(rf, rf_params, scoring='neg_mean_squared_error', cv=5)
rf_grid_search.fit(X_train_pca, y_train)
best_rf = rf_grid_search.best_estimator_

# Gradient Boosting 모델 학습 및 하이퍼파라미터 튜닝   => 정확도를 좀 더 향상시키기 위해 위 코드에 해당 코드까지 추가 (트리 기반 모델)
gb_params = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
             'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
gb = GradientBoostingRegressor(random_state=99)
gb_grid_search = GridSearchCV(gb, gb_params, scoring='neg_mean_squared_error', cv=5)
gb_grid_search.fit(X_train_pca, y_train)
best_gb = gb_grid_search.best_estimator_

# 각 모델의 최적 성능 평가
ridge_best_score = np.sqrt(-ridge_grid_search.best_score_)
rf_best_score = np.sqrt(-rf_grid_search.best_score_)
gb_best_score = np.sqrt(-gb_grid_search.best_score_)

print("Ridge Regression RMSE:", ridge_best_score)
print("Random Forest RMSE:", rf_best_score)
print("Gradient Boosting RMSE:", gb_best_score)

# 테스트 데이터에 대한 예측값 생성 (앙상블 기법 사용)
test_predictions_ridge = best_ridge.predict(X_test_pca) # Ridge Regression 
test_predictions_rf = best_rf.predict(X_test_pca) # RandomForestRegressor
test_predictions_gb = best_gb.predict(X_test_pca) # Gradient Boosting

# 앙상블 예측 (세 모델의 예측값 평균)
test_predictions2 = (test_predictions_ridge + test_predictions_rf + test_predictions_gb) / 3

# 예측값을 포함한 최종 결과 생성
test_results = test_data.copy()
test_results['Predicted_Value'] = test_predictions2

# csv 파일로 저장
test_results.to_csv('./test_predictions2.csv', index=False)  
print("예측 완료: test_predictions2.csv 파일에 저장.")