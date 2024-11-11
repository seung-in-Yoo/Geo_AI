# 공간 인공지능 프로젝트
프로젝트 주제:
Prediction Modeling Competition
- 제공하는 Training Data로 기계학습 기반 회귀 모델을 학습시킨 후에, 별도로 제공된 Test Data를 대상으로 예측

프로젝트 내용:
입력 특징: 19개 - 실제 특징 이름 미공개
타겟 변수: 1개 - 실제 타겟 변수 이름 미공개
training.csv - 입력 특징 19개와 타겟 변수 포함
test.csv - 입력 특징만 포함

자료 처리:
training.csv를 이용한 감독학습을 통해 test.csv 각 샘플별 타겟 변수값을 예측
예측 방법
OLS 다중 회귀와 geographically weighted regression을 제외하고, 수업시간에 다루었던 기계학습 모델들(ridge, lasso, DT, RF, NN 등)을 포함하여 적용
또한 자료 전처리가 필요할 경우 수업시간에 배운 내용을 토대로 적용
개별 모델마다 결정해야 하는 hyperparameter는 Grid Search 기능 혹은 별도 프로그래밍을 통해 결정 (임의의 특정 값만을 지정하면 X)
위 과정에서 cross validation/training data splitting 절차를 반드시 적용
