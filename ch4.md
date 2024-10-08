## 모델 훈련
### 4.1 선형 회귀
- y = θx
- 제곱근 오차(RMSE)를 최소화하는 θ를 찾아야 함
- 정규 방정식 : θ 값을 찾기 위한 해석적 방법
- X^T : 전치행렬, 행렬 내의 원소를 대각선축(주대각성분)을 기준으로 서로 위치를 바꾼 것을 말한다.
- LinearRegression은 유사역행렬을 이용한 함수를 이용해 θ 값을 찾는데 극단적인 경우도 처리할 수 있어 정규 방정식 보다 유리함
### 4.2 경사 하강법
- 비용 함수를 최소화하기 위해서 반복해서 파라미터를 조정해가는 방식
- 중요한 파라미터는 스텝의 크기, 학습률 하이퍼파라미터로 결정됨, 학습률이 너무 적으면 시간이 오래 걸리고 너무 크면 골짜기 반대편으로 건너 뛰어 해법을 찾지 못하게 됨
- 경사 하강법은 지역 최솟값이 있는 경우 전역 최솟값을 찾지 못하는 문제가 있음
- 선형  회귀를 위한 MSE 비용 함수는 볼록 함수여서 지역 최솟값이 없음
- 배치 경사 하강법 : 매 스탭에서 훈련 데이터 전체를 사용함, 속도가 느림
- 확률 경사 하강법 : 매 스탭에서 한 개의 샘플을 랜덤으로 선택하고 그레이디언트를 계산함, 지역 최솟값을 건너뛸 수 있음, 속도가 빠름, 최솟값에 안착하기 어려움
- 미니배치 경사 하강법 : 미니배치라 부르는 작은 샘플 세트에 대해서 그레이디언트를 계산, 행렬 연산에 최적화된 하드웨어(GPU)를 사용해 성능을 향상시킬 수 있음
### 4.3 다항 회귀
- 각 특성의 거듭제곱을 새로운 특성으로 추가하여 N차 방정식 형태의 선으로 회귀 가능
### 4.4 학습 곡선
- 고차 다항 회귀를 적용하면 일반 선형 회귀에서 비해서 훈련 데이터에 더 잘 맞추려 하여 과대적 합 될 가능성이 높음
- 학습 곡선 : 훈련 오차와 검증 오차를 훈련 반복 횟수의 함수로 나타낸 그래프
- 훈련 오차는 훈련 세트의 크기가 작을 때 매우 적다가 어느 정도 평편해질 때까지 오차가 상승함, 검증 오차는 세트의 크기가 작을 때 매우 크다가 평평해질 때까지 점점 오차가 하락함
- 과소적합에서는 훈련 오차와 검증 오차가 높은 오차에서 매우 가깝게 만남
- 과대적합에서는 훈련 오차와 검증 오차가 낮은 오차에서 만나지만 두 그래프 사이에 공간이 큼
### 4.5 규제가 있는 선형 모델
- 과대적합을 줄이는 좋은 방법은 모델을 규제하는 것
- 릿지 회귀 : 규제가 추가된 선형 회귀 버전, 모델의 가중치가 가능한 작게 유지되도록 함
- 라쏘 회귀 : 선형 회귀의 또 다른 규제된 버전, 덜 중요한 특성의 가중치를 제거하려는 경향이 있음, 라쏘 모델은 자동으로 특성을 선택을 수행하고 희소 모델을 만듬
- 엘라스틱넷 : 릿지 회귀와 라쏘 회귀를 절충한 모델
- 규제가 약간 있는 것이 대부분의 경우 좋음, 릿지가 기본이 되지만 몇 가지 특성만 유용하다고 생각되면 라쏘나 엘리스틱이 좋음
- 조기 종료 : 검증 오차가 최솟값에 도달하면 바로 훈련을 중지시키는 방식의 규제
### 4.6 로지스틱 회귀
- 일부 회귀 알고리즘은 분류에서도 사용 가능
- 로지스틱 회귀는 샘플이 특정 클래스에 속할 확률을 추정하는데 널리 사용됨
- 선형 회귀 모델과 같이 특성의 가중치의 합을 계산하지만 바로 결과를 출력하지 않고 로지스틱을 출력함
- 로지스틱은 0과 1사이의 값을 출력하는 시모이드 함수
- 로지스틱 회귀 모델은 여러 개의 이진 분류기를 훈련시켜 연결하지 않고 직접 다중 클래스를 지원하도록 일반화될 수 있음. 이를 소프트맥스 회귀 혹은 다항 로지스틱 회귀라고 함
