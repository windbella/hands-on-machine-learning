### 앙상블 학습과 랜덤 포레스트
- 앙상블 학습 : 여러 개의 모델을 훈련하고 그 예측들을 결합하여 최종 예측을 도출하는 방식
- 각 분류기가 약한 학습기라도 약한 학습기가 충분히 많고 다양하다면 강한 학습기가 될 수 있음
- 큰 수의 법칙 : 앞면이 51%, 뒷면이 49%가 나오는 동전이 있다고 하면 1000번을 던진 후 앞면이 다수가 될 확률은 75%에 가까움

![7-2](https://github.com/windbella/hands-on-machine-learning/blob/main/ch7/7-3.png)
- 51% 정확도를 가진 1000개의 분류기로 앙상블 모델을 구축한다면 75%의 정확도를 기대할 수 있음
- 단, 모든 분류기가 완벽하게 독립적이고 오차에 상관관계가 없어야 함
- 다양한 분류기를 얻는 대표적인 한 가지 방법으로 각기 다른 알고리즘으로 학습시키는 방법이 있음

## 7.1 투표 기반 분류기
![7-2](https://github.com/windbella/hands-on-machine-learning/blob/main/ch7/7-2.png)
- 직접 투표 : 다수결 투표
- 간접 투표 : 모든 분류기가 클래스의 확률을 예측할 수 있으면 개별 분류기의 예측을 평균 내어 확률이 가장 높은 클래스를 예측
- 간접 투표가 직접 투표보다 성능이 높음

## 7.2 배깅과 페이스팅
![7-4](https://github.com/windbella/hands-on-machine-learning/blob/main/ch7/7-2.png)
- 다양한 분류기를 얻는 다른 방법으로 훈련 세트의 서브셋을 랜덤으로 구성하여 분류기를 각기 다르게 학습 시키는 것
- 배깅(bagging) : 중복을 허용하여 샘플링 하는 방식 (bootstrap aggregating)
- 페이스팅(pasting) : 중복을 허용하지 않고 샘플링 하는 방식
- 모든 예측기가 훈련을 마치면 앙상블은 분류일 때는 통계적 최빈값(가장 많은 예측 결과 like 직접 투표 분류기)을, 회귀에 대해서는 평균을 계산해 결과를 냄
- 앙상블 결과는 원본 데이터셋으로 하나의 예측기를 훈련시킬 때와 비교해 편향은 비슷하지만 분산이 줄어듦
- 편향 : 베깅 < 페이스팅 (페이스팅이 더 우수)
- 분산 : 베깅 > 페이스팅 (베깅이 더 우수)
- 전반적으로 베깅이 더 나은 모델을 만듬
- OOB 평가 : 베깅에서 out-of-bag로 훈련에 사용되지 않은 나머지 샘플을 각 예측기가 평가해 그 평균으로 얻은 평가 점수

![pv](https://github.com/windbella/hands-on-machine-learning/blob/main/ch7/pv.png)   
[[이미지 출처]](https://datacookbook.kr/48)
![7-5](https://github.com/windbella/hands-on-machine-learning/blob/main/ch7/7-5.png)
- bias : 편향, variance : 분산
- 학습이나 예측을 병렬로 수행할 수 있어서 인기가 높음

## 7.3 랜덤 패치와 랜덤 서브스페이스
- 샘플 뿐 아니라 특성 샘플링도 지원
- 랜덤 패치 방식 : 특성과 샘플을 모두 샘플링
- 랜덤 서브스페이스 방식 : 특성만 샘플링
## 7.4 랜덤 포레스트
- 베깅 방법(또는 페이스팅)을 적용한 결정 트리 앙상블
- 엑스트라 트리 : 트리를 더욱 랜덤하게 만들기 위해 최적의 임곗값을 찾는 대신 후보 특성을 사용해 랜덤으로 분할한 다음 그중 최상의 분할을 선택하는 방식, 편향이 늘어나는 대신 분산이 낮아짐
- 어떤 특성을 사용한 노드가 평균적으로 불순도를 얼마나 감소시키는지 확인하여 특성의 중요도를 측정 가능

![7-6](https://github.com/windbella/hands-on-machine-learning/blob/main/ch7/7-6.png)
## 7.5 부스팅
- 약한 학습기 여러 개를 연결하여 강한 학습기를 만드는 앙상블 방법
- 앞의 모델을 보완해 나가면서 일련의 예측기를 학습 시키는 아이디어

![7-7](https://github.com/windbella/hands-on-machine-learning/blob/main/ch7/7-7.png)
![adb](https://github.com/windbella/hands-on-machine-learning/blob/main/ch7/adaboost.png)   
[[이미지 출처]](https://bommbom.tistory.com/entry/Boosting-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-Adaboost-%EB%8F%99%EC%9E%91-%EC%9B%90%EB%A6%AC)
- AdaBoost : 이전 예측기를 보완하는 새로운 예측기를 만드는 방법은 이전 모델이 과소적합했던 훈련 샘플의 가중치를 더 높이는 것
- 첫 번째 약한 학습기가 첫 번째 분류 기준(D1)으로 +와 -를 분류
- 잘못 분류된 데이터에 대해 가중치를 부여(두 번째 그림에서 커진 + 표시)
- 두 번째 약한 학습기가 두 번째 분류 기준(D2)으로 +와 -를 다시 분류
- 잘못 분류된 데이터에 대해 가중치를 부여(세 번째 그림에서 커진 - 표시)
- 세 번째 약한 학습기가 세 번째 분류 기준(D3)으로 +와 -를 다시 분류해서 오류 데이터를 찾음
- 마지막으로 분류기들을 결합하여 최종 예측 수행(네 번째 그림)

![7-9](https://github.com/windbella/hands-on-machine-learning/blob/main/ch7/7-9.png)
![gb](https://github.com/windbella/hands-on-machine-learning/blob/main/ch7/gb.png)   
[[이미지 출처]](https://beavekim23.tistory.com/3)
- 그레이디언트 부스팅 : 샘플의 가중치를 수정하는 대신 이전 예측기가 만든 잔여 오차에 새로운 예측기를 학습 시킴, 각 트리의 예측값을 합쳐서 최종 예측 수행
- 히스토그램 기반 그레이디언트 부스팅 : 특성의 구간을 분할하여 평가해야 하는 임곗값의 수를 크게 줄여 성능을 향상시킴, 규제처럼 작동해서 정밀도 손실을 유발하므로 데이터 셋에 따라서 과대 적합을 줄이는 데 도움이 될 수도 있고 과소 적합을 유발할 수도 있음
## 7.6 스태킹
- 앙상블에 속한 모든 예측기의 예측을 취합하는 간단한 함수 대신 취합하는 모델을 훈련 시키는 아이디어

![7-12](https://github.com/windbella/hands-on-machine-learning/blob/main/ch7/7-12.png)
- 성능을 조금 더 끌어올릴 수 있지만 훈련 시간과 시스템 복잡성 측면에서 비용이 증가
