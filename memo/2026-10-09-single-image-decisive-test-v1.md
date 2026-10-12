# 단일 이미지 PROBE: 결정적 검사 규약 v1

작성: 2026-10-09 CDT. 작성자: `single-proxy-balancing`. 승인: 용한 ("진행해", 2026-10-09). 상위 문서: `2026-10-08-single-image-probe-plan-v2.md`(이하 계획 v2). 이 규약은 계획 v2 §12의 1번과 2번을 실행 가능한 형태로 고정한 것이다. 실행 전에 고정하고, 실행 뒤에는 바꾸지 않는다.

실행 주체는 코드를 맡은 세션이다. 이 세션은 결과를 닫힌 식과 대조해 검토한다. 두 세션이 같은 파일을 동시에 고치지 않는다.

## 0. 이 검사가 결정하는 것

질문 하나다. **이미지가 중증도 정보를 충분히 담은 조건에서, 한 장 이미지 PROBE가 oracle 가까이 가고, 장치 영역을 걸러내며, 인증 통과를 내는가.** 이 질문에 "예"가 나오면 비선형 변형과 DeltaAI 격자로 간다. "아니오"가 나오면 그 원인(구현인지 방법인지)을 먼저 가른다.

지금까지의 결과($\sigma=0.85$에서 추정값 1.44, oracle 1.00)는 이 질문에 답하지 못한다. 그 조건에서는 작업 모형이 예측하는 posterior-mean 표현의 편향이 0.37이라 개선이 보일 수 없기 때문이다. 이 값은 작업 모형의 계산값이지 보편 하한이 아니다.

## 1. 생성기 변경 (phantom)

계획 v2 §10.1을 다음 세 가지만 바꾼다.

1. **bounded link.** 처치확률을 논문의 SCM과 같이
   $$P(A=1\mid U,I)=0.1+0.8\,\Phi(1.5U+1.0I)$$
   로 둔다. 구현은 `PhantomConfig`에 `link: str = "probit"` 필드를 더하고 `"bounded"`일 때 위 식을 쓰는 것이다. 기본값은 바꾸지 않아 기존 결과가 재현된다. 이 link에서는 참 처치확률이 항상 $[0.1,0.9]$ 안이므로 overlap 검사가 생성기와 충돌하지 않는다.
2. **image-only 모드 고정.** `image_only=True`.
3. **$\sigma$ 격자.** $\sigma\in\{0.3,\ 0.5,\ 0.85\}$. 0.3은 이미지가 $U$를 거의 결정하는 조건, 0.85는 지금까지의 조건, 0.5는 그 사이다.

나머지(영역 마스크, 병변 렌더링, 전역 밝기 0.3G, 영역 5의 장치 $I\sim\mathrm{Bern}(0.3)$, 결과 $Y=A+U+\epsilon_Y$, $\tau=1$, 몸통 기준 밝기 보정)는 그대로다.

## 2. 표본과 역할

- $n=48{,}000$. 역할은 R 12,000, D·C·P·S·E 각 7,200. 같은 환자의 조각은 같은 역할에 둔다.
- seed는 $\{20261009,\ 20261010,\ 20261011\}$ 세 개. $\sigma$마다 세 번 반복한다.
- 역할 회전은 하지 않는다. 이 검사는 결정이 목적이고 분산 추정은 세 seed로 한다.

## 3. 방법

분할은 singleton 다섯 개 $S_1,\dots,S_5$다. 각 분할에서 두 encoder를 돌린다.

- **고정 reader encoder.** 계획 v2 §8.1. reader 출력 위의 rank 2 선형 $B$.
- **neural encoder.** 계획 v2 §7. `train_alternating(rounds=30, encoder_steps=20)`. MAE 초기값. 밝기 보정 뒤 MAE의 hidden-pixel MSE가 평균 영상보다 나쁘면(현재 0.203 대 0.184) MAE 없이 무작위 초기값으로도 한 번 더 돌리고 둘 다 보고한다.

critic은 P에서 새로 적합하고 S에서 채점한다. 판정 $\widehat q_1$은 사다리 3단계(원본 픽셀 CNN + high-pass + reader residual)다. epoch 선택은 P 안의 validation으로만 한다.

## 4. 인증 상한의 수정

계획 v2 §3.2의 통과식 $\max(\widehat g,0)+r+b_1\le\delta^2$에서 두 항을 다음과 같이 바꾼다.

- **$b_1$의 목표를 $p(T,Z)$로.** 현재 코드는 $\widehat q_1$을 잠재 처치확률 $P(A=1\mid U,I)$와 비교한다. 그 차이에는 이미지로 해소할 수 없는 $\mathrm{Var}\{P(A\mid U,I)\mid T,Z\}$(약 0.035)가 들어 있어 어떤 predictor도 통과하지 못한다. 수정은 다음과 같다. encoder와 $T$의 정의를 고정한 채 생성기에서 $10^6$개를 새로 뽑고, 그 표본으로 $(T,Z)\mapsto A$의 near-oracle predictor $\tilde p$를 적합한다(사다리 3단계와 같은 구조, 충분한 epoch). 그러면 $b_1=\mathbb E_S[(\widehat q_1-\tilde p)^2]$의 상한을 쓴다. $\tilde p$ 자체의 오차는 $10^6$ 표본의 validation Brier와 $\tilde p$를 다시 적합한 두 번째 복제의 차이로 보고한다. 이 $b_1$은 phantom 전용이고, 실제 자료에서는 쓸 수 없다는 점을 결과에 적는다.
- **반경을 empirical Bernstein으로.** Brier 차이 $\ell_i=(A_i-\widehat q_0)^2-(A_i-\widehat q_1)^2$의 표본 분산 $\widehat V$로
  $$r=\sqrt{\frac{2\widehat V\log(3M/\alpha)}{n_S}}+\frac{3\log(3M/\alpha)}{n_S},\qquad M=12,\ \alpha=0.05$$
  를 쓴다 `[Axiom, 미확인: empirical Bernstein 부등식의 상수]`. Hoeffding 반경(현재 0.059)은 비교용으로 함께 적는다.

$\delta^2$ 격자는 $\{0.005,0.01,0.02\}$, 대표값 규칙은 계획 v2 §4다.

## 5. overlap 검사의 수정

범위 밖 비율 1%의 절대 기준을 **쓰지 않는다**. 상수 $Z$가 그 기준을 통과하므로 기준이 정보 손실을 보상하기 때문이다. 대신 논문처럼 성향을 $[0.05,0.95]$로 자르고, 잘린 비율을 세 가지로 보고한다. 학습한 $\widehat q_0$, known-$U$ 기준 $P(A=1\mid U,I)$, 그리고 bounded link에서의 모집단 값(0%). bounded link 아래에서 $\widehat q_0$의 잘린 비율이 2%를 넘으면 $Z$가 처치확률을 과장하고 있다는 뜻이므로 표시한다.

## 6. 효과 추정과 집계

- 통과한 분할마다 P에서 nuisance를 맞추고 E에서 AIPW를 계산한다.
- plurality 집계는 계획 v2 §11. $\rho$는 §3.6으로 예측한 분할 간 편향 차이의 최댓값에 표준오차를 더한 값.
- **방법 간 차이의 표준오차.** 같은 E 환자에서 두 방법의 influence function 차이 $\mathtt{UIF}^{(1)}_i-\mathtt{UIF}^{(2)}_i$의 표본 표준편차를 $\sqrt{n_E}$로 나눈다. 다음 쌍에 대해 계산한다. neural 대 고정 reader, neural 대 전체 이미지 reader 보정, 각각 대 oracle.
- 비교군은 계획 v2 §10.4의 전부. 특히 전체 이미지 reader 보정(균형 없음)과 oracle $U$.

## 7. 사전 지정한 판정 기준

$\sigma=0.3$에서 세 seed 평균으로 판단한다.

| 번호 | 기준 | 통과 조건 |
|---|---|---|
| K1 | 거짓 통과 | $S_5$가 $\delta^2\le0.01$에서 세 seed 모두 탈락 |
| K2 | 유효 분할의 인증 | $S_1$부터 $S_4$ 중 셋 이상이 $\delta^2=0.01$에서 **인증** 통과(§4의 $b_1$, $r$ 기준) |
| K3 | oracle과의 거리 | plurality 추정값과 oracle의 차이가 0.10 이하 |
| K4 | neural 대 고정 reader | neural이 고정 reader보다 나쁘지 않음: 차이가 $-2\,\mathrm{SE}$ 아래로 내려가지 않음 |
| K5 | 닫힌 식 대조 | 고정 reader 분할의 $\widehat g$와 편향이 계획 v2 §3.6의 예측과 30% 안에서 일치 |

판정 규칙:
- K1부터 K5 모두 통과: 방법은 정보가 있는 조건에서 작동한다. 다음은 비선형 변형과 DeltaAI 격자.
- K4만 실패: neural 구현 문제. MAE와 encoder 디버그가 다음 일이고, 그동안 고정 reader로 진행한다.
- K2만 실패하고 K3 통과: 인증 상한이 아직 느슨하다. §4의 $\tilde p$ 품질을 먼저 본다.
- K1 실패: 판정 predictor가 장치를 못 읽는다. 사다리 3단계의 학습을 먼저 고친다.
- K3 실패하고 K5 통과: 작업 모형과 맞지만 oracle에 못 간다. $\sigma=0.3$에서 예측 편향은 0.065이므로 K3 기준 0.10을 넘으면 결과모형이나 집계의 문제다.

$\sigma=0.5$와 $0.85$는 판정에 쓰지 않고 추세로 보고한다. 예측 편향은 posterior-mean 표현, probit link 기준으로 각각 0.164와 0.374다($\sigma=0.3$은 0.065). bounded link에서는 같은 식을 link에 맞게 다시 계산해 보고한다.

## 8. 보고 형식

$\sigma$ × seed마다 표 하나:
분할 | encoder | $\widehat g$ | $r$ | $b_1$ | 상한 | 0.005/0.01/0.02 판정 | 잘린 비율 | $\widehat\theta_S$ | SE

그리고 $\sigma$마다: plurality 추정값, 비교군 전부, 방법 간 차이와 SE, K1부터 K5의 판정, 닫힌 식 예측값과 실측값, 실행 시간.

결과 파일: `results/single_image/decisive_v1/decisive_sigma{σ}_seed{seed}.json`과 요약 `decisive_summary_v1.json`. 로그는 같은 폴더.

## 9. 비용과 멈춤 규칙

이전 실측(Mac CPU, $n=24{,}000$, 분할 하나 30바퀴 약 41분)에서 외삽하면 이 규약은 $\sigma$ 세 값 × seed 세 개 × 분할 다섯 × encoder 둘로 Mac에서 수십 시간이다. 그래서 순서를 둔다.
1. $\sigma=0.3$, seed 하나, 분할 다섯, 두 encoder. 이것만으로 K1부터 K5의 1차 판정이 나온다. 예상 4시간에서 6시간.
2. 1번이 6시간을 넘으면 멈추고 실측 시간을 보고한다. 나머지는 DeltaAI로 간다. DeltaAI 실행은 명령과 GPU-hour 예산을 따로 승인받는다.
3. 1번이 끝나면 $\sigma=0.3$의 나머지 두 seed, 그다음 $\sigma=0.5$, $0.85$.

## 10. 바꾸지 않는 것

영역 마스크, 인접 관계, 고정 요약 $\chi$, reader 구조, $d_Z=2$, $L_{\mathrm{keep}}$의 정의, $\lambda$ 갱신 규칙, 사다리의 세 단계, $\delta^2$ 격자, 판정 기준 K1부터 K5. 실행 중 이 중 하나를 바꾸면 그 실행은 development run으로 표시하고 규약 v2를 새로 쓴다.
