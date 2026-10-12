# 단일 이미지 PROBE: 균형 objective 비교 규약 v1

작성: 2026-10-09 CDT. 작성자: `single-proxy-balancing` (수학 검토: `math-theory-agent`). 지시: 용한 ("그렇게 계획세워봐"). 상위 문서: 계획 v2(`2026-10-08-single-image-probe-plan-v2.md`), 결정적 검사 규약 v2(`2026-10-09-single-image-decisive-test-v2.md`, 코드 세션 작성). 이 규약은 결정적 검사의 **다음** 실험이며, 그 생성기와 fold를 그대로 쓴다.

실행 주체는 코드를 맡은 세션이다. 이 세션은 결과를 검토한다.

## 0. 질문

논문의 조건은 $A\perp T\mid Z$ 하나이고, 이 조건은 $A$가 이진이므로 $D^2=\mathbb E[\{P(A=1\mid T,Z)-P(A=1\mid Z)\}^2]=0$과 동치다. 질문은 다음이다. **reader 같은 중간 표현 없이, 픽셀을 받는 encoder $\phi$를 균형 조건만으로 학습해도 논문의 선형 해만큼 되는가. 된다면 어떤 objective가 가장 안정적인가.**

지금까지의 결과: 고정 reader 위의 선형 PROBE는 $\sigma=0.3$에서 참 불균형 $0.0004$, 추정값 $1.09$(oracle $1.00$)였다. 요약 복원 $L_{\mathrm{keep}}$을 섞은 교대학습 neural encoder는 모든 split에서 그보다 나빴다. 이 규약은 $L_{\mathrm{keep}}$을 빼고 균형만 쓰는 세 objective를 같은 조건에서 비교한다.

## 1. 바꾸지 않는 것

- 생성기: 결정적 검사 v2의 phantom 그대로. bounded link $0.1+0.8\Phi(1.5U+1.0I)$, image-only, 장치는 영역 5, 몸통 기준 밝기 보정. $\sigma\in\{0.3,\ 0.85\}$.
- 표본과 역할: $n=48{,}000$, R 12,000, D·C·P·S·E 각 7,200. seed $\{20261009,\ 20261010\}$.
- 분할: singleton $S_1,\dots,S_5$. 떼어 둔 쪽 $T_S$는 보정 전 원 픽셀. 어떤 objective도 $T$를 고르지 않는다.
- encoder 함수군: 결정적 검사 v2의 신경망 $\phi$. 입력은 떼어 둔 영역을 가린 전체 영상, 출력 2차원에 밝기 기준 스칼라를 더해 $Z\in\mathbb R^3$.
- 검사(audit): 결정적 검사 v2의 predictor 사다리. P에서 새로 적합, S에서 채점, 사다리 3단계(원본 픽셀 CNN)가 판정. $\delta^2$ 격자 $\{0.005,0.01,0.02\}$.
- 효과: 같은 E에서 AIPW, 방법 간 paired 표준오차.
- 기준선: 고정 reader 위의 rank 2 선형 PROBE(수준 A).

학습 objective만 바뀐다.

## 2. 비교하는 objective

세 arm 모두 overlap 벌점 $\mu\cdot\mathbb E[\mathrm{softplus}(\widehat q_0-0.95)+\mathrm{softplus}(0.05-\widehat q_0)]$을 더한다. $\widehat q_0$는 $Z$ 위의 logistic 회귀로 20 step마다 다시 맞춘다. 이것은 증인이 아니라 벌점용이다.

### O1. Brier 차이 (논문의 objective, $L_{\mathrm{keep}}$ 없음)
- 손실: $\widehat R_0(\widehat q_0)-\widehat R_1(\widehat q_1)$.
- 고리: 결정적 검사 v2의 교대학습 그대로. encoder 20 step마다 두 predictor를 다시 수렴시킨다. 30바퀴.
- v2와의 차이는 $L_{\mathrm{keep}}$ 항이 없다는 것뿐이다.

### O2. log-loss 차이 (조건부 상호정보)
- 손실: $\widehat{\mathrm{CE}}_0(\widehat q_0)-\widehat{\mathrm{CE}}_1(\widehat q_1)$, 교차 엔트로피 차이. 모집단에서 $I(A;T\mid Z)$다.
- 고리: O1과 같다.

### O3. kernel conditional independence (안쪽 고리 없음)
- $T$의 고정 특징 $f(T)$: 떼어 둔 영역을 $16\times16$으로 줄인 픽셀 256개와 기존 고정 high-pass 특징을 이어 붙이고 R에서 표준화한다. 학습하지 않는다.
- kernel: $f(T)$에 Gaussian kernel $K_T$(대역폭은 R에서 median heuristic), $A$에 선형 kernel $K_A=aa^\top$의 중심화, $Z$에 Gaussian kernel $K_Z$(대역폭은 미니배치 median, 매 step 갱신).
- 손실:
  $$
  \mathcal L_{\mathrm{KCI}}(\phi)=\frac{1}{m^2}\,\mathrm{tr}\bigl(\tilde K_T\,\tilde K_A\bigr),\qquad
  \tilde K=R\,K\,R,\quad R=\epsilon\,(K_Z+\epsilon I)^{-1},\quad \epsilon=10^{-2}.
  $$
  미니배치 $m=512$. $\mathcal L_{\mathrm{KCI}}$는 $K_Z$를 통해 $\phi$로 미분된다 `[Axiom, 미확인: KCI 통계량의 형태와 일치성]`.
- 갱신: encoder 600 step, Adam, 학습률은 O1과 같다. predictor 재적합이 없다.
- 멈춤과 후보 선택: C에서 $\mathcal L_{\mathrm{KCI}}$가 세 번 연속 1% 넘게 줄지 않으면 멈춘다. C에서 가장 낮은 checkpoint를 고른다.

### 초기값
- (i) MAE 가중치. 결정적 검사 v2와 같다.
- (ii) warm start: 고정 reader 위의 선형 해 $Z^{\mathrm{lin}}$을 목표로 CNN을 회귀 사전학습해 논문의 해에서 출발한다. 사전학습은 R의 이미지와 그 선형 해만 쓴다.

O1과 O3는 두 초기값 모두, O2는 (ii)만 돌린다.

## 3. 측정

분할 × arm × 초기값 × seed마다 다음을 기록한다.

1. **참 불균형의 근사** $\widehat D^2_{\mathrm{fresh}}$: 학습한 $\phi$와 $T$를 고정한 채 생성기에서 $10^5$개를 새로 뽑고, 그 표본에 사다리 3단계 구조의 $q_0^\ast$, $q_1^\ast$를 충분한 epoch로 적합한 뒤 $\frac1{10^5}\sum(q_1^\ast-q_0^\ast)^2$를 계산한다. 이것이 $p(T,Z)$ 대비 불균형의 근사이고, 결정적 검사 v2가 쓴 $p(L_{1:5},I)$ 대비 값과 다르다. $q^\ast$의 validation Brier와 두 번째 복제와의 차이를 함께 적는다.
2. **검사 gap** $\widehat g$와 사다리 단계별 $R_0,R_1$ (S에서).
3. **효과 후보** $\widehat\theta_S$와 수준 A 대비 paired 차이의 표준오차 (E에서).
4. **overlap**: $\widehat q_0$의 $[0.05,0.95]$ 밖 비율과 잘린 비율.
5. **흔들림**: 같은 arm 안에서 초기값 둘과 seed 둘에 대한 $\widehat\theta_S$의 범위.
6. **시간**: arm당 encoder 학습 시간.

## 4. 사전 지정한 판정

$\sigma=0.3$, 유효 분할 $S_1$부터 $S_4$, seed 둘의 평균으로 판정한다. 수준 A를 기준선으로 쓴다.

| 번호 | 기준 | 통과 조건 |
|---|---|---|
| R1 | 균형 | arm의 $\widehat D^2_{\mathrm{fresh}}$가 수준 A의 1.5배 이하 |
| R2 | 정확도 | arm의 $|\widehat\theta_S-\tau|$가 수준 A의 값 더하기 paired 표준오차 1배 이하 |
| R3 | 안정성 | 초기값과 seed에 대한 $\widehat\theta_S$ 범위가 수준 A의 2배 이하 |
| R4 | 장치 탐지 | $S_5$의 검사 gap이 $S_1$부터 $S_4$의 최댓값보다 큼 (모든 arm) |

판정 규칙:
- O3가 R1부터 R3을 통과: reader를 빼고 "픽셀에서 균형만 맞춘다"를 채택한다. 다음 실험(SCM-1 거울 phantom, 아래 §7)은 O3로 한다.
- O3가 R1 실패, 사다리 3단계가 R에서 큰 gap을 보고: kernel 증인의 사각지대다. $f(T)$를 넓히고(32×32, 더 많은 특징) 한 번 더 돌린다.
- O1과 O2가 R3 실패, O3는 통과: 안쪽 고리가 불안정의 원인이다. 교대학습을 버린다.
- 세 arm 모두 R2 실패: 픽셀에서 균형만으로는 선형 해를 못 따라간다. reader를 유지하고, 이것을 메모리의 future work("독립성을 달성하는 representer를 찾는 보편적 최적화")에 대한 첫 음성 결과로 기록한다.
- R4 실패: 판정 predictor 문제이고, 결정적 검사 v2의 K1과 같은 이유로 장치 계수를 1.5로 올려 다시 본다.

$\sigma=0.85$는 판정에 쓰지 않고 추세로 보고한다. 거기서는 측정 잡음의 바닥 때문에 모든 arm이 수준 A와 비슷하게 치우칠 것으로 예상한다.

## 5. 비용과 순서

- O3는 안쪽 고리가 없어 분할당 10분 안팎으로 예상한다. O1과 O2는 결정적 검사 v2의 실측(다섯 분할과 검사에 137분)으로 보면 분할당 25분 안팎이다.
- 순서: (1) $\sigma=0.3$, seed 하나, O3 두 초기값, 다섯 분할. (2) 같은 조건에서 O1 두 초기값, O2 하나. (3) seed 둘째. (4) $\sigma=0.85$.
- 멈춤: (1)과 (2)를 합쳐 Mac에서 6시간을 넘으면 멈추고 실측을 보고한다. 나머지는 DeltaAI로 가고, 그 실행은 명령과 GPU-hour를 따로 승인받는다.

## 6. 보고 형식

$\sigma$ × seed마다 표 하나: 분할 | arm | 초기값 | $\widehat D^2_{\mathrm{fresh}}$ | 검사 $\widehat g$ | 잘린 비율 | $\widehat\theta_S$ | 수준 A 대비 차이 ± SE | 시간. 그리고 R1부터 R4의 판정표.

결과 파일: `results/single_image/objective_v1/objective_sigma{σ}_seed{seed}.json`과 요약 `objective_summary_v1.json`.

## 7. 그다음

이 규약이 끝나면 phantom을 논문 SCM-1의 거울로 바꾼다. 영역 넷 중 하나를 $L_4=U-3I+\sigma\varepsilon_4$($W_4$형)로, 관측 공변량 $X\in\mathbb R^2$를 처치와 결과에 넣고, 비교군에 역할을 맞게 준 2SLS와 틀리게 준 2SLS를 넣는다. 그 실험이 "PROBE가 필요한 상황에서 PROBE가 작동한다"를 보이는 본 실험이고, 이 규약은 그 실험에 쓸 encoder를 고르는 예비 실험이다.

## 8. 바꾸지 않을 것

생성기, 역할 분할, $T$의 정의, encoder 함수군, 사다리, $\delta^2$ 격자, 판정 R1부터 R4, 초기값 둘. 실행 중 바꾸면 development run으로 표시하고 v2를 쓴다.
