# Sim5 단일 MNIST 이미지 CNN 구현 PRD v1

작성일: 2026-09-18 CDT  
상태: **REVISION 1.1 · OPERATIONAL ARM · DEV RUN COMPLETE · STOPPED BY DECISION 15** (`memo/2026-09-18-sim5-cnn-dev-result.md`)  
대상 원고: `manuscript/main/2.tex`, `3.tex`, `4.tex`  
보존 대상: `memo/2026-09-17-sim5-v2-digit-proxy-prd-v1.md`와 기존 코드·결과 전체

## Revision 1.1 (2026-09-18, 사용자 승인)

- **결정 3 변경.** 이 CNN 실행은 Algorithm 1 절차를 따르되 정리 조건을 인증했다고 주장하지 않는다.
  정리의 조건과 집계 보증은 positive-control 감사가 맡는다. P0 결과는 `memo/2026-09-18-sim5-cnn-p0-feasibility.md`.
- **개발 질문.** 같은 warm 표현에서 균형 학습이 $|\widehat\tau-1|$을 줄였는가. 핵심 비교는 raw CNN, 균형 전(warm), 균형 후(final) 셋이며
  warm과 final은 심사, nuisance, AIPW, 그래프 집계를 똑같이 거친다. $X$-only와 선별 후보의 평균·중앙값은 ablation이다.
- **6절에서 바꾼 공학 기본값** (dev 결과를 보기 전에 정함, `results/sim5_cnn/dev_protocol_v1.json`에 기록):
  warm start 하나를 모든 split이 공유하고 trunk를 고정한다. 균형 학습은 split별 head만 갱신한다(100회, lr $10^{-3}$, critic 5회마다 갱신).
  code는 256개 hard code 대신 연속 16차원이다. augmented critic은 뺀 픽셀을 고정 trunk 특징으로 읽는다. warm start 상한은 60 epoch다.
- **개발 실행.** 자료 seed 71,000,000, split seed 71,000,001, 알고리즘 seed 71,000,002. 문턱 $t=0.002$. $n_D=6000$, $n_N=6000$, $n_E=20000$, $m=128$.
  시간 측정(seed 71,800,000, 시간만 출력): 전체 약 40분 CPU, GPU 0분.

## 0. 문서의 목적과 완료 정의

이 문서는 고정된 요구사항, 이후 구현할 부분, 구현 전에 풀어야 할 수학적 설계 문제를 구분한다.
실행자는 먼저 P0 타당성 검사를 끝내고, P0가 통과한 경우에만 CNN 개발 실행으로 넘어간다.
이 문서의 작성 완료는 DGP 타당성, 원고 정리 조건, CNN 성능의 검증 완료를 뜻하지 않는다.

최종 사용자의 목표는 하나의 실제 MNIST 이미지 $W$를 64개 좌표 블록으로 나누고,
잠재 digit·exemplar label을 보지 않는 CNN으로 원고의 Algorithm 1을 실행하는 것이다.
성공 결과는 강한 matched raw-image CNN보다 유의하게 정확하면서 SCM의 참 ATE에 가까워야 한다.
사용자는 하나의 DGP, 빠른 실행, 최대 10 GPU-hour의 조건부 예산을 선택했다.

현재 reference DGP는 완전하게 명세할 수 있다.
그러나 전체 이미지에서 잠재상태 $U$를 복원할 수 있으므로 ideal raw-image adjustment의 population target도 $\tau=1$이다.
따라서 “raw 방법의 population MAE가 적어도 $0.20$”이라는 선호 목표는 현재 reference DGP에서 구조적으로 성립하지 않는다.
reference DGP를 실행한다면 raw-CNN의 실패는 finite-sample 학습 실패로만 해석해야 한다.

두 번째 P0 blocker는 Algorithm 1 정리의 후보 universe다.
$J=64$이면 전체 후보 수는 $T=2^{64}-2$이고, 정리의 $N$, $\Delta$, $L$은 $m=128$개 표본이 아니라 전체 $T$개 learned candidates에 대해 정의된다.
전체 universe를 symmetry classes로 환원하거나 모든 split에 적용되는 analytic certificate가 없으면 literal theorem PASS를 선언할 수 없다.

## 1. 권위와 원고 대응

### 1.1 관측 자료와 block orientation

원고 Section 2는 관측치를 $O=(X,W,A,Y)$로 정의한다 (`main/2.tex:1–9`).
원고는 $W=(W_1,\ldots,W_J)$를 사전 지정된 블록으로 분할한다 (`main/2.tex:15–21`).
split $S$에서 held-out 입력은 $W_S$이고 representation 입력은

$$
V_S=(X,W_{-S})
$$

이다 (`main/3.tex:155–169`).
CNN은 $A$, $Y$, $U$, digit label, exemplar label을 forward 입력으로 받을 수 없다.

### 1.2 고정 split의 식별 조건

원고의 exact identification은 다음을 함께 요구한다.

1. consistency와 outcome integrability (`main/3.tex:15–23`),
2. latent exchangeability와 common held-out channel (`main/3.tex:25–46`),
3. outcome-relevant completeness, 이하 ORC (`main/3.tex:49–106`),
4. representation overlap (`main/3.tex:108–115`),
5. $(X,W_S)\perp A\mid Z_S$ (`main/3.tex:116–122`).

이 조건들이 성립하면 representation adjustment functional $\theta_S$가 $\tau$와 같다.
관측된 낮은 Brier gap만으로 2번이나 3번을 증명할 수 없다.

### 1.3 근사 balance와 learned representation

원고는 residual treatment discrepancy $D_{\mathrm{res},S}$를 정의한다 (`main/3.tex:254–293`).
원고의 approximate bias bound는 overlap 상수 $\eta$와 stability 상수 $\Gamma_S(\phi)$ 아래에서

$$
|\tau_\phi-\tau|
\le
\frac{\Gamma_S(\phi)}{\eta(1-\eta)}D_{\mathrm{res},S}(\phi)
$$

를 준다 (`main/3.tex:309–346`).
원고 Section 4는 Brier-risk 차이로 empirical discrepancy를 정의한다 (`main/4.tex:37–81`).
CNN의 learned representation은 approximate empirical minimizer 조건을 충족해야 한다 (`main/4.tex:121–147`).
비볼록 optimizer가 낸 checkpoint라는 사실만으로 이 조건을 인증할 수 없다.

### 1.4 honest AIPW와 Algorithm 1

표현 학습 표본 $D$, nuisance 표본 $N$, 평가 표본 $E$는 서로 독립이어야 한다 (`main/4.tex:6–20`).
honest AIPW 정의와 conditional error radius는 `main/4.tex:189–291`에 있다.
Algorithm 1은 다음 순서를 고정한다 (`main/4.tex:312–338`).

1. 전체 $\mathfrak S$에서 $m$개 split을 균일 비복원추출한다.
2. 각 split에서 $D$만 사용해 representation을 학습하고 screen한다.
3. retained split마다 $N$으로 nuisance를 학습하고 $E$에서 AIPW estimate를 계산한다.
4. 추정치 거리가 $2\rho$ 이하인 후보를 연결한다.
5. largest component의 중앙값을 반환한다.
6. 최대 component가 동률이면 모든 최대 component의 중앙값을 candidate set으로 반환한다.

Algorithm 1의 finite-sample theorem은 plurality, uniform radius, distinct-value separation을 요구한다 (`main/4.tex:386–405`).
원고는 approximate valid candidates에 대해 $\rho+b$를 쓰는 확장도 명시한다 (`main/4.tex:406–410`).
이 확장은 원고에 충실하지만 $b$의 독립적 상계가 있어야 한다.

## 2. 25개 인터뷰 결정 ledger

| 번호 | 동결된 결정 | 구현상 결과 |
|---:|---|---|
| 1 | strong raw-CNN에 유의하게 승리하고 oracle에 가까워야 한다 | paired data-seed comparison과 truth error를 모두 보고한다 |
| 2 | 숫자별 20개, 총 200개 고정 MNIST bank를 허용한다 | fixed empirical bank에서 unit을 복원추출한다 |
| 3 | learned CNN 실험 자체가 literal Algorithm 1 finite-sample theorem을 검증해야 한다 | 과거 exact control로 대체하지 않는다 |
| 4 | DGP는 하나만 사용한다 | coefficient grid와 여러 가족을 만들지 않는다 |
| 5 | 빠르게 실행하고 필요하면 최대 10 GPU-hour까지 허용한다 | smoke timing 전 GPU-hour를 가정하지 않는다 |
| 6 | raw 비교군은 같은 CNN과 총 training/tuning budget을 받는다 | compute ledger를 방법별로 저장한다 |
| 7 | MAE $\le0.05$와 oracle-excess one-sided 95% UCB $\le0.05$ | oracle error가 0이므로 UCB criterion이 실질적 primary gate다 |
| 8 | raw의 큰 오차는 12번에서 정한다 | 25% 상대개선 gate는 추가하지 않는다 |
| 9 | SSL과 $A/Y$ warm-start를 허용한다 | $U$, digit, exemplar label 사용은 금지한다 |
| 10 | 여러 $n$보다 하나의 $n$과 독립 반복을 우선한다 | primary $n$은 P0·timing 뒤 confirmation 전에 동결한다 |
| 11 | population raw bias가 우선이고 finite-sample failure는 fallback이다 | 두 주장을 섞지 않는다 |
| 12 | raw mean MAE가 적어도 $0.20$이어야 한다 | reference DGP의 구조적 blocker를 P0에 기록한다 |
| 13 | background는 순수 비정보성이며 별도 treatment marker가 없다 | state-dependent foreground 이외 cue를 만들지 않는다 |
| 14 | oracle은 SCM truth $\tau=1$이다 | true-nuisance fitted oracle을 별도 성능 방법으로 만들지 않는다 |
| 15 | 첫 end-to-end development failure 후 중단·보고한다 | 사후 rescue loop를 금지한다 |
| 16 | $n$은 이론과 timing 근거로 추천한다 | confirmation 결과를 보고 $n$을 선택하지 않는다 |
| 17 | $E$는 training sample보다 크게 둘 수 있다 | 독립성·공정성을 유지하며 radius를 줄이는 용도로만 쓴다 |
| 18 | end-to-end development dataset은 하나다 | 작은 반복으로 성공을 주장하지 않는다 |
| 19 | confirmation은 fresh independent datasets 20개다 | data seed가 추론 단위다 |
| 20 | return rate $\ge0.95$, no-return은 0으로 대체하지 않는다 | conditional loss의 분모도 공개한다 |
| 21 | primary $J=64$ | $J=196$ stress arm을 만들지 않는다 |
| 22 | $m=128$, 전체 nonempty proper splits에서 균일 비복원추출 | sampled $R$을 theorem의 $N$으로 오인하지 않는다 |
| 23 | ablation 유의성은 성공 gate가 아니다 | 원인 분해 표로만 쓴다 |
| 24 | learned finite/discrete code를 허용한다 | label-free hard code를 provisional architecture로 둔다 |
| 25 | 이번 승인 범위는 PRD 문서뿐이다 | 코드와 실험을 시작하지 않는다 |

## 3. Reference SCM의 완전한 명세

### 3.1 고정 MNIST bank와 잠재상태

MNIST source는 `materials/benchmarks/mnist/mnist.npz`다.
파일 SHA-256은 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`이다.
각 digit $d\in\{0,\ldots,9\}$에서 train bank의 처음 20개 이미지를 source order 그대로 고른다.
정규화된 template을 $T_{d,h}\in[0,1]^{28\times28}$, $h\in\{1,\ldots,20\}$로 쓴다.

잠재상태는 $U=(D,H)$이고 200개 쌍에서 균등하다.
각 unit은 bank에서 복원추출하므로 fixed empirical distribution에서 i.i.d.다.
이 설정은 전체 자연 MNIST 분포가 아니라 finite-state semisynthetic benchmark다.

### 3.2 digit severity와 handwriting style

$$
f(d)=d-4.5+2\sin(\pi d/5),
\qquad
r(d)=\frac{f(d)-10^{-1}\sum_{k=0}^9 f(k)}
{\max_{k}|f(k)-10^{-1}\sum_{\ell=0}^9f(\ell)|}.
$$

template의 mean gray를 $\operatorname{ink}(d,h)$라 한다.
digit별 평균을 $\overline{\operatorname{ink}}_d=20^{-1}\sum_h\operatorname{ink}(d,h)$라 한다.
style score는

$$
s(d,h)=
\frac{\operatorname{ink}(d,h)-\overline{\operatorname{ink}}_d}
{\max_{k,\ell}|\operatorname{ink}(k,\ell)-\overline{\operatorname{ink}}_k|}
$$

이다.
Digit과 exemplar label은 generator와 사후 evaluator만 사용한다.

### 3.3 관측 covariates, treatment, outcome

$\epsilon_1,\epsilon_2,\epsilon_4,\epsilon_Y$는 서로 독립인 표준정규 변수다.
$X_3$는 $\{-1,1\}$에서 균등하며 나머지 원시변수와 독립이다.

$$
X_1=0.6r(D)+\epsilon_1,
\quad
X_2=0.6s(D,H)+\epsilon_2,
\quad
X_4=\tanh\{r(D)s(D,H)\}+\epsilon_4.
$$

$$
G(U)=1.2r(D)+0.8s(D,H)+0.4r(D)s(D,H).
$$

$$
P(A=1\mid U,X)=
0.1+0.8\Phi\{G(U)+0.3X_3+0.2\tanh(X_1)\}.
$$

$$
\tau_U=1+0.25r(D)+0.15s(D,H),
$$

$$
h_X=0.3\tanh(X_1)-0.2\tanh(X_2)+0.3X_3+0.2\tanh(X_4).
$$

잠재 potential outcome은

$$
Y(a)=a\tau_U-cG(U)+h_X+\{0.3+0.2|s(D,H)|\}\epsilon_Y,
\qquad c=4.2996324990
$$

이다.
Gauss–Hermite 계산에서 $E(\tau_U)=1$, naive contrast는 $-0.5$, $X$-adjustment target은 약 $-0.53656$이다.
상수 $c$는 naive contrast를 $-0.5$로 만드는 population 값이다.
이 수치는 generator preflight target이지 learner 선택 기준이 아니다.

### 3.4 하나의 이미지 $W$와 64개 블록

각 pixel $p$에 대해 독립적으로 $\xi_p\sim\operatorname{Unif}[-0.05,0.05]$를 생성한다.
관측 이미지는

$$
W_p=0.1+0.8\{T_{D,H,p}+\xi_p\}.
$$

$W$는 하나의 $28\times28$ grayscale image다.
별도 treatment marker, digit label channel, exemplar ID channel은 없다.
모든 bank template에서 공통으로 0인 pixel만 pure background라고 부른다.
그 pixel의 conditional law는 $U$, $A$, $Y$에 의존하지 않는다.
전체 border나 특정 block이 blank라고 가정하지 않는다.

행과 열 index를 각각 `numpy.array_split(arange(28), 8)` 규칙으로 나눈다.
두 index 조각의 Cartesian product가 $8\times8=64$개 disjoint spatial blocks를 만든다.
각 block의 한 변 길이는 3 또는 4다.

200-state bank와 $\sigma=0.05$ bounded noise에 대한 사전 구조 검사에서,
무작위 orientation의 양쪽이 모든 latent-state 쌍을 구별하지 못할 union bound는 $0.179$였다.
따라서 가능한 decoder 관점의 하한은 $0.821$이다.
이 값은 CNN 성공률, screen 통과율, target-valued candidate 비율, plurality margin이 아니다.

## 4. P0: 비용을 쓰기 전 반드시 해결할 문제

### 4.1 P0-A: population raw-bias 목표

full image에서 $U$가 복원되고 $X$도 관측되면 full raw adjustment는 population에서 $\tau=1$을 식별한다.
현재 reference DGP는 preferred claim인 “raw population MAE $\ge0.20$”을 만족하지 않는다.

P0-A의 허용 판정은 둘 중 하나다.

- `STRUCTURAL PASS`: 별도 승인된 단일-image DGP가 원고 가정을 유지하면서 raw population error $\ge0.20$을 수학적으로 보인다.
- `FINITE-SAMPLE FALLBACK`: reference DGP를 유지하고 raw-CNN MAE $\ge0.20$을 finite learner 결과로만 판정한다.

이 PRD는 새로운 structural-bias DGP를 발명하지 않는다.
사용자가 fallback을 이미 허용했으므로 P0-A의 현재 상태는 `STRUCTURAL FAIL / FALLBACK AVAILABLE`이다.

### 4.2 P0-B: 전체 candidate universe

$$
\mathfrak S=\{S\subset\{1,\ldots,64\}:1\le |S|\le63\},
\qquad T=2^{64}-2.
$$

Algorithm 1은 이 집합에서 $m=128$개를 균일 비복원추출한다.
정리의 $N=|\widehat{\mathcal B}|$는 전체 $T$개 중 learned screen을 통과할 후보 수다.
$\Delta$와 $L$도 전체 learned candidate values에서 정의된다.

P0-B는 다음 중 하나를 요구한다.

1. 전체 split을 유한한 symmetry classes로 환원하고 CNN 학습·screen·$\theta_S$가 class-invariant임을 증명한다.
2. 모든 split에 동시에 적용되는 analytic certificate로 $N$, $\Delta$, $L$을 계산하거나 유효하게 bound한다.

128개 결과에서 $R/m$, component 크기, empirical frequencies를 계산하는 것은 진단이다.
그 값을 $N/T$, $\Delta$, $L$의 theorem certificate로 쓰지 않는다.
P0-B가 해결되지 않으면 literal Algorithm 1 theorem verification은 `NOT VERIFIED`다.

### 4.3 P0-C: exact 또는 certified approximate candidates

Exact path는 target-valued split마다 $\theta_S=\tau$를 수학적으로 보여야 한다.
CNN code가 200 states를 잘 분류했다는 유한 test accuracy는 equality proof가 아니다.

Approximate path는 원고의 $\rho+b$ 확장을 사용한다.
각 split의 bias band는

$$
b_S\ge
\frac{\Gamma_S(\widehat\phi_S)}{\eta(1-\eta)}
D_{\mathrm{res},S}(\widehat\phi_S)
$$

를 독립적으로 인증해야 한다.
Observed Brier gap, ATE truth error, bootstrap interval을 $b_S$로 정의하지 않는다.
Exact와 approximate candidate counts를 한 plurality count로 합치지 않는다.

### 4.4 P0-D: uniform radius와 separation

정리의 radius는 전체 $S\in\widehat{\mathcal B}$에 동시에 적용되어야 한다.
평가표본 분산뿐 아니라 signed nuisance remainder의 유효한 상계를 포함해야 한다.
독립 $E$의 bootstrap만으로 nuisance bias를 덮었다고 선언하지 않는다.

서로 다른 population candidate values $c,c'$는

$$
|c-c'|>4\rho
$$

를 만족해야 한다.
Approximate path에서는 원고의 지시에 따라 $\rho+b$를 대신 사용한다.
Numerical clustering이 separation assumption을 생성하지 않는다.

### 4.5 P0 stop rule

P0-B, P0-C, P0-D 중 하나라도 해결되지 않으면 expensive CNN confirmation을 시작하지 않는다.
그 경우 산출물은 blocker, 반례 또는 미충족 inequality와 필요한 사용자 선택이다.
10 GPU-hour를 소진한 뒤 theorem verification 불가능을 발견하는 순서를 금지한다.

## 5. 조건부 구현 파일 지도

P0가 통과하거나 사용자가 명시적으로 operational empirical arm으로 범위를 낮춘 뒤에만 다음 파일을 만든다.

| 제안 파일 | 단일 책임 |
|---|---|
| `code/sim5_cnn_dgp.py` | bank manifest, SCM, image, block masks, row generation |
| `code/sim5_cnn_models.py` | masked CNN, discrete encoder, critics, nuisances, raw comparator |
| `code/sim5_cnn_algorithm1.py` | uniform split draw, screen, AIPW, radius graph, tie return |
| `code/run_sim5_cnn.py` | immutable protocol validation과 stage runner |
| `code/check_sim5_cnn_theorem.py` | P0 universe, exact/approximate band, radius, separation 검사 |
| `code/test_sim5_cnn.py` | unit, adversarial, integration tests |

이 표는 파일 생성 승인이 아니다.

## 6. Provisional CNN specification

이 절의 수치는 **미검증 engineering defaults**다.
P0와 one-dataset timing 후 confirmation protocol에서 동결하기 전에는 성능 기준이 아니다.

### 6.1 Split encoder

split $S$마다 retained pixel만 남기고 나머지는 0으로 둔다.
입력 채널은 masked grayscale image와 retained binary mask 두 개다.
모든 split이 같은 architecture와 optimizer 규칙을 사용한다.

CNN trunk의 provisional 구조는 다음과 같다.

1. `Conv2d(2,16,3,padding=1)`, SiLU, `MaxPool2d(2)`.
2. `Conv2d(16,32,3,padding=1)`, SiLU, `MaxPool2d(2)`.
3. `Conv2d(32,64,3,padding=1)`, SiLU, `AdaptiveAvgPool2d((4,4))`.
4. flatten 1024, concatenate $X\in\mathbb R^4$.
5. `Linear(1028,128)`, SiLU, `Linear(128,256)` code logits.

hard code는

$$
C_S=\operatorname*{argmax}_{k\in\{1,\ldots,256\}}\ell_{S,k}
$$

이고 representation은 $Z_S=(X,\operatorname{onehot}(C_S))$다.
256 categories는 200 latent states를 표현할 용량을 주지만 recovery를 보장하지 않는다.

Training surrogate는 straight-through hard one-hot과 temperature-softmax를 사용할 수 있다.
모든 theorem-facing empirical risk는 hard code에서 다시 계산한다.
Surrogate gradient가 exact ERM을 인증한다고 쓰지 않는다.

### 6.2 Label-free warm start

Warm start는 digit·exemplar label을 쓰지 않는다.
허용 손실은 다음 세 항이다.

$$
L_{\mathrm{warm}}
=L_{\mathrm{masked\ recon}}
+L_{A,\mathrm{BCE}}
+L_{Y,\mathrm{MSE}}.
$$

각 가중치의 provisional default는 1이다.
Adam learning rate는 $10^{-3}$, weight decay는 $10^{-4}$, batch size는 256, 최대 epoch는 30이다.
Early stopping patience는 5이고, warm checkpoint는 balance selection의 초기 후보로 보존한다.

### 6.3 Brier critics와 balance objective

Base critic은 $(X,\operatorname{onehot}(C_S))$를 읽는 `256+4 → 128 → 64 → 1` MLP다.
Augmented critic은 같은 $Z_S$와 held-out full pixels·held-out mask를 읽는다.
Held-out pixel branch는 encoder와 같은 폭의 독립 CNN을 사용하며 hand-crafted scalar로 대체하지 않는다.

두 critic 출력은 provisional $\eta=0.05$에 맞춰 $0.05+0.90\operatorname{sigmoid}(\cdot)$로 $[0.05,0.95]$에 제한한다.
이 mapping은 fitting objective 자체에 포함하며 적합 뒤의 사후 clip으로 구현하지 않는다.
두 empirical Brier risks를 같은 $D$ rows에서 계산한다.

$$
\widetilde D_{S,n_D}^2(\phi)
=
\max\{\widehat R_{0,S}(\phi)-\widehat R_{1,S}(\phi),0\}.
$$

Base는 zero start 하나, augmented는 lifted-base·zero starts와 unoptimized lifted fallback을 쓴다.
Provisional critic budget은 Adam $10^{-3}$, batch 256, 최대 40 epochs, patience 5다.
Encoder balance budget은 Adam $10^{-4}$, 최대 50 epochs, critic refresh 5 epochs마다다.
Selected checkpoint, 모든 optimizer status, signed unclipped gap, clipped gap을 저장한다.

이 최적화는 approximate ERM 후보를 만든다.
Global optimum이나 $\varepsilon_{\mathrm{enc},S}$ 상계를 계산하지 못하면 Section 4의 representation-learning oracle inequality는 `NOT VERIFIED`다.
그러나 이 미검증 상태를 Algorithm 1의 conditional aggregation theorem 실패로 자동 전파하지 않는다.
고정된 learned candidates가 plurality, uniform radius, separation을 직접 만족한다고 독립적으로 인증하면 aggregation theorem은 별도로 판정할 수 있다.

## 7. 표본 역할과 row accounting

Primary 구현은 one-shot three-way split이다.
Cross-fitting이나 rotate averaging은 literal theorem 결과를 대체하지 않는다.

Provisional timing defaults는 $n_D=6000$, $n_N=6000$, $n_E=20000$이다.
이 수치는 confirmation freeze가 아니다.
P0와 첫 end-to-end timing이 끝난 뒤 하나의 primary size를 동결한다.

각 row는 정확히 하나의 outer role만 가진다.

- $D$: SSL, $A/Y$ warm start, Brier encoder·critic optimization.
- $N$: 고정된 $Z_S$의 propensity와 two-arm outcome nuisance fits.
- $E$: 각 retained split의 AIPW scores와 method comparison.

모든 AIPW score의 encoder와 nuisance는 그 score의 $E$ row를 보지 않고 적합되어야 한다.
Row ledger에는 dataset seed, row ID, role, split mask, model digest를 기록한다.
$D\cap N$, $D\cap E$, $N\cap E$가 모두 공집합인지 test한다.

## 8. Candidate sampling과 screen

각 dataset seed와 독립인 split RNG seed를 protocol에 고정한다.
RNG는 64-bit unsigned word를 균일하게 뽑고 all-zero와 all-one을 reject한다.
중복을 reject하여 정확히 128개가 될 때까지 반복한다.
이 절차는 $2^{64}-2$개 nonempty proper subsets에서 균일 비복원추출이다.

Screen threshold $t$는 development ATE 결과를 보기 전에 사전 고정한다.
Algorithm 1의 conditional aggregation theorem은 고정된 learned screen 결과에 조건화하므로, $t$ 자체가 uniform discrepancy bound에서 유도될 필요는 없다.
Uniform critic·encoder bound는 representation-learning oracle inequality를 함께 주장할 때 필요하다.
그 bound가 없으면 learning-bound claim만 `NOT VERIFIED`로 표시하고 fixed-candidate aggregation theorem은 별도로 검사한다.

Screen은 empirical Brier discrepancy와 overlap을 함께 검사한다.
Negative signed gaps, critic reliability, actual propensity min/max를 모두 저장한다.
Low gap을 common-channel 또는 ORC 검정이라고 부르지 않는다.

## 9. Nuisance fitting과 matched raw-CNN

PROBE nuisance는 고정된 $Z_S$에서 propensity와 arm-specific outcome regression을 학습한다.
Provisional MLP는 `260 → 128 → 64 → 1`, Adam $10^{-3}$, batch 256, 최대 60 epochs, patience 8이다.
Propensity는 provisional $\eta=0.05$를 fitting map $0.05+0.90\operatorname{sigmoid}(\cdot)$에 직접 넣어 $[0.05,0.95]$ 범위에 둔다.
실제 fitted propensity의 최소·최대와 $[0.1,0.9]$ 밖 비율을 별도 진단으로 저장한다.

Matched raw-CNN은 $(X,W)$ 전체를 읽고 balance objective를 사용하지 않는다.
그 CNN trunk는 6.1절과 같은 layer 폭을 사용한다.
Propensity와 two-arm outcome heads는 같은 $N/E$ rows를 사용한다.
Raw arm은 128개 split refinement를 포함한 PROBE의 aggregate training·tuning budget과 같은 총예산을 받는다.
이 예산은 사전 지정한 raw restarts와 ensemble candidates에 배분하고, model selection은 $D/N$에서만 하며 $E$를 보지 않는다.
가짜 optimizer step이나 의미 없는 padding으로 예산을 소진하지 않는다.
사용하지 않은 예산은 그대로 기록하며 추가 결과 탐색에 전용하지 않는다.
두 방법의 backward passes, restarts, tuning trials, ensemble 크기와 wall-time ceiling을 confirmation freeze 전에 ledger로 고정한다.

추가 ablation은 다음 세 개다.

1. $X$-only AIPW.
2. masked CNN warm representation without Brier balance.
3. screen-passing candidates의 unweighted mean 또는 median without graph.

Ablation별 유의한 개선은 통과 조건이 아니다.

## 10. Honest AIPW, radius graph, return rule

각 retained split의 estimate는 원고 Definition 4.2의 AIPW score를 따른다.
$E$ score와 그 influence-value variance를 split별로 저장한다.

Exact theorem arm의 common radius는 모든 $S\in\widehat{\mathcal B}$에 대한 valid simultaneous bound여야 한다.
Sampled retained splits에 대한 bootstrap maximum은 전체-universe radius를 대체하지 않는다.
Operational diagnostic은 별도 이름으로 저장한다.

Candidate $S,S'$는

$$
|\widehat\theta_S-\widehat\theta_{S'}|\le2\rho
$$

일 때 연결한다.
Approximate theorem arm에서는 원고의 인증된 $\rho+b$를 사용한다.
Largest connected component가 하나이면 그 추정치들의 중앙값을 반환한다.
여러 component가 최대 크기로 동률이면 median set을 반환하고 primary run에서는 no-single-return failure로 센다.
$R=0$, 비유한 estimate, radius certificate 실패도 no-return이다.

## 11. 실행 단계와 stop conditions

### Stage P0: 수학적 feasibility

P0-A부터 P0-D까지 결과를 machine-readable ledger로 만든다.
P0-A의 structural path가 실패해도 이미 허용된 finite-sample fallback을 명시적으로 선택하면 진행할 수 있다.
P0-B부터 P0-D 중 하나라도 unresolved이면 literal-theorem CNN run을 시작하지 않는다.

### Stage S0: generator-only preflight

고정 200-state bank, template hash, $r$, $s$, $c$, overlap, $E(\tau_U)=1$, naive $=-0.5$, $\theta_X\approx-0.53656$을 재현한다.
이미지 range, block cover, common-zero pixels, noise independence를 검사한다.
이 단계는 CNN 성능 실험이 아니다.

### Stage S1: one end-to-end development dataset

P0가 허용한 단일 protocol로 128개 split, screen, nuisance, AIPW, graph aggregation까지 실행한다.
실제 wall time, peak device memory, fit counts를 기록한다.
Catastrophic condition은 row overlap, latent-label leakage, nonuniform split draw, missing optimizer trace, nonfinite loss, theorem ledger 불완전이다.
Catastrophic failure가 하나라도 있으면 중단하고 보고한다.
성능 실패도 사용자 결정 15에 따라 rescue tuning 없이 보고한다.

### Stage S2: freeze

P0 status, DGP branch, $n_D,n_N,n_E$, architecture, optimizer, $t$, $\rho$ construction, all seeds와 gates를 결과 보기 전에 동결한다.
Source와 protocol hash를 unique run directory에 snapshot한다.

### Stage S3: fresh confirmation

서로 독립인 20개 dataset seed에서 전체 pipeline을 새로 학습한다.
같은 seed의 관측 rows를 모든 방법이 공유한다.
한 setting의 20개를 다른 setting과 합쳐 100개 독립표본처럼 보고하지 않는다.
10 GPU-hour ceiling을 넘을 것으로 실측되면 자동 시작하지 않고 예상시간과 최소 변경안을 먼저 보고한다.

## 12. Frozen empirical gates

Truth는 $\tau=1$이다.
독립 dataset seed가 통계적 단위다.

1. **Single return:** 적어도 19/20 datasets에서 단일값을 반환한다.
2. **Oracle proximity:** 반환 dataset의 $|\widehat\tau-1|$에 대한 one-sided 95% mean upper bound가 $0.05$ 이하다.
3. **Raw level:** matched raw-CNN의 mean absolute error가 적어도 $0.20$이다.
4. **Paired raw win:** joint-return datasets의 $|e_{\mathrm{PROBE}}|-|e_{\mathrm{raw}}|$ one-sided 95% upper bound가 0보다 작다.
5. **Implementation/theorem:** 선택한 exact 또는 certified approximate theorem ledger의 모든 필수 조건이 PASS다.

Oracle MAE mean $\le0.05$는 2번이 통과하면 기술적으로 중복이므로 descriptive statistic으로 함께 보고한다.
사용자가 고르지 않은 25% relative improvement와 p90 threshold를 gate로 추가하지 않는다.

No-return에 0, 1 또는 다른 효과값을 impute하지 않는다.
Loss summary는 returned datasets에 조건부임을 제목과 분모에 표시한다.
Return gate는 전체 20개를 분모로 계산한다.
PROBE가 단일값을 반환한 dataset 수를 $k$라 하고 그 absolute errors를 $a_1,\ldots,a_k$라 한다.
Oracle-proximity upper bound는

$$
\overline a+t_{0.95,k-1}\frac{s_a}{\sqrt{k}}
$$

로 계산한다.
Paired raw comparison에서도 joint-return dataset 수를 $k_{\mathrm{pair}}$라 하고
$d_i=|e_{\mathrm{PROBE},i}|-|e_{\mathrm{raw},i}|$에 같은
$\overline d+t_{0.95,k_{\mathrm{pair}}-1}s_d/\sqrt{k_{\mathrm{pair}}}$ 공식을 쓴다.
Raw-CNN이 한 dataset에서라도 반환하지 않으면 그 dataset을 조용히 제외하지 않고 raw-level·paired 판정을 `FAIL/NOT VERIFIED`로 둔다.
$k<2$, $k_{\mathrm{pair}}<2$, 표준편차 비유한, 필수 결과 누락은 사전 정의한 degenerate/missing failure다.
이 Student-$t$ upper bound는 20-dataset 근사 CI이며 distribution-free bound나 Algorithm 1 theorem certificate가 아니다.
20개 empirical benchmark는 population success probability의 joint 95% certificate가 아니다.

## 13. Result schema와 provenance

Protocol JSON은 다음을 포함한다.

- protocol ID, created time, phase, dataset seed list, split seed list,
- exact DGP coefficients와 MNIST SHA-256,
- 200 source indices와 per-template hashes,
- block construction과 $J=64$, $T=2^{64}-2$, $m=128$,
- sample roles와 sizes,
- architecture, losses, optimizer, epochs, restarts, code size,
- screen, radius, exact/approximate theorem mode,
- five empirical gates와 no-return rule,
- source snapshot paths와 SHA-256.

Per-dataset raw JSON은 다음을 포함한다.

- generated row-role digests와 disjointness result,
- 128 split masks in draw order,
- split별 encoder/critic/nuisance seed와 checkpoint digest,
- signed and clipped Brier risks, overlap, screen reason,
- $\widehat\theta_S$, score variance, nuisance-bound fields, $\rho$, $b$,
- graph edges, all components, largest-component tie status, returned candidate set,
- raw, $X$-only, warm, PROBE estimates and absolute errors,
- optimizer status, update counts, wall time, device, peak memory,
- `complete`와 explicit failure reason.

Summary JSON은 raw input paths와 hashes를 기록하고 20 datasets에서 다시 계산 가능해야 한다.
기존 output을 overwrite하지 않는다.
JSON은 `allow_nan=False`와 atomic rename으로 쓴다.

## 14. Test plan

### 14.1 Generator tests

- 정확히 20 templates per digit와 200 unique bank states.
- bank SHA와 source index order.
- $E(r)=E(s)=0$, $E(\tau_U)=1$.
- propensity가 $[0.1,0.9]$ 안에 있음.
- 64 masks가 disjoint이고 784 pixels를 정확히 한 번 덮음.
- common-zero pixel noise가 state-independent임.
- no marker, no label channel, no hidden metadata in learner view.

### 14.2 Sampling and honesty tests

- 128 masks가 unique, nonempty, proper임.
- fixed seed replay가 bitwise identical함.
- empirical uniformity sanity test는 구현검사일 뿐 theorem certificate로 사용하지 않음.
- $D,N,E$ row intersections가 0임.
- 각 $E$ score가 자신을 학습한 encoder/nuisance fit에 들어가지 않음.

### 14.3 Model tests

- forward input에 $U,D,H$가 없음.
- block ordering permutation과 mask permutation이 동일 규칙을 보존함.
- hard code가 $\{1,\ldots,256\}$ 안에 있음.
- augmented critic class가 base prediction을 lifted fallback으로 포함함.
- same-$D$ Brier risk와 unclipped signed gap을 재현함.
- reported checkpoint가 실제 optimizer state와 일치함.

### 14.4 Aggregation adversarial tests

- distance $2\rho$ 경계의 edge 포함.
- chain-connected component 처리.
- unique largest component median.
- tied largest components가 모든 medians를 반환함.
- candidate set output이 single-return success로 오기록되지 않음.
- $R=0$과 nonfinite estimate가 fail closed함.

### 14.5 Theorem-ledger tests

- $N$, $\Delta$, $L$ 필드가 sampled $R$에서 만들어지지 않음.
- exact and approximate candidate classes가 분리됨.
- $b$가 observed outcome error에서 계산되지 않음.
- uniform radius가 declared universe와 동일한 index set을 사용함.
- separation은 distinct population values에 대해 검사됨.
- 하나의 missing certificate가 전체 theorem PASS를 차단함.

## 15. 보고 표와 그림

필수 표는 다음 네 개다.

1. P0 assumptions와 PASS/FAIL/NOT VERIFIED 및 근거.
2. 20 datasets의 return, PROBE, raw, $X$-only, warm estimates.
3. 다섯 frozen empirical gates와 exact denominators.
4. compute ledger와 matched-budget 검사.

필수 그림은 다음 세 개다.

1. candidate estimates와 $\rho$ 또는 $\rho+b$ graph component plot.
2. paired absolute-error PROBE 대 raw plot.
3. screen gap 대 candidate error scatter, 단 이 그림을 causal certificate로 해석하지 않음.

## 16. 확인된 사실과 미확인 주장

### 확인된 사실

- fixed 200-state bank와 SHA-256이 존재한다.
- SCM 수식은 $\tau=1$, naive $=-0.5$, $\theta_X\approx-0.53656$을 준다.
- treatment overlap은 생성식으로 $[0.1,0.9]$다.
- $J=64$ spatial masks의 decoder-existence union bound 진단은 $0.179$다.
- 원고는 exact Algorithm 1 theorem과 certified approximate $\rho+b$ 확장을 모두 명시한다.

### 미확인 주장

- 현재 reference DGP에서 raw population MAE $\ge0.20$.
- learned CNN이 latent state 또는 outcome-relevant function을 정확히 보존함.
- full $T=2^{64}-2$ universe의 $N$, $\Delta$, $L$ certificate.
- finite $\Gamma$, valid $D_{\mathrm{res}}$ upper bound, nuisance-bias bound.
- uniform radius와 separation.
- CNN approximate ERM tolerance.
- 10 GPU-hour 안의 전체 confirmation runtime.
- five empirical gates의 통과.

## 17. 최종 go/no-go 규칙

문서 작성 이후의 첫 작업은 코딩이 아니라 P0-A부터 P0-D까지의 feasibility memo다.
Preferred structural raw-bias path가 불가능하면 이미 승인된 `FINITE-SAMPLE FALLBACK`을 protocol에 명시하고 진행할 수 있다.
이 fallback은 raw 우월성의 해석만 바꾸며 Algorithm 1 theorem 조건을 면제하지 않는다.
P0-B부터 P0-D의 literal aggregation-theorem certificate가 불가능하면 `NO-GO`다.
Operational empirical claim으로 범위를 낮추려면 새 사용자 승인과 protocol revision이 필요하다.
그 변경을 현재 PRD의 literal-theorem PASS로 소급하지 않는다.

P0가 통과하면 Stage S0와 S1만 먼저 실행한다.
첫 development failure는 그대로 보존하고 보고한다.
성공값을 얻기 위한 DGP, threshold, seed, architecture rescue loop는 이 PRD 범위가 아니다.
