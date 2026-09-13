# SCM-1부터 SCM-5까지, 그리고 다음 실험 계획

작성일: 2026-09-12
근거 실행: `results/probe_structured_scm_brier_g_confirm_summary_v1.json` (`overall: PASS_5_OF_5`)
구현: `code/probe_structured_scm.py`의 `SCM_FAMILY`
이전 이름: 개발 중에는 `g1_low_noise`부터 `g5_sinh_low_noise`까지였다. 저장된 결과 파일은 옛 이름을
그대로 쓰므로 `resolve_scm()`이 두 이름을 모두 받는다.

---

## 1. 다섯 SCM은 무엇인가

### 1.1 다섯이 공유하는 구조

먼저 기호를 세운다. $U$는 관측되지 않는 교란변수다. $I$와 $C$는 관측되지만 proxy 블록 **안에**
들어 있는 변수다. $A$는 처치, $Y$는 결과, $X$는 관측 공변량이다. $M_j$는 잠재 측정값이고
$W_0, \ldots, W_4$는 실제로 관측되는 다섯 개의 proxy 블록이다. $U$, $I$, $C$와 모든 오차항은 서로
독립인 표준정규분포를 따른다. $\Phi$는 표준정규 누적분포함수다.

$$X = U + 2\epsilon_X, \qquad M_j = U + \sigma_j \epsilon_j \quad (j = 0, \ldots, 4)$$

$$W_0 = \bigl(I,\; C,\; T(M_0)\bigr), \qquad
W_j = T(M_j) \;\; (j = 1, 2, 3), \qquad
W_4 = T(M_4 - 0.6\, I)$$

$$P(A = 1 \mid U, I, C, X) = 0.1 + 0.8\,\Phi\bigl(U + 2.5 I + 0.2 X + 0.3 C\bigr)$$

$$Y(a) = a - c_U\, U + 0.2 X - 0.5 C + \text{(결과 잡음)}$$

참 평균 처치효과는 다섯 모두 $1$이다.

이 구조에서 중요한 점 세 가지를 먼저 말한다.

**첫째, $W_4$가 오염된 블록이다.** 다른 블록은 $U$의 잡음 섞인 측정값인데, $W_4$에는 $-0.6 I$가
섞여 있다. $I$는 처치에만 영향을 주는 원인이므로, $W_4$를 잘못 다루면 균형이 깨진다. 이것이 후보
블록 집합 중 어떤 것이 유효한지를 실제로 가려야 하는 이유다.

**둘째, $C$는 수동적인 측정값이 아니다.** $C$는 $W_0$ 안에 관측되면서 처치와 결과 **양쪽에** 직접
영향을 준다. 따라서 이 설계는 "proxy는 잠재변수의 수동적 흔적일 뿐"이라는 쉬운 경우가 아니다.

**셋째, 무조정 차이가 음수다.** 모집단 계산에서 약 $-0.174$이고 SCM-3만 $-0.200$이다. 참 효과가
$+1$이므로 부호가 뒤집혀 있다. 확증 실행에서도 다섯 SCM 각각 20회 전부 무조정 차이가 음수였다.

### 1.2 다섯이 서로 다른 점

다섯은 같은 인과구조 위의 다섯 변형이다. 각각 한 가지 축만 바꾼다.

| 이름 | 무엇을 바꾸는가 | 결과 잡음 | $\sigma_j$ | 관측 변환 $T$ | $c_U$ | 효과 이질성 |
| --- | --- | --- | --- | --- | ---: | ---: |
| **SCM-1** | 기준이 되는 기본형 | $0.3\,\epsilon_Y$ | 전부 $0.85$ | $T(x) = x$ | 2.5 | 없음 |
| **SCM-2** | 결과 잡음이 잠재변수에 의존 | $(0.3 + 0.2\lvert U\rvert)\,\epsilon_Y$ | 전부 $0.85$ | $T(x) = x$ | 2.5 | 없음 |
| **SCM-3** | 처치효과 자체가 잠재변수에 의존 | $0.3\,\epsilon_Y$ | 전부 $0.85$ | $T(x) = x$ | 2.7 | $a(1 + 0.3U)$ |
| **SCM-4** | 블록마다 측정 잡음이 다름 | $0.3\,\epsilon_Y$ | $(0.8, 0.9, 1.0, 0.85, 0.95)$ | $T(x) = x$ | 2.5 | 없음 |
| **SCM-5** | proxy를 비선형 렌즈로 봄 | $0.3\,\epsilon_Y$ | 전부 $0.85$ | $T(x) = \sinh(x)$ | 2.5 | 없음 |

각 변형이 무엇을 시험하는지 한 줄씩 적는다. SCM-2는 결과 모형의 이분산이 AIPW 추정을 흔드는지
본다. SCM-3은 처치효과가 일정하다는 가정 없이도 평균 효과를 되찾는지 본다. SCM-4는 블록의 품질이
고르지 않을 때 후보 선택이 좋은 블록 쪽으로 기우는지 본다. SCM-5는 관측 변환이 단조 비선형일 때
표현 학습이 그 변환을 흡수하는지 본다.

### 1.3 확증된 성적

$n = 6000$, 독립 자료 20회, 역할 회전 4회다. 표는 평균 절대오차이고 참값은 $1$이다.

| SCM | PROBE | $X$ 조정 | $(X,W)$ 조정 | Oracle | PROBE p90 | raw 대비 상한 | oracle 초과 상한 | 단일값 반환 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SCM-1 | 0.0280 | 0.9514 | 0.1854 | 0.0052 | 0.0541 | −0.1386 | 0.0332 | 20/20 |
| SCM-2 | 0.0292 | 0.9526 | 0.1843 | 0.0094 | 0.0597 | −0.1343 | 0.0308 | 20/20 |
| SCM-3 | 0.0294 | 0.9688 | 0.1892 | 0.0062 | 0.0577 | −0.1407 | 0.0342 | 20/20 |
| SCM-4 | 0.0298 | 0.9514 | 0.2001 | 0.0052 | 0.0563 | −0.1490 | 0.0355 | 20/20 |
| SCM-5 | 0.0285 | 0.9514 | 0.1859 | 0.0052 | 0.0504 | −0.1374 | 0.0333 | 20/20 |

동결한 관문은 평균 절대오차 $\le 0.05$, 90백분위수 $\le 0.10$, 단일값 반환율 $\ge 0.95$,
$X$ 및 $(X,W)$ 대비 짝지은 오차 차이의 95% 상한이 $0$ 미만, oracle 초과 오차의 상한이 $0.05$
이하다. 다섯 모두 통과했다.

**Algorithm 1의 블록 선택이 실제로 일했다.** 30개 블록 분할을 전부 처리했고, 오염된 $W_4$를 포함한
잘못된 후보가 screen을 SCM별로 16회에서 18회 통과했지만 **최종 연결집단에는 한 번도 들어오지
않았다.** 최종 집단에는 올바른 후보가 118개에서 125개 남았다. screen만으로는 거르지 못한 것을
집계 단계가 걸러냈다는 뜻이다.

### 1.4 확정하지 않은 것

이 다섯은 같은 인과구조의 다섯 변형이다. 다음은 아직 확인하지 않았다.

- 전역 경험위험최소화. 사용한 것은 제한된 모수 표현과 예측모형이다.
- 신경망 인코더와 이미지 proxy.
- 표본이 커질 때의 점근적 수렴. $n = 6000$ 한 점에서만 확인했다.
- 원고의 유한표본 균일 보장.
- 고정된 101점 격자를 벗어난 탐색.

또한 여기서 "oracle에 가깝다"는 **허용 오차 기준을 만족한다**는 뜻이다. oracle이 여전히 더
정확하며 통계적 동률이 아니다.

### 1.5 제안된 고차원·이미지 benchmark: SCM-6과 SCM-7

아래 둘은 SCM-1부터 SCM-5까지의 확증 결과에 포함되지 않는다. 아직 코드도 결과도 없는
`proposed/unrun` 설계다. 둘 다 SCM-1의 처치·결과식과 참 ATE $1$을 유지하고, $W$의 관측 형식만
바꾼다. 목적은 새로운 인과구조를 다섯 개 성공시킨 것처럼 세는 것이 아니라, 이미 확인한 식별구조를
고차원 관측과 이미지 관측에서 학습할 수 있는지 시험하는 것이다. 각 benchmark는 정보복원이 쉬운
calibration arm인 A와, 단순 shortcut을 제거해 가설을 더 강하게 검증하는 주 arm인 B를 짝지어 둔다.
SCM-6A/7A와 SCM-6B/7B는 모두 아직 `proposed/unrun`이며, 아래 예상 순서는 결과가 아니다.

#### SCM-6A: 256차원 선형 직교혼합 positive control

잠재 scalar channel은 다음과 같다.

$$
K_j=U+0.85\epsilon_j\quad(j=0,1,2,3),
\qquad
K_4=U+0.85\epsilon_4-0.6I.
$$

각 블록마다 $\xi_j\sim N(0,I_{255})$를 독립적으로 만들고

$$
H_j=Q_j\begin{bmatrix}K_j\\0.5\xi_j\end{bmatrix}\in\mathbb R^{256},
\qquad
W_0=(I,C,H_0),\quad W_j=H_j\;(j=1,2,3,4)
$$

로 관측한다. 따라서 전체 raw $W$는 $258+4\times256=1282$차원이다. $I$와 $C$는 $Q_0$ 안에
섞지 않는다. 두 변수를 섞으면 $X$와의 공분산만으로 복원할 수 없는 별도의 식별·학습 문제가 동시에
추가되기 때문이다.

$Q_j$는 NumPy `PCG64DXSM` seed $620260912+j$로 256×256 표준정규 행렬을 만든 뒤 QR 분해해
한 번만 생성한다.
$R$의 대각 부호를 양수로 만드는 sign rule을 적용하고, float64 little-endian byte hash를 manifest에
저장한다. 같은 $Q_j$를 모든 replicate와 역할 fold에서 사용하지만 learner에는 $Q_j$나 $K_j$를
주지 않는다. $Q_j$가 가역이고 $\xi_j$가 독립 nuisance이므로 모집단에서 $H_j$가 담는 $K_j$ 정보는
사라지지 않는다. 특히

$$
q_j=\frac{\operatorname{Cov}(H_j,X)}{
\lVert\operatorname{Cov}(H_j,X)\rVert_2}=Q_je_1,
\qquad q_j^\top H_j=K_j
$$

이므로 $X$-predictive projection 방향은 관측자료만으로 식별된다. 이 등식은 모집단 사실이고,
유한표본 projection의 정확도를 보장하지는 않는다.

주 분석은 총 $n=6000$, 네 fold 각 1500명이다. 각 회전에서 representation fold를
$D_{\mathrm{emb}}=750$과 $D_\phi=750$으로 미리 나눈다. $D_{\mathrm{emb}}$에서 블록별
$\widehat q_j$와 scale을 추정해 고정하고, $D_\phi$에서만 101점 $r$ 격자와 Brier 목적을 계산한다.
screen, nuisance, evaluation fold에는 embedding을 다시 맞추지 않는다. augmented treatment critic은
held-out $H_j$ 전체 256차원에 대한 선형 branch를 받는다. held-out 블록까지 $\widehat q_j^\top H_j$
하나로 줄인 검사는 별도 compressed-target 진단일 뿐 원고의 full-$T$ discrepancy로 부르지 않는다.

공정한 baseline은 같은 $D_{\mathrm{emb}}$ 접근권과 같은 nuisance/evaluation fold를 쓴다. $X$-only,
1282차원 raw ridge/MLP, 모든 블록을 $X$-predictive projection한 뒤 ordinary adjustment하는 compressed-raw,
그리고 evaluator만 쓰는 true-$K$ diagnostic을 나눠 보고한다. true-$K$와 $Q_j^{-1}$ 결과는 정보보존
검사이지 학습기 성적이 아니다. 주 setting은 차원 $256$, nuisance amplitude $0.5$로 고정하고,
$d\in\{16,64,256\}$과 amplitude $\nu\in\{0,0.25,0.5\}$는 SCM-6 내부 난이도 ablation으로만 둔다.

모집단에서는 SCM-1과 같은 Simpson reversal, raw ordinary-adjustment functional, 7개 target 후보와
1개 잘못된 singleton을 기대한다. 유한표본 성능의 조건부 예상 순서는

$$
\text{oracle}\;\lesssim\;\text{learned PROBE}
\;<\;\text{raw }(X,W)\;<\;X\text{-only}
$$

이지만 아직 결과가 아니다. 먼저 새 개발 seed 5회로 projection 오차, full-heldout screen TPR/FPR,
단일값 반환을 점검한다. 사양을 동결한 뒤에만 겹치지 않는 새 seed 20회로 1.3절 관문을 평가한다.

#### SCM-6B: 등방적 비선형 bilinear carrier

SCM-6A는 선형 공분산으로 $K_j$를 찾을 수 있으므로 고차원 코드와 정보보존을 확인하는 쉬운
positive control이다. 주 분석인 SCM-6B는 같은 잠재 scalar channel, 처치식, 결과식, $I$, $C$를
그대로 두되 선형/PCA shortcut을 제거한다. 각 블록에 대해

$$
v_j=\operatorname{Var}(K_j),\qquad
R_j\sim\operatorname{Unif}\{-1,+1\},\qquad
\xi_j\sim N(0,I_{254})
$$

를 독립적으로 생성하고 carrier를

$$
G_j=
\begin{bmatrix}
R_jK_j/\sqrt{v_j}\\
R_j\\
\xi_j
\end{bmatrix}\in\mathbb R^{256},
\qquad
H_j=QG_j,
$$

$$
W_0=(I,C,H_0),\qquad W_j=H_j\quad(j=1,2,3,4)
$$

로 정의한다. clean block에서는 $v_j=1+0.85^2=1.7225$, 오염 block $4$에서는
$v_4=1+0.85^2+0.6^2=2.0825$다. 전체 raw $W$는 SCM-6A와 같은 1282차원이다.
$I$와 $C$는 $Q$ 안에 섞지 않는다.

$Q$는 다섯 블록 모두에 쓰는 **하나의 같은** 직교행렬이다. NumPy `PCG64DXSM` seed
`621260912`로 256×256 표준정규 행렬을 만든 뒤 QR 분해하고, $R$ factor의 대각이 양수가 되도록
열 부호를 고정한다. float64 little-endian bytes와 SHA-256을 manifest에 저장한다. 같은 $Q$를
모든 replicate와 역할 fold에 고정하지만 learner에는 주지 않는다. block마다 다른 $Q_j$를 쓰면
common-channel 학습과 block 정렬을 동시에 시험하므로 이 주 실험에는 넣지 않고 별도 강건성 분석으로
남긴다.

이 carrier는 Gaussian이 아니다. 정확한 모집단 사실은

$$
E(H_j)=0,\qquad \operatorname{Cov}(H_j)=I_{256},\qquad
\operatorname{Cov}(H_j,X)=0
$$

이다. 따라서 선형 회귀와 population PCA는 $K_j$ 방향을 찾지 못한다. 그러나 $Q$의 첫 두 열을
$q_1,q_2$라 하면

$$
K_j=\sqrt{v_j}\,(q_1^\top H_j)(q_2^\top H_j)
$$

이므로 정보는 정확히 보존된다. 더 구체적으로

$$
\mathcal B_j=E\left[X\{H_jH_j^\top-I_{256}\}\right]
$$

는 $\operatorname{span}(q_1,q_2)$에서만 신호를 갖는 rank-2 행렬이다. 그 양·음 고유벡터를
$u_+=(q_1+q_2)/\sqrt2$, $u_-=(q_1-q_2)/\sqrt2$라 하면

$$
P_j(H_j)=\frac12\left\{(u_+^\top H_j)^2-(u_-^\top H_j)^2\right\}
=\frac{K_j}{\sqrt{v_j}},
\qquad
c_j=\operatorname{Cov}\{P_j(H_j),X\}=\frac1{\sqrt{v_j}},
$$

따라서 $P_j/c_j=K_j$다. 고유벡터의 부호는 제곱에서 사라지고, 양·음 고유값이 두 방향의 순서를
정한다. 이 등식은 evaluator가 확인하는 모집단 구조이지 learner에게 $Q$, $K_j$, $v_j$를 주는
절차가 아니다.

주 data-only front end는 $D_{\mathrm{emb}}=750$을 600개 tensor-fit 행과 150개 calibration 행으로
미리 나눈다. 앞의 600개에서 $\widehat{\mathcal B}_j$를 계산하고 양·음 극단 고유벡터로
$\widehat P_j$를 만든 뒤, 뒤의 150개에서
$\widehat c_j=\widehat{\operatorname{Cov}}(\widehat P_j,X)$를 추정한다. 고유값 separation이
미리 동결한 $\epsilon_{\mathrm{eig}}$ 이하이거나 $\widehat c_j\leq\epsilon_c$이거나 nonfinite이면
fail closed하여 no-return으로 기록한다. 이 threshold들은 5-seed pilot에서만 정하고 confirm 전
동결한다. tensor 방향과 scale을 고정한 뒤 별도의 $D_\phi=750$에서만 101점 $r$ 격자와 Brier
목적을 계산한다. 주 Brier encoder는 이 관측자료 기반 scalar와 $X,C,I$ 중 해당 split에서 이용 가능한
변수만 사용한다. augmented critic은 full held-out $H_j$ branch를 받아야 하며, projected scalar만 본
검사는 compressed-target 진단일 뿐 원고의 full-$T$ discrepancy가 아니다.

반드시 함께 보고할 강한 baseline은 선형 projection, PCA, quadratic spectral recovery,
second-order polynomial ridge/MLP, raw 1282차원 ridge/MLP다. quadratic spectral 방법이 PROBE와
같거나 더 좋으면 결론은 “신경망이 필요하다”가 아니라 “관측자료로 학습한 2차 구조가 충분하다”다.
SCM-6A와 SCM-6B는 동일한 $U,I,C,K,A,Y$와 동일한 자료 seed를 공유해 paired 비교한다.
정보삭제 negative control은 행을 섞지 않는다. 각 $K_j$ 대신 $U,A,Y$와 독립인 새 $K'_j$를 만들고
같은 $R_j,\xi_j,Q$ 생성법으로 새 $H'_j$를 렌더링하며 $I,C$는 유지한다. 이렇게 해야 unit의 iid성과
role 간 독립을 깨지 않는다.

실수값 모집단에서 $K_j$는 $H_j$의 가측함수로 정확히 복원되며, carrier의 추가 nuisance는 $K_j$를
조건으로 $A,Y$와 무관하다. 따라서 clean channel의 Gaussian convolution completeness,
outcome-relevant completeness, 7개 target 후보와 1개 잘못된 singleton 구조는 SCM-1에서 그대로
옮겨온다. 그러나 유한표본 tensor 방향과 Brier encoder가 이를 학습한다는 보장은 없다. 이것이
SCM-6B가 검증할 가설이다.

#### SCM-7A: MNIST-template amplitude-norm positive control

같은 $K_j$를 28×28 이미지로 렌더링한다. 자료원은
[MNIST 공식 페이지](https://yann.lecun.org/exdb/mnist/)의 60,000개 train image이며,
[CVDF 공식 mirror](https://github.com/cvdfoundation/mnist)를 보조 provenance로 기록한다. 이 요청에서는
다운로드하지 않는다. 실제 실행 전 source file hash, class별 image ID, pool 생성 seed를 manifest에
동결한다. 10,000개 공식 test bank는 주 iid 분석에 섞지 않고 별도 style-OOD 분석에만 쓴다.

0이 아닌 train image를 class 안에서 고정 bank로 만들고, 각 template $B_{d,r}$를 한 번만
$\lVert B_{d,r}\rVert_2=1$로 정규화한다. clean channel에는
$v_K=1+0.85^2$, bad channel에는 $v_K=1+0.85^2+0.6^2$를 쓴다. 각 unit과 block에서

$$
d_j=\min\left\{9,
\left\lfloor10\Phi\!\left(K_j/\sqrt{v_{K,j}}\right)\right\rfloor\right\},
\qquad
r_j\mid d_j\sim\operatorname{Uniform}(\text{train bank of class }d_j),
$$

$$
H_j=\exp(K_j/4)B_{d_j,r_j},
\qquad
W_0=(I,C,H_0),\quad W_j=H_j\;(j=1,2,3,4)
$$

로 만든다. 전체 raw $W$ 차원은 $(2+784)+4\times784=3922$다. template draw는 replicate, role,
unit, block마다 독립이고 고정된 class bank 안에서 복원추출한다.
class label과 source image ID는 learner에게 주지 않는다. 렌더링 뒤 per-image normalization, clipping,
binarization, uint8 변환, pixel noise를 모두 금지한다. 실수값 모형에서는

$$K_j=4\log\lVert H_j\rVert_2$$

이므로 channel은 scalar 정보를 정확히 보존한다. float32 구현의 norm 복원 허용오차는 실행 전에
동결한다. 이 설계는 MNIST-template semisynthetic positive control이지, 자연 이미지의 인과 benchmark가
아니다. 또한 밝기 norm이 쉬운 shortcut이라는 한계가 있다. amplitude를 제거하는 per-image-normalized
negative control과 digit-class를 shuffle하는 진단을 함께 두되, exact positive-control 결과와 섞지 않는다.

주 CNN은 모든 이미지 블록이 공유하는
`Conv(1,16,3,pad=1) → ReLU → MaxPool(2) → Conv(16,32,3,pad=1) → ReLU → MaxPool(2)
→ Conv(32,64,3,pad=1) → ReLU → AdaptivePool(4,4) → Linear(1024,32) → ReLU` trunk와
블록별 scalar head를 쓴다. $D_{\mathrm{emb}}=750$을 미리 600개 embedding-fit 행과 150개
scale-calibration 행으로 나눈다. 앞의 600개에서 $X$ 예측 MSE로 trunk와 head를
50 epoch, batch 128, AdamW learning rate $3\times10^{-4}$, weight decay $10^{-4}$로 학습한다.
이는 관측 channel을 찾는 auxiliary loss다. 학습 출력 $P_j=f_j(H_j)$의 scale은 그대로 쓰지 않는다.
calibration 150개에서

$$
\widehat c_j=\widehat{\operatorname{Cov}}(P_j,X),
\qquad
\widehat K_j=\frac{P_j-\overline P_{j,\mathrm{cal}}}{\widehat c_j}
$$

로 block별 scale을 맞춘 뒤 고정한다. 모집단의 이상적 head가 $E(X\mid H_j)=K_j/v_{K,j}$이면
$c_j=1/v_{K,j}$이므로 이 규칙이 $K_j$ scale을 복원하지만, 유한표본의 근사 CNN에 대해서는 보장하지
않는다. $\widehat c_j$가 nonfinite이거나 $\widehat c_j\le\epsilon_c$이면 해당 replicate는
no-return으로 기록한다. $\epsilon_c$는 5-seed pilot에서만 정하고 confirm 전에 동결하며, true
$v_{K,j}$나 $K_j$는 학습에 쓰지 않는다. 가중치와 정규화를 고정한 뒤 $D_\phi=750$에서 Brier
목적으로 $r$을 선택하며, auxiliary-$X$ loss를 Brier discrepancy라고 부르지 않는다.

augmented Brier critic에는 압축된 scalar만이 아니라 raw held-out image를 처리하는 별도 CNN branch를
제공하고 base critic을 정확히 포함시킨다. full raw image를 보지 않는 compressed-target audit은 별도
진단이다. held-out branch는 위와 같은 16/32/64 convolution trunk를 독립된 parameter로 쓰고 32차원
출력을 base features에 잇는다. base/augmented critic은 $D_\phi$에서 batch 128, AdamW learning rate
$10^{-4}$, weight decay $10^{-4}$, 고정 100 epoch로 맞추며 early stopping을 쓰지 않는다. augmented
critic은 lifted-base 초기값을 포함한다. optimizer seed는 data seed와 분리해 기록한다.
architecture, epoch, optimizer, random seed와 제한된 hyperparameter 후보는 5-seed pilot에서만
선택하고 confirm 전에 동결한다. raw-CNN baseline은 같은 shared trunk, $D_{\mathrm{emb}}$ sample,
optimizer, epoch 예산으로 다섯 block embedding을 이어 nuisance를 학습한다. raw-PCA baseline도 같은
$D_{\mathrm{emb}}$의 600/150 fit/calibration 접근권으로 block별 PC와 scale을 맞춘다. PROBE만 MNIST
label, source ID, log-norm inverse, 더 많은 pretraining data를 받지 않는다.

모집단 정보와 exact-inverse diagnostic은 SCM-1과 같은 값을 가져야 하지만, learned CNN embedding이
그 정보를 보존한다는 보장은 없다. 유한표본의 조건부 예상 순서는
$\text{oracle}\lesssim\text{learned PROBE}<\text{raw-CNN}<X\text{-only}$지만 결과로 쓰지 않는다.
주 분석은 총 $n=6000$, 네 fold 각 1500명이고 representation fold는 위와 같이
$D_{\mathrm{emb}}=750$, $D_\phi=750$으로 나눈다. round-trip log-norm 오차, class별/template별 빈도,
source-ID 비노출, full-heldout screen과 compressed-target 진단의 차이를 structural check로 저장한다.
5개 새 pilot seed가 screen, voting, 정보복원 기준을 통과한 경우에만 사양을 동결하고 20개 겹치지 않는
confirm seed로 1.3절 관문을 평가한다.

#### SCM-7B: constant-mass spatial-deformation image channel

SCM-7A는 $\lVert H_j\rVert_2$ 하나로 $K_j$가 복원되는 쉬운 image positive control이다. 주 분석인
SCM-7B는 $K_j$-dependent global amplitude를 제거하고, 동일한 latent $K_j,A,Y$를 공간 이동으로
렌더링한다. 자료원과 고정 train/test bank provenance는 SCM-7A와 같되, 각 원본 MNIST template
$B_{d,r}\geq0$는 먼저

$$
\sum_{p,q}B_{d,r}(p,q)=1
$$

이 되도록 $L^1$ 정규화하고 원본 수평·수직 무게중심 $(\mu_p,\mu_q)$를 저장한다. 주 arm에서는
digit $d$를 $K_j$와 독립인 고정 class-balanced 분포에서 뽑고, 그 class의 고정 train bank에서
template index $r$을 복원추출한다. 즉 주 arm에는 digit-semantic shortcut이 없다. quantile로 digit
class를 정하는 SCM-7A 방식은 secondary semantic arm에서만 사용한다.

각 unit과 block에서 $S_j\sim\operatorname{Uniform}[-8,8]$을 $K_j$와 독립적으로 뽑는다. 원본 pixel
$(p,q)$의 질량을 96×96 canvas의 연속 좌표

$$
x_{j,p}=48+(p-\mu_p)+8\tanh(K_j/3),\qquad
y_{j,q}=48+(q-\mu_q)+S_j
$$

로 옮긴다. $a=\lfloor x\rfloor$, $b=\lfloor y\rfloor$,
$\alpha=x-a$, $\beta=y-b$로 두고 네 이웃 $(a,b)$, $(a+1,b)$, $(a,b+1)$, $(a+1,b+1)$에 각각

$$
(1-\alpha)(1-\beta),\quad \alpha(1-\beta),\quad
(1-\alpha)\beta,\quad \alpha\beta
$$

비율로 질량을 나눈다. signed floor는 음수에서도 위 정의를 그대로 쓴다. 원본 extent가 최대 27이고
수평 이동은 8, 수직 style 이동도 8 이하이므로 target 좌표는 최소 13, 최대 83이고 이웃 pixel도
84 이하여서 crop이 없다. renderer는 clipping, quantization, pixel noise, 사후 정규화를 하지 않은
continuous float image를 반환한다.

이 bilinear renderer는 총질량과 first moment를 보존하므로 실수 산술에서

$$
\sum_{x,y}H_j(x,y)=1,\qquad
\operatorname{centroid}_x(H_j)=48+8\tanh(K_j/3),
$$

$$
K_j=3\operatorname{atanh}
\left\{\frac{\operatorname{centroid}_x(H_j)-48}{8}\right\}.
$$

마지막 역변환은 evaluator-only 정보보존 검사다. learner와 screen에는 상수 $3$, $8$, true $K_j$,
digit label, source ID를 주지 않는다. 전체 raw $W$ 차원은
$2+5\times96^2=46082$다. exact-channel을 주장하기 전에 mass 오차, centroid 오차, inverse-$K$ 오차,
boundary loss와 $\operatorname{atanh}$ tail 안정성의 허용오차를 pilot 전에 동결하고 모두 통과시킨다.
하나라도 실패하면 그 구현은 approximate channel로 표시하며 exact completeness를 호출하지 않는다.

주 data-only image learner는 모든 블록이 공유하는 96×96 CNN trunk와 block head
$f_j(H_j)\approx E(X\mid H_j)$를 $D_{\mathrm{emb}}$의 600개 fit 행에서 $X$ 예측 MSE로 학습한다.
별도의 150개 calibration 행에서

$$
\widehat c_j=\widehat{\operatorname{Cov}}\{f_j(H_j),X\},\qquad
\widehat K_j=
\frac{f_j(H_j)-\overline{f_j(H_j)}_{\mathrm{cal}}}{\widehat c_j}
$$

를 계산해 head의 중심과 scale을 고정한 뒤 $D_\phi$에서 Brier encoder를 학습한다. 알려진 centroid
역변환 상수는 넣지 않는다. $\widehat c_j\leq\epsilon_c$이거나 nonfinite이면 fail closed하여
no-return으로 기록한다. spline 또는 작은 MLP는 600개 fit 행에서 $f_j$를 근사하는 learner 후보이지,
150개 calibration 행에서 scale을 다시 임의로 학습하는 절차가 아니다. approximate CNN이 $K_j$를
복원한다는 보장은 없다. augmented critic에는 압축 scalar가 아니라 full held-out 96×96 image를
처리하는 branch를 제공해야 한다.

강한 shortcut baseline은 raw-CNN과 PCA뿐 아니라 generic horizontal/vertical centroid, total mass,
second moments, $L^2$ norm과 intensity histogram을 반드시 포함한다. 각 statistic baseline도
600개 fit 행에서 $f_j\approx E(X\mid\text{statistics})$를 학습하고, 같은 150개 calibration 행에서
위의 centered-$f_j/\widehat c_j$ 규칙으로 scale을 맞춘다. 알려진 역변환은 쓰지 않는다.
PROBE, raw-CNN, PCA, image-statistic baseline은 같은 representation-training 행, optimizer 후보 수,
nuisance/evaluation 예산을 쓴다. 총질량을 1로 고정하고 평균 밝기 shortcut을 없앴다고 해서 모든
intensity shortcut이 사라졌다고 주장하지 않는다. subpixel interpolation 때문에 $L^2$ norm 등이
$K_j$와 연관될 수 있으며, 위 baseline이 이를 직접 측정한다. 단순 centroid statistic이 이기면
신경망 필요성을 주장하지 않고 공간적 충분통계를 관측자료로 학습할 수 있었다고 보고한다.

paired arm은 동일한 latent draw와 renderer random seed를 공유한다. 7A의 quantile digit과 7B의
$K$-independent digit은 class가 달라질 수 있으므로 같은 template ID를 공유한다고 주장하지 않는다.
template ID까지 같은 amplitude-vs-shift 비교는 class-selection rule도 동일하게 고정한 별도 paired
control에서만 한다. 7A amplitude-norm과 7B spatial shift를 직접 비교하고, 7B 안에서는
(i) $K$-independent digit + spatial shift를 주 arm, (ii) quantile digit
class + shift 없음을 semantic-only 보조 arm, (iii) 필요하면 quantile digit + shift를 결합 arm으로
분리한다. 정보삭제 control은 $U,A,Y$와 독립인 새 $K'_j$에서 image를 새로 생성하고 $I,C$는 유지한다.
행 permutation은 role 간 의존과 leakage를 만들 수 있어 쓰지 않는다. 정보삭제는 completeness를
깨뜨리므로 balance gap이 작아도 biased point를 반환할 수 있다. 따라서 반드시 abstain하라는 관문을
두지 않고, 여러 독립 seed에서 oracle-close 성능이 정보보존에 의존하는지를 진단한다. 뜻밖에 계속
oracle-close이면 leakage, 우연 또는 다른 관측정보 경로를 조사한다.

7B도 실수값 모집단에서는 $K_j$가 $H_j$의 가측함수로 정확히 복원되며, template와 style nuisance는
$K_j$를 조건으로 $A,Y$와 무관하다. 따라서 SCM-1의 정보, clean-channel completeness, 7개 target와
1개 wrong-candidate 구조를 보존한다. 다만 이것은 MNIST-template semisynthetic spatial positive
control이지 자연 이미지 causal benchmark가 아니며, CNN의 유한표본 성공을 보장하지 않는다.

SCM-6과 SCM-7 모두 기존과 같은 30개 oriented split을 하나도 truth로 제거하지 않고 처리한다.
독립 screen 뒤 공통 radius graph를 만들고, 최대 연결성분의 median을 반환하며, 최대 성분 tie는
set-valued output으로 남긴다. clean, bad, mixed, block-0이라는 구조 label과 오염 블록 정답은 평가표에만
쓰고 representation, screen, Algorithm 1에는 제공하지 않는다.

#### SCM-6/7 업그레이드의 단계별 paired validation

목적은 숫자를 쉽게 높이는 것이 아니라, (i) balance representation 학습, (ii) screen,
(iii) 후보 집계가 각각 최종 인과오차에 기여하는 정도를 분리하는 것이다. 모든 조합을 한꺼번에
Cartesian product로 돌리지 않고 다음 세 단계의 paired contrast만 순서대로 실행한다.

1. **정보와 난이도.** 동일한 latent draw에서 6A 대 6B, 7A 대 7B, 그리고 각 B arm 대 독립-$K'$ 정보삭제
   control을 비교한다. exact-inverse evaluator는 구조 검증에만 쓰며 fitted learner 성적에 섞지 않는다.
2. **학습기.** 같은 B arm에서 선형/PCA shortcut, 가장 강한 prespecified analytic statistic
   (SCM-6B의 quadratic tensor, SCM-7B의 centroid·second-moment·intensity 집합), 그리고 flexible
   MLP/CNN을 비교한다. 주장은 “강한 data-only learner가 oracle-close인가”이며, gradient 방법이나
   neural learner가 반드시 우월해야 한다는 관문은 두지 않는다.
3. **집계.** 같은 자료와 같은 학습된 30-candidate library를 고정한 채 no-balance adjustment,
   screen-only 단일 후보, 모든 retained 후보의 단순평균, 모든 retained 후보의 median,
   Algorithm 1의 최대 연결성분 median을 비교한다. 최대 성분 tie는 임의로 한 점을 고르지 않고
   set-valued output으로 유지한다.

Algorithm 1은 30개 oriented split 전부에 동일하게 적용한다. split budget은
$m\in\{5,10,20,30\}$에서 30개 중 비복원 균등추출하고 sampling seed를 저장한다. 다만 같은 완성된
30-candidate library를 여러 번 subsample한 offline draw는 독립 자료 반복이 아니다. 먼저 각 data seed
안에서 subsampling 평균과 failure rate를 요약한 뒤, data-seed 단위 paired interval을 계산한다.
구조 label은 평가에만 쓰고 $\rho$, screen threshold, graph radius를 truth로 조율하지 않는다.
현재 7 target + 1 wrong 구조에서는 전체 retained median도 충분할 수 있으므로 graph가 median보다
반드시 낫다고 주장하지 않는다. budget별 wrong-member 비율, no-return, tie, unique return과 인과오차를
함께 보고해 실제 기여를 판정한다.

각 B arm은 먼저 개발에 쓰지 않은 새 data seed 5개로 구현 오류, exact-rendering gate, fail-closed 비율,
full-heldout screen과 baseline 학습 가능성을 점검한다. pilot에서 시험한 모든 architecture와 threshold를
기록한 뒤 하나의 사양을 동결한다. 확증은 겹치지 않는 새 data seed 20개를 추론 단위로 한다.
optimizer seed, 네 rotation, $m$-subsample은 독립 반복 수에 넣지 않는다. 새로운 confirm 비교 family는
두 B arm 각각의 PROBE 대 $X$, raw, pilot에서 동결한 최강 shortcut baseline, 그리고 oracle-excess의
8개 paired contrast로 미리 정의한다. 각 contrast는 20개 독립 data seed의 짝지은 차이에 대해

$$
\overline\Delta+t_{1-0.05/8,19}\,\operatorname{SE}(\Delta)
$$

인 Bonferroni one-sided upper bound를 쓴다. 이는 20-seed 근사 구간이며 원고 정리의 certificate가
아니다. Holm 보정 $p$-value는 원하면 보조표에만 싣고 주 판정을 바꾸지 않는다.

주 성능 관문은 1.3절과 동일하다. 단일값 반환율 $\geq0.95$, 평균 절대오차 $\leq0.05$,
90백분위 절대오차 $\leq0.10$, $X$와 raw 대비 짝지은 절대오차 차이의 동시보정 상한 $<0$,
oracle 대비 추가 절대오차의 동시보정 상한 $\leq0.05$를 모두 만족해야 한다. 여기에 (a) full 30-split
membership과 최대성분 구성을 저장하고, (b) 정보삭제 control이 정보보존 arm과 다른 모집단·여러-seed
행동을 보이는지, (c) 가장 강한 shortcut과 flexible learner 중 무엇이 성공을 설명하는지를 보고한다.
정보삭제 한 번의 실패나 성공을 정리처럼 해석하지 않는다. 6B/7B의 기대 순서는 조건부 가설

$$
\text{oracle}\;\lesssim\;\text{learned PROBE}
\;<\;\text{raw }(X,W)\;<\;X\text{-only}
$$

이며, 아직 어떤 숫자도 관측하지 않았다. 실제 결과가 다르면 그대로 실패 또는 동률로 보고한다.

---

## 2. 실험 계획

### 2.0 계획의 논리

1.4절이 열거한 다섯 공백이 그대로 실험 목록이 된다. 순서는 **싸고 결정적인 것부터**다. 각 실험은
앞 실험이 통과해야 의미가 있고, 실패하면 그 자리에서 멈춘다.

이 계획은 지난 일곱 라운드에서 배운 규칙을 그대로 따른다. 기준은 실행 전에 적는다. 평가행은 학습과
선택에 들어가지 않는다. 통계적 반복 단위는 독립 자료이고 optimizer 난수는 반복이 아니다. 모든 결과
파일은 `code/provenance.py`의 헤더와 검증을 거친다. 구현 판정과 성능 판정을 섞지 않는다.

### 2.1 E1. 표본 크기 곡선

**묻는 것.** $n = 6000$의 통과가 한 점에서의 우연인가, 아니면 표본과 함께 좋아지는 추세의 일부인가.

**설계.** 다섯 SCM 전부, $n \in \{1500,\; 3000,\; 6000,\; 12000,\; 24000\}$, 독립 자료 20회.
같은 자료 계열에서 작은 $n$이 큰 $n$의 앞부분이 되도록 중첩한다. 역할 회전은 현재와 같이 4회다.

**미리 정하는 통과 조건.**
- 평균 절대오차가 $1500 \to 24000$ 사이에서 단조로 줄어든다. 인접한 두 크기가 뒤집히는 것은
  허용하되, 양 끝의 비교는 짝지은 95% 구간이 $0$ 아래여야 한다.
- $n = 12000$에서 1.3절의 다섯 관문을 다섯 SCM 모두 통과한다.

**실패하면.** 크기를 키워도 오차가 멈추면 그 값이 표현 오차의 바닥이다. 그 바닥을 모집단
$\tau_Z$로 분해해서 표현 오차와 추정 오차를 나눈다. 이것은 이미 design F에서 쓴 분해다.

**비용.** 5개 SCM × 5개 크기 × 20회 = 500 셀. 가장 큰 두 크기가 시간의 대부분을 쓴다.

### 2.2 E2. 확증 실행

**묻는 것.** 계획서 1.3절의 정식 기준을 다섯 SCM이 통과하는가.

**설계.** $n = 12000$, **개발에 쓰지 않은 새 독립 자료 50회**, 다섯 SCM 전부. 학습기, 격자, 회전
방식은 E1이 끝난 시점의 것으로 동결하고 이후 바꾸지 않는다.

**통과 조건.** 계획서 1.3절 그대로다. 단일값 반환율 $\ge 0.95$, 반환된 반복의 평균 절대오차
$\le 0.10$, PROBE 오차가 $X$ 및 $(X,W)$ 각각의 절반 이하, 세 짝지은 동시보정 구간의 상한이
$0$ 미만이다. 여러 값을 반환하거나 반환하지 못한 반복은 버리지 않고 집합값 평가에 포함한다.

**이것이 논문의 표가 된다.** E1 이전에는 실행하지 않는다.

### 2.3 E3. 학습기 절제

**묻는 것.** 라운드 7에서 design F에서 얻은 진단, 곧 "목적에는 좋은 해가 있는데 gradient 탐색이
못 찾는다"가 이 다섯 SCM에서도 성립하는가.

이 실험이 계획 전체에서 가장 정보량이 크다. 지금까지 성공한 구현과 실패한 구현이 **다른 SCM에서**
돌았기 때문에 둘을 직접 비교한 적이 없다. 같은 SCM, 같은 자료에서 비교하면 그 비교가 처음으로
성립한다.

**설계.** 다섯 SCM, $n = 6000$, 독립 자료 20회, 자료와 분할을 셋이 공유한다.

| 학습기 | 표현 | 탐색 |
| --- | --- | --- |
| A (현재) | 제한된 모수 표현, 계수 하나 | 101점 격자 |
| B | 전체 선형 인코더 | AdamW gradient |
| C | 신경망 인코더 (`probe_brier_neural.py`) | AdamW gradient |

**보고할 것.** 각 조합의 평균 절대오차, 모집단 $\tau_Z$로 분해한 표현 오차와 추정 오차,
그리고 A 대비 짝지은 차이의 구간이다.

**미리 적는 예상.** A는 통과한다(이미 확인했다). B와 C가 실패하면 라운드 7의 진단이 이 SCM
가족으로 옮겨진다. B나 C가 통과하면 라운드 7의 실패는 design F 고유의 것이었다는 뜻이고, 그 경우
design F의 어떤 성질이 원인인지 따로 규명해야 한다. **어느 쪽이 나와도 결론이 있다.**

### 2.4 E4. 탐색 방법 절제

**묻는 것.** E3에서 B와 C가 실패한다면, 목적함수를 그대로 두고 탐색만 바꿔서 고칠 수 있는가.

**설계.** 표현 클래스를 전체 선형 인코더 하나로 고정하고 탐색만 넷으로 바꾼다. 단일 시작점
gradient, 다중 시작점 gradient, 거친 격자에서 세밀한 격자로 좁히기, 미분을 쓰지 않는 탐색이다.
목적함수, 자료, critic, 예산을 전부 같게 맞춘다.

**통과 조건.** 어느 하나가 A의 성적을 짝지은 구간 안에서 따라잡는다.

**이것이 원고에 주는 것.** 4절의 목적함수를 건드리지 않고 그 최소화 절차만 바꾸는 변경이므로
이론을 다시 쓰지 않아도 된다. E3이 B와 C의 실패를 보이고 E4가 고칠 수 있음을 보이면, 논문은
"이 목적함수는 옳고 표준적인 gradient 탐색이 부적합하다"는 구체적 기여를 하나 더 갖는다.

### 2.5 E5. 난이도 경계

**묻는 것.** 다섯 SCM 각각에서 PROBE가 무너지는 지점은 어디인가.

**설계.** SCM마다 한 축을 골라 다섯 값으로 훑는다. 나머지는 고정한다.

| SCM | 훑을 축 | 값 |
| --- | --- | --- |
| SCM-1 | 결과의 잠재변수 계수 $c_U$ | 1.5, 2.5, 3.5, 5.0, 7.0 |
| SCM-2 | 이분산 강도 | 0.0, 0.2, 0.4, 0.8, 1.6 |
| SCM-3 | 효과 이질성 | 0.0, 0.3, 0.6, 1.0, 1.5 |
| SCM-4 | 블록 잡음의 퍼짐 | 전부 0.85, ±0.1, ±0.2, ±0.4, ±0.6 |
| SCM-5 | 비선형 강도 $T(x) = \sinh(\lambda x)$ | 0.5, 1.0, 1.5, 2.0, 3.0 |

**보고할 것.** 축 위에서 1.3절 관문이 마지막으로 통과되는 지점과, 그 지점을 넘어선 뒤 오차가
표현 오차 때문인지 추정 오차 때문인지의 분해다.

**이것이 논문에 주는 것.** 성공 하나가 아니라 성공의 **범위**를 보여준다. 심사자가 "이 다섯은 잘
고른 것 아닌가"라고 물을 때 답이 된다.

### 2.6 E6. 블록 구조 스트레스

**묻는 것.** Algorithm 1의 탐색과 집계가 더 어려운 블록 구조에서도 버티는가.

현재 확증 실행은 블록 5개, 오염 블록 1개, 분할 30개였다. 다음 셋을 바꾼다.

1. **블록 수.** 5, 7, 9개. 분할 수가 30에서 126, 510으로 늘어난다.
2. **오염 블록 수.** 1개, 2개, 3개. 오염이 과반이 되기 직전까지 간다.
3. **오염의 종류.** 지금은 $-0.6 I$ 한 가지다. 여기에 $U$와 무관한 잡음만 있는 블록과, $Y$에
   직접 영향을 주는 블록을 추가한다.

**보고할 것.** 잘못된 후보의 screen 통과율과 최종 집단 진입률을 따로 적는다. 확증 실행에서 그 둘이
16에서 18회와 0회로 크게 달랐으므로, 그 간격이 어려운 구조에서도 유지되는지가 핵심이다.

### 2.7 E7. 이미지 proxy

**묻는 것.** $T(M_j)$를 실제 이미지 렌더링으로 바꿔도 성립하는가.

**설계.** 다섯 SCM 중 SCM-1과 SCM-5를 고른다. $T(M_j)$를 MNIST 숫자 렌더링으로 바꾸고 나머지
생성식은 그대로 둔다. 학습기는 E3과 E4에서 이긴 조합을 쓴다.

**왜 마지막인가.** 이전 E4 MNIST 실험에서 PROBE가 raw에 졌고($1.126$ 대 $0.708$), 그때는 원인을
가릴 수 없었다. 이제 E3과 E4가 학습기와 탐색을 분리해 두었으므로, 여기서 실패하면 원인이 이미지
인코딩 자체라는 것을 처음으로 말할 수 있다.

SCM-6과 SCM-7을 구현하는 경우 E7은 단순히 SCM-1/5에 이미지를 덧씌우는 분석과 구분해 보고한다.
SCM-6A/7A는 쉬운 information-preserving positive control이고, SCM-6B는 bilinear high-dimensional
benchmark, SCM-7B는 constant-mass spatial MNIST-template benchmark다. 각 B arm은 5-seed pilot 뒤
사양을 동결하고, 새 20-seed confirm에서 1.3절과 1.5절의 관문을 판정한다.

### 2.8 E8. discrepancy에서 causal error까지의 calibration

**묻는 것.** Section 4의 Brier discrepancy가 실제로 어느 정도 해상도로 측정되고, 작은 discrepancy가
작은 causal error와 연결되는가. 이 실험은 encoder optimizer를 비교하는 E3/E4와 달리 표현 $r$을
외부 격자에 고정해 critic estimation과 causal amplification만 분리한다.

**설계.** SCM-1부터 SCM-5까지에서 모집단 clean root $r^*$ 주위

$$
r=r^*+\delta,
\qquad
\delta\in\{0,\pm0.01,\pm0.02,\pm0.05,\pm0.10,\pm0.20\}
$$

를 미리 고정한다. 7개 구조적으로 valid한 후보와 ORC-invalid인 bad singleton을 반드시 분리한다.
각 점에서

$$
D_{\mathrm{res}}^2(r)
=E\left[\{P(A=1\mid T,Z_r)-P(A=1\mid Z_r)\}^2\right],
\qquad
\tau_{Z_r},
$$

empirical same-$D$ Brier gap, standard error, screen acceptance를 함께 기록한다. scalar·linear Gaussian
arm에서는 quadrature로 모집단 $D_{\mathrm{res}}^2$와 $\tau_{Z_r}$를 계산한다. SCM-5 및 향후 CNN처럼
학습된 nonlinear representation에서는 Monte Carlo 오차가 있는 population diagnostic이라고 표시하고
`exact`라고 부르지 않는다.

same-$D$ gap은 critic을 맞춘 바로 그 $D$ 행에서 계산한 최적화 목적값이다. 이것과 별도로 critic과
encoder 선택에 전혀 쓰지 않은 $n_{\mathrm{audit}}=12000$개의 독립 audit 행에서 risk gap과 그 standard
error를 다시 계산한다. audit 행은 모든 $D$, calibration, screen, nuisance, final evaluation 행과
겹치지 않는다. 따라서 same-$D$ optimization gap과 out-of-sample critic resolution을 같은 수치로
취급하지 않는다.

$n_D\in\{375,750,1500,3000\}$, 각 50개 독립 seed에서 correctly specified restricted critic과 더 넓은
prespecified critic을 나란히 평가한다. 보고량은 $D^2$ 추정 bias/RMSE, valid와 invalid 후보별
false acceptance, $\sqrt{\widehat D^2}$와 $\lvert\widehat\tau_Z-1\rvert$의 paired scatter와 bin summary,
representation error와 final AIPW estimation error의 분해다. development seed에서 정한 calibration
constant나 bin은 fresh audit seed 전에 동결한다.

이 실험만으로 $\Gamma$를 보장하지 않는다. 독립적인 operator/stability upper bound를 실제로 계산하지
못하면 $\eta(1-\eta)\lvert\tau_Z-1\rvert/D_{\mathrm{res}}$는 필요한 상수의 진단값으로만 보고한다.
empirical 상관이나 fitted upper envelope를 Section 4의 theorem certificate라고 부르지 않는다.

### 2.9 E9. 고정 총표본에서 sample use와 outcome noise의 2×2 절제

**묻는 것.** 현재 성공을 만든 두 변화, 곧 rotate-four sample reuse와 작은 outcome noise가 각각 얼마를
설명하는가.

**설계.** SCM-1의 $A,X,W$와 모든 계수는 그대로 둔다. 총 생성 unit은 모든 cell에서 정확히
$n=6000$, 네 fold 각 1500명이다. 같은 20개 새 base seed와 같은 latent draw, treatment,
$\epsilon_Y$를 네 cell이 공유한다.

| Factor | Level 1 | Level 2 |
| --- | --- | --- |
| outcome noise SD | $1.0$ | $0.3$ |
| sample-use regime | hard four-way split | rotate-four |

restricted Brier critic, 101점 $r$ 격자, screen threshold, nuisance class, Algorithm 1, baseline은 네 cell에서
바꾸지 않는다. 보고량은 MAE, p90, return, common radius, learned-$r$ 변동, nuisance remainder, wall-clock
time이다. 같은 seed의 절대오차 차이로 noise main effect와 sample-use main effect를 paired interval로
계산하고,

$$
\{L_{\mathrm{rotate},0.3}-L_{\mathrm{hard},0.3}\}
-\{L_{\mathrm{rotate},1.0}-L_{\mathrm{hard},1.0}\}
$$

의 interaction도 보고한다. 여기서 rotate-four는 evaluation 표본만 키우는 처리가 아니다.
representation, screen, nuisance, evaluation을 여러 회전에 걸쳐 평균하는 묶음 intervention이므로,
결과를 “E sample size만이 원인”이라고 해석하지 않는다.

### 2.10 기존 계획 문구에 대한 해석 정정

E2의 “1.3절 그대로”와 뒤의 MAE $\le0.10$·baseline 절반 기준은 서로 다르다. 향후 E2 실행은 일단
보류하고, 별도 기준 변경 승인이 없다면 1.3절의 현재 기준, 곧 MAE $\le0.05$, p90 $\le0.10$,
단일값 반환율 $\ge0.95$, paired $X$/raw interval upper $<0$, oracle-excess upper $\le0.05$를 쓰는 것을
권고한다. 이 문장은 기존 `frozen_criteria.json`을 바꾸는 승인이나 사후 기준 변경이 아니다.

E3/E4의 “gradient가 못 찾는다”, “탐색만 고치면 된다”는 문구도 배타적 원인 판정이 아니라 검증할
가설이다. 현재 restricted parametric same-$D$ Brier 구현은 SCM-1부터 SCM-5까지 통과했으므로,
향후 선형/신경망 arm의 실패는 표현 클래스, regularization, critic approximation, optimization,
sample reuse의 상호작용일 수 있다. E8이 critic resolution과 causal amplification을, E9가 sample use와
outcome noise를 먼저 분리한 뒤 E3/E4를 해석해야 한다.

같은 이유로 E7의 실패만으로 원인이 “이미지 인코딩 자체”라고 확정하지 않는다. image channel 변경은
source distribution, finite critic class, optimization, nuisance fitting도 함께 바꾼다. SCM-7의 exact
log-norm diagnostic, compressed embedding, full-heldout critic을 나눠 어느 층에서 정보가 사라지는지
확인한 뒤에만 원인을 좁힌다.

---

## 3. 실행 순서와 판단 지점

| 순서 | 실험 | 선행 조건 | 이 실험이 실패하면 |
| ---: | --- | --- | --- |
| 1 | E8 discrepancy–causal calibration | 없음 | critic 해상도, screen, causal amplification 중 병목을 좁힌다 |
| 2 | E9 sample use × outcome noise | 없음 | 현재 성공이 어느 묶음 intervention에 의존하는지 밝힌다 |
| 3 | E3 학습기 절제 | E8, E9 완료 | 표현 클래스·critic·optimization 가설을 분리해 다음 단계를 정한다 |
| 4 | E4 탐색 절제 | E3에서 B 또는 C 실패 | 탐색만의 문제라는 가설을 직접 시험한다 |
| 5 | E1 표본 크기 곡선 | 없음 | 표현 오차의 바닥을 분해한다 |
| 6 | E5 난이도 경계 | E1 통과 | 성공 범위가 좁다는 사실 자체가 보고할 결과다 |
| 7 | E2 확증 실행 | E1과 E5 통과, 설정과 기준 동결 | 논문 표를 다시 설계해야 한다 |
| 8 | E6 블록 구조 | E2 통과 | Algorithm 1의 적용 범위를 좁혀 서술한다 |
| 9 | E7 및 SCM-6/7 benchmark | E3, E4 완료 | high-dimensional/image pipeline의 손실 지점을 층별로 찾는다 |

**E3을 가장 먼저 두는 이유**를 한 번 더 적는다. 지금 프로젝트에는 통과한 구현 하나와 실패한 구현
하나가 있는데 둘이 서로 다른 SCM에서 돌았다. 그래서 "무엇이 성공을 만들었는가"를 아직 말할 수
없다. 같은 SCM 같은 자료에서 셋을 나란히 돌리는 것이 그 질문에 답하는 가장 짧은 길이다.

위 문단은 초기 계획의 기록으로 남긴다. 현재 실행 우선순위는 표의 정정된 순서처럼 E8과 E9가
E3보다 앞이다. 같은 SCM에서 학습기를 비교하기 전에 discrepancy 해상도와 sample-use/noise 효과를
분리해야 E3/E4의 결과를 optimizer-only 설명으로 과대해석하지 않을 수 있기 때문이다.

---

## 4. 시작 전에 정해야 할 것

다음 다섯 가지는 실행 전에 동결하고 결과를 본 뒤에 바꾸지 않는다.

1. **E1과 E5의 관문.** 1.3절의 다섯 관문을 그대로 쓸지, 큰 표본에서는 더 엄격하게 할지.
2. **E3의 예산 맞추기.** 학습기 A는 101점 격자이고 B와 C는 gradient 단계다. 무엇을 같게 맞출
   것인가. 벽시계 시간, 목적함수 평가 횟수, 또는 둘 다 보고할 것인가.
3. **E5의 축과 값.** 위 표를 그대로 쓸지.
4. **E6의 오염 종류 세 가지의 정확한 생성식.**
5. **SCM-6/7의 provenance와 학습 예산.** $Q_j$ 생성 seed·PRNG·QR sign rule·hash, MNIST
   source/template manifest, $D_{\mathrm{emb}}/D_\phi$ 분할, CNN/PCA/raw baseline 예산과
   pilot-to-confirm 동결 시점.

이 다섯 항목은 용한의 승인이 필요하다. 승인이 나면 `code/frozen_criteria.json`에 날짜와 함께 적고 그
시점 이후의 실행만 증거로 쓴다.
