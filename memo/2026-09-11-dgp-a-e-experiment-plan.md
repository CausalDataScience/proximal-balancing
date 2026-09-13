# PROBE DGP A--E: 고정 실험 계획

작성일: 2026-09-11  
지위: 구현 전 고정 사양(prospective experiment specification)  
대상 이론: 현재 원고 Section 2--4  
주 분석 표본 크기: $n=12{,}000$  
주 반복 횟수: DGP별 50회  

## 0. Output Contract

### 0.1 Deliverable

이 문서는 다섯 DGP A--E의 자료생성, PROBE 학습, 비교 방법, 판정 기준, 진단,
불확실성 계산, 실패 처리와 재현 규칙을 구현자가 추가 결정을 하지 않고 코드로 옮길 수 있게 고정한다.

### 0.2 Consumer와 form

- Consumer는 연구 책임자, 구현자, 결과 검토자다.
- Form은 한국어 Markdown 실험 명세다.
- 이 문서는 결과 보고서가 아니며, 새 알고리즘의 성공을 주장하지 않는다.

### 0.3 Acceptance criteria

완성된 구현은 다음을 만족해야 한다.

1. 아래 공통 SCM과 A--E 채널을 식 그대로 재현한다.
2. 모든 주 분석은 $30$개 held-out set과 세 번의 D/N/E rotation을 사용한다.
3. $E$는 representation, discrepancy/overlap screen 또는 hyperparameter tuning에는 쓰지 않는다.
   $E$에서 계산한 AIPW 후보값은 사전에 고정한 Algorithm 1의 linking과 aggregation에만 쓴다.
4. 주 방법인 PROBE-Brier-neural과 과거 구현 계열인 PROBE-RFF를 구분한다.
5. raw 비교군과 PROBE는 같은 nuisance learner와 같은 영상 encoder 예산을 사용한다.
6. no-return과 여러 component의 tie를 숨기거나 단일 값으로 바꾸지 않는다.
7. DGP별 50회 paired 결과와 사전 고정된 동시추론을 보고한다.
8. 생성식, seed, split, optimizer, checkpoint와 결과 schema로 재현 가능하다.

### 0.4 Evidence

향후 evidence는 생성식 unit test, population diagnostic, 50회 confirmatory JSON/CSV,
paired interval, no-return rate, screen TPR/FPR와 코드 commit hash로 구성한다.
현재 이 문서에는 그 결과가 아직 없다.

### 0.5 Boundaries

- 대화 중 계산한 population scout 값은 검증된 project artifact가 아니다.
- exact ORC가 neural ERM의 성공, exact balance의 유한표본 달성 또는 oracle 성능을 보장한다고 쓰지 않는다.
- continuous Gaussian/image DGP에서 유한 stability constant $\Gamma$를 자동으로 주장하지 않는다.
- 실제 결과를 보고 난 뒤 DGP, $n$, threshold 또는 learner를 바꾸어 주 결과로 재사용하지 않는다.
- 아래에 적은 미래 코드·결과 파일명은 계획이며 현재 파일의 존재를 뜻하지 않는다.

## 1. 연구 질문과 판정 단위

### 1.1 추정대상량

모든 DGP의 처치효과는 상수이며,

$$
\tau=\mathbb E\{Y(1)-Y(0)\}=1.
$$

$A\in\{0,1\}$는 처치, $Y$는 결과, $X\in\mathbb R^{20}$는 관측 공변량,
$U$는 숨은 중증도, $C$는 관측 가능한 기저 중증도, $W=(W_0,\ldots,W_4)$는
다섯 개의 사전 정의된 block이다.

### 1.2 검증할 네 주장

1. 처치 전 proxy block이 연속형, 비선형, count, 합성 image, 실제 MNIST image여도
   Section 2--4의 절차를 operationalize할 수 있는가.
2. $W_0$가 처치를 강하게 예측하지만 숨은 $U$를 직접 측정하지 않을 때,
   모든 $W$를 그대로 조정하는 방법보다 PROBE가 나을 수 있는가.
3. PROBE가 $X$-only Simpson reversal을 고치고 oracle $(X,U,C)$에 가까워지는가.
4. unknown valid block과 component aggregation이 실제로 올바른 후보군을 반환하는가.

이 중 2--4는 empirical ambition이다. Section 3--4의 정리가 모든 DGP에서 그 순위를 자동으로
보장하는 것은 아니다.

### 1.3 주 성공 기준

DGP별 replicate $r$에서 반환값이 단일 값일 때 절대오차를

$$
L_{d,r}(M)=|\widehat\tau_{d,r}(M)-1|
$$

로 둔다. 주 paired contrast는

$$
\Delta^{\mathrm{raw}}_{d,r}=L_{d,r}(\mathrm{PROBE})-L_{d,r}(X,W),
$$

$$
\Delta^{\mathrm{orc}}_{d,r}=L_{d,r}(\mathrm{PROBE})-L_{d,r}(X,U,C)
$$

이다. 최종 판정은 아래 네 family를 함께 사용한다.

동시추론에 쓸 세 paired contrast는

$$
G^X_{d,r}=L_{d,r}(\mathrm{PROBE})-0.5L_{d,r}(X),
$$

$$
G^{\mathrm{raw}}_{d,r}=L_{d,r}(\mathrm{PROBE})-0.5L_{d,r}(X,W),
$$

$$
G^{\mathrm{orc}}_{d,r}=L_{d,r}(\mathrm{PROBE})-L_{d,r}(X,U,C)-0.10.
$$

각 평균의 동시보정 one-sided upper interval이 $0$보다 작아야 한다.

confirmatory 성공은 다음 네 family를 모두 만족해야 한다.

1. $n=12{,}000$에서 단일 값을 반환한 replicate의 mean absolute error가 $0.10$ 이하이다.
2. 같은 replicate에서 PROBE mean absolute error가 $X$-only와 raw $(X,W)$ 각각의 절반 이하이다.
3. 전체 50회 중 단일 값 반환 비율이 $0.95$ 이상이다.
4. 위 세 paired interval 기준을 함께 통과한다.

1--2의 conditional error만 좋아도 3을 만족하지 못하면 실패다. 여러 값을 반환하거나 no-return인
replicate는 임의의 값으로 대체하지 않고 Section 11의 unconditional/set-valued 평가에 포함한다.

## 2. 공통 structural model

### 2.1 고정 계수 생성

공통 $20$차원 계수는 현재 s9의 생성 규칙을 그대로 사용한다.

```python
ROOT_SEED = 190909
coef_key = (ROOT_SEED, 1)  # seed_key("coef")
g = np.random.default_rng(np.random.SeedSequence(coef_key))

def unit(v):
    return v / np.linalg.norm(v)

b_x  = unit(g.standard_normal(20))
a_x  = unit(g.standard_normal(20))
y_x  = unit(g.standard_normal(20))
yt_x = unit(g.standard_normal(20))
b_w  = 2.0 * g.choice(np.array([-1.0, 1.0]), size=10) \
           * g.uniform(0.35, 1.0, size=10)
b_good = b_w[2:]  # 아래의 고정된 8개 값
```

향후 구현은 다음 수치가 생성되는지 assert한다.

```text
b_good = [
  1.1489038010649661,  1.7705451453361223,
 -1.4216890020446680, -0.7855101753703675,
 -1.5159414732646240, -1.6443659023490418,
  1.3794377203214340, -1.2844937715884432
]
```

$b_x,a_x,y_x,yt_x$는 위 code로 생성한 full-precision array를 결과 metadata에 저장한다.
반올림한 문서 수치를 코드의 authority로 사용하지 않는다. 네 벡터의 norm은 각각 $1$이어야 한다.

### 2.2 공통 외생변수

unit $i$마다 다음 변수를 서로 독립으로 생성한다.

$$
U_i,\epsilon_{C,i},I_{1i},I_{2i},\epsilon_{Y,i}\overset{\mathrm{iid}}\sim N(0,1),
$$

$$
\epsilon_{X,i}\sim N(0,I_{20}).
$$

각 proxy noise는 unit, block, coordinate에 걸쳐 독립이며 위 변수와도 독립이다.
관측 기저 중증도는

$$
C_i=0.5U_i+\sqrt{0.75}\epsilon_{C,i}.
$$

따라서 $\operatorname{Var}(C)=1$이고 $\operatorname{Corr}(U,C)=0.5$다.

공변량과 공통 outcome surface는

$$
X_i=b_xU_i+\epsilon_{X,i},
$$

$$
g(X_i)=0.7y_x^\top X_i+0.3(yt_x^\top X_i)^2
$$

이다.

### 2.3 A, C, D, E의 처치와 결과

DGP A, C, D, E는 같은 처치와 잠재결과를 쓴다.

$$
e_i=0.05+0.9\operatorname{expit}
\{1.5U_i+0.6a_x^\top X_i+2(I_{1i}+I_{2i})+0.6C_i\},
$$

$$
A_i\sim\operatorname{Bernoulli}(e_i),
$$

$$
Y_i(a)=a+g(X_i)-4U_i-0.8C_i+\epsilon_{Y,i},
$$

$$
Y_i=Y_i(A_i).
$$

$0.05$와 $0.95$는 structural propensity bounds다. $C$는 처치 확률을 높이고 outcome을 낮추는
pre-treatment cause다. 이 부호는 $X$-only Simpson reversal을 강화하려는 고정 설계다.

### 2.4 $W_0$의 공통 의미

A, B, C, E에서

$$
W_{0i}=(I_{1i},I_{2i},C_i).
$$

$I_1,I_2$는 $A$의 직접 원인이지만 $Y(a)$의 직접 원인은 아니다. $C$는 $U$와 상관되어 $U$의 정보도
가지며, 동시에 $A$와 $Y(a)$의 공통 원인으로 관측된다. 따라서 $W_0$를 포함한 held-out set을 알고 제거하는 것은 privileged diagnostic일
뿐 주 알고리즘에는 허용하지 않는다.

D에서는 같은 latent payload를 image로 렌더링한다. Section 6의 조건부 channel 문장은 관측된
$C\in V$에 조건화하여 읽는다. 숨은 변수의 정의를 임의로 바꾸는 것이 필요조건은 아니다.

### 2.5 oracle과 보조 조정집합

- 주 oracle: $(X,U,C)$.
- raw: $(X,W_0,\ldots,W_4)$.
- $X$ only: $X$.
- 관측 중증도 진단: $(X,C)$.
- privileged without-$I$: $(X,C,W_1,\ldots,W_4)$.
- privileged known-valid PROBE: 후보 생성 전에 $W_0$ 포함 held-out set을 제거한 진단.

마지막 두 방법은 mechanism diagnostic이며 주 비교군이 아니다.

## 3. DGP A: 선형 Gaussian blocks

### 3.1 block 생성

$j=1,\ldots,4$에 대해 $b_{j1}=b_{2j-1}^{\mathrm{good}}$,
$b_{j2}=b_{2j}^{\mathrm{good}}$로 둔다. 서로 독립인 표준정규 noise로

$$
W_{ji}=\left(
b_{j1}U_i+\epsilon_{ji1},\;
b_{j2}U_i+\epsilon_{ji2},\;
C_i+0.35\epsilon_{ji3}
\right).
$$

모든 good block은 $U$의 서로 다른 두 noisy measurement와 $C$의 noisy replicate를 가진다.

### 3.2 역할

A는 기존 성공 s9와 가장 가까운 transfer case다. 차이는 $C$가 $A,Y$를 직접 움직이고 각 good
block이 $C$도 측정한다는 점이다. 따라서 과거 s9 결과는 이 DGP의 결과가 아니다.

### 3.3 이론·실험 지위

conditional mean map $(u,c)\mapsto(b_{j1}u,b_{j2}u,c)$는 rank 2다. 여기서는 관측
$C\in V$가 channel에 들어가므로 Proposition 1의 $X$만으로 index된 specialization을 문구 그대로
적용하지 않는다. 별도 general-ORC 논리는 다음과 같다: raw held-out Gaussian law가 같으면
deconvolution과 rank-2 mean map으로 $(U,C)\mid X$의 law가 같고, 따라서
$\mu_a(X,U,C)$의 평균도 같다. 이는 raw held-out channel에 관한 논리이며 neural embedding의
injectivity나 exact balance 학습을 보장하지 않는다.

### 3.4 사전 기대

- Simpson reversal: 높은 신뢰.
- raw bias amplification: 높은 신뢰지만 새 $C$ 경로 때문에 재검증 필요.
- PROBE가 raw보다 낫다는 기대: 다섯 DGP 중 가장 높음.
- PROBE가 oracle 기준을 통과한다는 기대: 중간.

## 4. DGP B: polynomial continuous blocks

### 4.1 orthonormal polynomial

$$
h_2(u)=\frac{u^2-1}{\sqrt 2},\qquad
h_3(u)=\frac{u^3-3u}{\sqrt 6}.
$$

고정 block 계수는

$$
\beta=(0.15,0.25,0.35,0.45),\qquad
\delta=(-0.6,-0.2,0.2,0.6)
$$

이다.

### 4.2 block 생성

$j=1,\ldots,4$에 대해

$$
W_{ji1}=1.4\frac{U_i+\beta_jh_3(U_i)}{\sqrt{1+\beta_j^2}}+\epsilon_{ji1},
$$

$$
W_{ji2}=1.4\frac{h_2(U_i)+\delta_jU_i}{\sqrt{1+\delta_j^2}}+\epsilon_{ji2},
$$

$$
W_{ji3}=C_i+0.35\epsilon_{ji3}.
$$

noise는 모두 독립 표준정규다.

### 4.3 B 전용 처치와 결과

$$
e_i=0.05+0.9\operatorname{expit}
\{1.4U_i+0.5h_2(U_i)+0.6a_x^\top X_i+2(I_{1i}+I_{2i})+0.6C_i\},
$$

$$
Y_i(a)=a+g(X_i)-3.5U_i-0.8h_2(U_i)-0.8C_i+\epsilon_{Y,i}.
$$

### 4.4 이론·실험 지위

첫 conditional mean의 derivative에서

$$
1+\beta_j\frac{3u^2-3}{\sqrt 6}
$$

의 하한은 $1-3\beta_j/\sqrt 6>0$이다. 따라서 첫 coordinate mean은 $u$에 대해 엄격히
증가하며 $C$ coordinate와 함께 $(U,C)$를 분리한다. 적용할 ORC는 A와 같은 conditional
deconvolution 논리이고 neural embedding으로 자동 이전되지 않는다. 두 번째 coordinate는 nonlinear
decoding 난도를 더한다.

### 4.5 사전 기대

- Simpson reversal: 중간 이상.
- ORC의 구조적 점검: 높음.
- neural encoder의 polynomial 학습 성공: 불확실.
- raw 대비 우월성과 oracle 근접성: 불확실.

## 5. DGP C: mixed Poisson count blocks

### 5.1 rate map

$$
\lambda=(0.35,-0.35,0.50,-0.50),
$$

$$
\kappa=(0.50,0.50,-0.35,-0.35).
$$

$j=1,\ldots,4$와 $r=1,\ldots,4$에 대해 조건부 독립으로

$$
W_{jir}\mid U_i,C_i\sim
\operatorname{Poisson}\left[4\exp\{\lambda_rU_i+\kappa_rC_i\}\right].
$$

block과 coordinate는 $(U,C)$가 주어졌을 때 모두 독립이다. count clipping이나 library-size
normalization을 생성 단계에 넣지 않는다.

### 5.2 전처리

encoder와 nuisance input에는 training split에서 계산한

$$
\widetilde W_{jir}=\frac{\log(1+W_{jir})-\widehat\mu_{jr}}
{\max(\widehat\sigma_{jr},10^{-6})}
$$

를 사용한다. raw baseline도 같은 변환을 사용한다. 원 count는 audit와 provenance를 위해 저장한다.

### 5.3 이론·실험 지위

$4\times2$ loading matrix $[\lambda\;\kappa]$는 rank $2$다. Poisson probability-generating-function의
유일성과 rank-2 log-rate map을 쓰면 raw held-out law의 동일성이 $(U,C)\mid X$ law의 동일성을
함의하고, 따라서 outcome mean도 같다. 이는 structural general-ORC check이며 finite-sample neural
optimization과 screen 통과를 보장하지 않는다.

### 5.4 사전 기대

- Simpson reversal: 높은 신뢰.
- count rate-map 구조 검증: 높은 신뢰.
- encoder와 critic 최적화: 중간 이하.
- raw 대비 우월성·oracle 근접성: 불확실.

## 6. DGP D: 32x32 합성 multimodal image blocks

### 6.1 block별 세 template

pixel 중심은 $p,q\in\{0,\ldots,31\}$에 대해

$$
x_p=2(p+0.5)/32-1,\qquad y_q=2(q+0.5)/32-1
$$

로 둔다. good block $j=1,\ldots,4$에서 $\alpha_j=j\pi/10$이고 중심은

$$
c_{j1}=0.35(\cos\alpha_j,\sin\alpha_j),\quad
c_{j2}=-c_{j1},\quad
c_{j3}=0.35(-\sin\alpha_j,\cos\alpha_j)
$$

이다. width $s_T=0.18$로

$$
G_{jr}(x)=\exp\{-\|x-c_{jr}\|^2/(2s_T^2)\},\qquad r=1,2,3
$$

를 만든다. 각 block 안에서 $G_{j1},G_{j2},G_{j3}$ 순서로 modified Gram--Schmidt를 적용하고
Euclidean norm $1$인 $T_{j1},T_{j2},T_{j3}$를 얻는다. 구현은
$|T_{jr}^\top T_{js}-1(r=s)|<10^{-10}$을 assert한다.

### 6.2 latent payload와 렌더링

good block $j$의 payload는

$$
H_{ji}=\left(b_{j1}U_i+\epsilon_{ji1},
b_{j2}U_i+\epsilon_{ji2},C_i+0.35\epsilon_{ji3}\right).
$$

관측 good-block image는

$$
W_{ji}=0.7\sum_{r=1}^3H_{jir}T_{jr}
+0.1E_{ji},
$$

여기서 $E_{ji}$의 $1024$ pixel은 독립 $N(0,1)$이다. $W_0=(I_1,I_2,C)$는 image로 렌더링하지
않는 numeric block이다. pixel clipping, uint8 변환, contrast normalization은 하지 않고 float32를
저장한다. 따라서 D의 관측 $W$ 차원은 $3+4\times1024=4099$이고, 별도로 $X$ 20차원이 있다.

### 6.3 영상 encoder

각 block에 D/E 공통 CNN을 동일 weight로 적용한다.

1. Conv$(1,16,3)$, padding 1, ReLU, $2\times2$ max pool.
2. Conv$(16,32,3)$, padding 1, ReLU, $2\times2$ max pool.
3. Conv$(32,64,3)$, padding 1, ReLU.
4. adaptive average pool to $4\times4$.
5. flatten, Linear$(1024,16)$, LayerNorm, ReLU.

block embeddings와 standardized $X$를 concatenate한 뒤 Section 8의 representation head에 넣는다.

### 6.4 이론·실험 지위

raw image 자체의 orthonormal matched-filter projection은 payload에 대한 injective Gaussian location
channel을 보존한다. 그러나 learned CNN embedding이 injectivity나 completeness를 자동으로 보존하지 않는다.
주 분석은 알려진 projection을 사용하지 않고 raw image를 learned CNN으로 처리한다. matched-filter arm은
Section 9.4의 privileged diagnostic일 뿐이다.

### 6.5 사전 기대

- structural channel check: 높음.
- CNN이 세 template score를 학습: 중간.
- PROBE가 raw CNN보다 우월: 낮음.
- oracle 근접성: 낮음에서 중간.

## 7. DGP E: MNIST image와 numeric $C$ replicate

### 7.1 latent class

이 DGP에서

$$
K_i\sim\operatorname{Unif}\{0,\ldots,9\},
$$

$$
U_i=\frac{K_i-4.5}{\sqrt{8.25}}.
$$

이 표준화로 $\mathbb E(U)=0$, $\operatorname{Var}(U)=1$이다. $C,X,A,Y$는 Section 2의
공통식을 사용한다.

### 7.2 fixed MNIST pool

1. [공식 TensorFlow Datasets MNIST catalog](https://www.tensorflow.org/datasets/catalog/mnist)의
   training split에서 label별 1,000장, 총 10,000장을 한 번만 뽑는다. 해당 catalog의 기준 schema는
   train 60,000, test 10,000, image $(28,28,1)$ uint8, class 10개다.
2. pool seed는 `(190909, 10)`이다.
3. label별 원래 dataset index를 오름차순으로 놓고 seeded permutation의 처음 1,000개를 택한다.
4. 원 이미지를 float32 $[0,1]$로만 변환한다. 회전, occlusion, pixel noise, 증강은 넣지 않는다.
5. project manifest의 원자료 SHA-256
   `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`
   (11,490,434 bytes), 선택 index와 pool digest를 provenance에 저장한다.
6. implementation 전에 서로 다른 class 사이의 exact pixel duplicate를 검사한다. 발견되면 개수와
   index를 보고하고 그 상태로 confirmatory claim을 시작하지 않는다.

이 10,000장은 image source pool이지 분석 unit sample이 아니다. 각 분석 unit은 아래 규칙으로 pool에서
독립적으로 image를 뽑은 뒤 unit 단위로 D/N/E에 배정한다. 같은 source image의 재사용은 허용하되,
동일 unit이 둘 이상의 split에 나타나는 일은 없다.

### 7.3 corrupted label과 block

각 $j=1,\ldots,4$에서 독립적으로

$$
\widetilde K_{ji}=\begin{cases}
K_i,&\text{확률 }0.75,\\
\text{나머지 9개 class 중 균등 draw},&\text{확률 }0.25.
\end{cases}
$$

그 다음 label $\widetilde K_{ji}$의 fixed pool에서 image $M_{ji}$를 균등 복원추출한다. numeric channel은

$$
R_{ji}=C_i+0.35\epsilon_{jiC}
$$

이고, 관측 block은 multimodal tuple

$$
W_{ji}=(M_{ji},R_{ji})
$$

이다. $R$를 image pixel에 append하거나 image에 직접 그리지 않는다. CNN embedding과 standardized
$R$을 representation head 직전에 concatenate한다. 알려진 digit label이나 pretrained classifier는
주 방법에 제공하지 않는다.

$W_0=(I_1,I_2,C)$는 numeric block으로 유지한다.

### 7.4 encoder

MNIST도 Section 6.3의 Conv$(16,32,64)$--adaptive-pool--16 architecture를 그대로 쓴다. feature extractor는
외부 label supervision이나 MNIST test split을 사용하지 않고 현재 causal objective 안에서 end-to-end로
학습한다.

### 7.5 이론·실험 지위

true corrupted class를 관측한 finite-state channel matrix는 $q=0.25$에서 full rank인지 수치적으로
확인할 수 있다. 실제 image channel의 linear independence는 fixed pool의 class-conditional empirical
pixel-distribution matrix에 관한 별도 조건이다. CNN embedding에 그 rank가 자동으로 상속되지 않는다.
같은 pool에서 복원추출하므로 주 E는 iid unit을 갖는 transductive finite-pool benchmark이지 unseen-image
generalization benchmark가 아니다. label별 source pool을 fold 사이에 분리한 분석을 robustness로 둔다.

### 7.6 사전 기대

- finite-class source channel: 중간 이상.
- end-to-end CNN이 필요한 representation을 학습: 낮음.
- raw CNN이 PROBE보다 나을 가능성: 큼.
- 따라서 E는 성공 시 확장성 증거, 실패 시 encoder/정보 병목 진단이다.

## 8. 주 방법: PROBE-Brier-neural

### 8.1 후보 held-out set

$J=5$이므로 주 분석은

$$
\mathcal S=\{S:\varnothing\ne S\subsetneq\{0,1,2,3,4\}\}
$$

의 $2^5-2=30$개 set을 전수평가한다. $W_0$의 정체를 알고 후보를 제거하지 않는다.
이 설계에서는 $W_0$ 미포함 set 15개와 포함 set 15개가 있다는 구조적 사실만으로 target plurality가
성립하지 않는다. learned population candidate, screen 통과 수와 cluster separation을 직접 확인한다.

### 8.2 unit-level split과 세 rotation

replicate마다 unit index를 한 번 permutation하여 크기가 가능한 한 같은 세 fold $F_0,F_1,F_2$로 나눈다.
rotation $k=0,1,2$에서

$$
D=F_k,\qquad N=F_{(k+1)\bmod3},\qquad E=F_{(k+2)\bmod3}
$$

로 둔다. standardization, image encoder fitting, representation selection, discrepancy screen은 $D$에서만,
nuisance와 overlap screen은 $N$에서만, AIPW summand 평가는 $E$에서만 한다.

$E$의 자료는 representation, early stopping, discrepancy/overlap threshold와 screen에 쓰지 않는다.
다만 screen 뒤 $E$에서 얻은 AIPW 후보값은 Algorithm 1이 정한 linking과 component aggregation에
사용한다. 이는 사후 tuning이 아니라 사전 고정된 평가 단계다.

각 rotation은 자기 $D,N$에 조건부로 E score를 평가하므로 per-rotation honesty를 갖는다. 하지만 한
rotation의 E unit이 다른 rotation에서는 D/N에 들어가므로 세 rotation 평균에 manuscript의 단일
global conditioning argument를 그대로 적용할 수 없다. 세 rotation 결합은 empirical cross-fitting
extension으로 보고한다. 별도로 rotation 0 하나만 사용한 theory-aligned diagnostic을 낸다.

### 8.3 D 내부 독립 audit

D를 seeded permutation으로 $50\%/25\%/25\%$의
$D_{\mathrm{fit}},D_{\mathrm{val}},D_{\mathrm{audit}}$로 나눈다.

- $D_{\mathrm{fit}}$: encoder와 inner critics 학습.
- $D_{\mathrm{val}}$: early stopping과 restart 선택.
- $D_{\mathrm{audit}}$: 최종 Brier risk gap과 그 표준오차 계산.

이는 manuscript Section 4의 original empirical-ERM을 구현하기 위한 하나의 operational
independent-audit variant다. manuscript의 finite-class uniform bound를 neural network에 증명한 것으로
해석하지 않는다.

### 8.4 representation

후보 $S$마다

$$
V_S=(X,W_{-S}),\qquad T_S=(X,W_S),\qquad Z_S=\phi_S(V_S)\in\mathbb R^4.
$$

tabular A--C에서는 block마다 parameter가 분리된 MLP$(d_j,64,64,16)$를 쓰되 architecture는 같게 한다.
numeric $W_0$에도 같은 두-hidden-layer form을 쓰되 input dimension만 3이다. D--E는 Section 6--7의
weight-shared CNN으로 각 image를
16차원에 encode하고 numeric channel은 MLP로 encode한다. available block embeddings와 standardized
$X$를 concatenate한 뒤 fusion MLP$(\cdot,64,32,4)$로 $Z$를 만든다. block order는 고정하고 missing
held-out block은 concatenate하지 않는다.

### 8.5 nested Brier critics

base critic은 $q_{0,S}(Z)\in[0,1]$, augmented critic은 $q_{1,S}(Z,T)\in[0,1]$이다.
둘 다 width $(128,64)$ ReLU MLP와 sigmoid output을 쓴다. augmented class는 base network의 모든
parameter와 $T$ residual branch를 포함하며 residual branch weight를 $0$으로 두면 base function과
동일하도록 구현한다.

Brier loss는

$$
\ell(q;A)=(A-q)^2.
$$

population discrepancy에 대응하는 operational estimate는

$$
\widehat D_S^2=\widehat R_{0,S}-\widehat R_{1,S}
$$

이고 두 risk는 독립 $D_{\mathrm{audit}}$에서 같은 row별 loss difference로 평가한다.
sampling fluctuation으로 음수가 나오면 보고에는 원값을 저장하고 screen 계산에만
$\max(0,\widehat D_S^2)$를 사용한다.

### 8.6 alternating optimization

각 restart에서 다음을 고정한다.

- optimizer: AdamW.
- encoder learning rate $10^{-3}$, critic learning rate $2\times10^{-3}$.
- weight decay $10^{-4}$.
- batch size 256.
- critic update 5회당 encoder update 1회.
- 최대 300 epochs.
- gradient norm clipping 5.
- $D_{\mathrm{val}}$ objective가 20 epochs 개선되지 않으면 정지.
- 개선 최소량 $10^{-5}$.
- restart 3개.

critic은 Brier loss를 최소화한다. 주 encoder는 manuscript empirical-ERM에 맞추어 오직

$$
\widehat D_{\mathrm{fit},S}^2
$$

를 최소화한다. treatment-prediction과 overlap regularization의 주 계수는 모두 $0$이다. 별도 ablation은
$0.05\widehat L_{\mathrm{treat}}+10\widehat P_{\mathrm{overlap}}$을 더할 수 있으나 `non-ERM weighted-loss`
arm으로 명시한다. 그 ablation 결과를 main PROBE-Brier 결과와 합치지 않는다.

restart는 $D_{\mathrm{val}}$의 raw, unclipped risk gap이 가장 작고 overlap violation이 2% 이하인 것을
선택한다. 해당 restart가 없으면 gap 최소 restart를 보존하되 overlap screen에서 탈락시킨다.

### 8.7 discrepancy screen

$D_{\mathrm{audit}}$의 row loss difference 표준오차를 $\widehat s_{D,S}$라 한다. operational tolerance는

$$
t_n=0.02+2\widehat s_{D,S}.
$$

후보는

$$
\max(0,\widehat D_S^2)\le t_n
$$

일 때 discrepancy screen을 통과한다. 상수 $0.02$는 pilot에서 동결할 calibration parameter이며
causal validity 검정의 유의수준이 아니다. confirmatory 결과에는 structural-valid set 대비 TPR/FPR,
$\widehat D_S^2$, $\widehat s_{D,S}$와 threshold margin을 모두 저장한다.

### 8.8 overlap screen

각 surviving $S$에 대해 $N$만 사용해 Section 8.9의 propensity learner를 fit한다. $N$의 fitted
propensity 가운데 적어도 98%가 $[0.05,0.95]$에 있을 때 통과한다. 이 판정은 E를 보지 않는다.
평가 시 propensity는 $[0.02,0.98]$로 clip하되 clipping 비율을 보고한다.

### 8.9 nuisance와 AIPW

각 후보 $S$에서 $Z_S$ mapping은 D에서 고정한다. N의 80% fit/20% early-stop split에서 다음을 fit한다.

- propensity head: MLP$(d,128,64,1)$, ReLU, sigmoid, binary cross-entropy.
- outcome head: 처치군별 MLP$(d,128,64,1)$, ReLU, squared loss.
- optimizer: AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, batch 256,
  maximum 500 epochs, patience 20, minimum improvement $10^{-5}$, restart 3.

raw, $X$-only, $(X,C)$와 oracle도 같은 head와 optimizer를 쓴다. A--C raw의 $d$는 standardized raw
input이다. D--E raw는 각 propensity/outcome model이 N에서 자기 CNN과 fusion head를 end-to-end로
학습한다. PROBE와 raw의 CNN architecture, restart와 epoch cap은 같고, 선택은 N validation loss로만 한다.

E unit $i$의 score는

$$
\widehat\psi_{i,S}=\widehat\mu_1(Z_i)-\widehat\mu_0(Z_i)
+\frac{A_i\{Y_i-\widehat\mu_1(Z_i)\}}{\widehat e(Z_i)}
-\frac{(1-A_i)\{Y_i-\widehat\mu_0(Z_i)\}}{1-\widehat e(Z_i)}.
$$

$$
\widehat\theta_{S,k}=|E|^{-1}\sum_{i\in E}\widehat\psi_{i,S}.
$$

각 $S$의 rotation별 discrepancy와 overlap pass mask를 저장한다. 주 cross-fit arm에서는 세 rotation을
모두 통과한 $S$만 retained candidate로 둔다. retained $S$의 세 E-fold estimate를 unit score 평균으로
pool하여 $\widehat\theta_S$를 만든 뒤 한 번 aggregation한다. 이 intersection rule과 pooling은 empirical
cross-fitting extension이며 manuscript의 원래 단일-split theorem을 그대로 적용하지 않는다. rotation 0
하나만 D/N/E로 사용하는 theory-aligned diagnostic을 별도로 낸다.

### 8.10 linking과 aggregation

retained candidates 전체에 5,000회 Gaussian multiplier bootstrap을 한다. draw $b$에서 unit별
$g_i^{(b)}\overset{\mathrm{iid}}\sim N(0,1)$을 후보 전체에 공통 적용하고

$$
M^{(b)}=\max_{S}\left|n^{-1}\sum_{i=1}^n
g_i^{(b)}\{\widehat\psi_{i,S}-\widehat\theta_S\}\right|
$$

를 계산한다. 하나의 공통 radius는 $\rho=\operatorname{quantile}_{0.95}
\{M^{(1)},\ldots,M^{(5000)}\}$로 둔다.
후보 $S,S'$는

$$
|\widehat\theta_S-\widehat\theta_{S'}|\le 2\rho
$$

이면 연결한다.

가장 큰 connected component가 하나면 그 component estimate의 median을 반환한다. component 크기가
짝수면 중앙 두 값의 산술평균을 median으로 정의한다.

같은 최대 크기의 component가 여러 개면 tie-break하지 않는다. 각 최대 component median을 모두
candidate set으로 반환한다. 이는 non-single-output이다. survivor가 0이면 no-return, 하나면 singleton
component 반환과 `single_survivor=true`를 기록한다.

approximate-valid candidate의 구조 bias를 자동으로 $\rho$가 덮는다고 주장하지 않는다. 별도로
$b\in\{0,0.025,0.05,0.10\}$을 더한 linking sensitivity를 보고한다.

## 9. 비교 방법과 공정성

### 9.1 주 비교군

1. Naive difference in means.
2. Honest AIPW on $X$.
3. Honest AIPW on $(X,C)$.
4. Honest AIPW on raw $(X,W)$.
5. Oracle AIPW on $(X,U,C)$.
6. PROBE-Brier-neural.
7. PROBE-RFF reference.

모든 AIPW는 같은 D/N/E rotation, clipping, propensity와 outcome learner를 쓴다. adjustment input만
달라진다.

### 9.2 영상 비교의 compute parity

D와 E의 raw baseline도 PROBE와 동일한 CNN architecture, initialization namespace, optimizer,
restart 수, epoch cap과 early-stopping patience를 사용한다. raw baseline은 nuisance prediction loss로
end-to-end 학습한다. PCA-only raw baseline을 주 비교군으로 쓰지 않는다.

PROBE가 subset별 CNN을 여러 번 학습하므로 두 가지 예산을 함께 보고한다.

- architecture parity: 후보 하나와 raw의 architecture가 동일.
- total-compute parity sensitivity: PROBE의 총 gradient-step 예산만큼 raw에 restart를 허용.

### 9.3 PROBE-RFF reference arm

현재 `code/probe_e1_pipeline.py`의 RFF ridge critic, noise-control subtraction, 기존 screen과 aggregation을
사용하는 arm이다. 이는 과거 s9 recipe와의 연결을 위한 reference이며 manuscript Brier neural main과
같은 알고리즘이 아니다. 과거의 절대오차 $0.055$는 새 Brier critic의 검증 evidence가 아니다.

### 9.4 privileged diagnostics

- known-valid: $W_0$ 포함 held-out set을 사전에 제거.
- without-$I$: raw adjustment에서 $I_1,I_2$ 제거.
- matched-filter D: CNN 대신 $(\langle W,T_1\rangle,\langle W,T_2\rangle,
  \langle W,T_3\rangle)$ 사용.
- class-oracle E: MNIST image 대신 실제 corrupted class $\widetilde K$ 사용.
- population raw functional: 가능한 A--C/E에서 numerical integration/대규모 MC.

이들은 원인 분해용이며 main winner 표에 섞지 않는다.

## 10. pilot와 confirmatory 실행

### 10.1 표본 크기와 반복

고정 grid는

$$
n\in\{3{,}000,6{,}000,12{,}000,24{,}000\}.
$$

각 DGP와 $n$에서 50 independent replicate를 실행한다. primary endpoint는 결과를 보기 전에 고정한
$n=12{,}000$이다. 나머지는 수렴 곡선이며 더 좋아 보이는 $n$을 사후 primary로 고르지 않는다.

### 10.2 pilot

$n\in\{6{,}000,12{,}000\}$ 각각에서 confirmatory seed와 겹치지 않는 5 replicate를 실행한다.
pilot replicate id는 10000--10004, confirmatory id는 0--49로 고정하고 seed tuple에
`phase=pilot/confirmatory` namespace도 넣는다. pilot에서는 다음만 점검한다.

1. 생성 moment, propensity bounds와 Simpson sign.
2. NaN, exploding gradient, empty arm.
3. Brier nested critic의 train/val/audit gap.
4. fixed tolerance $0.02$의 null-control screen rate.
5. memory와 runtime.

pilot에서 바꾼 값은 날짜가 찍힌 protocol amendment에 기록하고 confirmatory 전체를 새 seed로 시작한다.
pilot 결과를 confirmatory 50회에 합치지 않는다.

### 10.3 confirmatory workload

restart와 inner critic update 전 최소 subset-rotation fit 수는

$$
5\ \mathrm{DGP}\times4\ n\times50\ \mathrm{rep}
\times30\ S\times3\ \mathrm{rot}=90{,}000.
$$

따라서 실행시간을 미리 몇 시간이라고 약속하지 않는다. pilot에서 unit fit time과 peak memory를 측정해
전체 compute budget을 별도 산출한다.

### 10.4 $m$ ablation

main은 30개 전수 후보다. 별도 ablation에서 $m\in\{5,10,20\}$개를 30개 중 비복원 균등추출한다.
subset draw seed를 저장하며 known-valid 정보를 쓰지 않는다.

## 11. 불확실성과 다중성

### 11.1 replicate-level paired interval

DGP 5개와 Section 1.3의 contrast 3개, 총 15개 one-sided 검정을 한 family로 둔다. $R$개의
단일반환 replicate에서 contrast 평균과 표준오차가 $\bar G,\operatorname{SE}(\bar G)$이면 주 upper bound는

$$
U_{0.95}^{\mathrm{Bonf}}
=\bar G+t_{1-0.05/15,R-1}\operatorname{SE}(\bar G)
$$

이다. suite는 15개 bound가 모두 $0$보다 작아야 한다. unadjusted bound와 Studentized paired bootstrap
10,000회 결과는 robustness로 저장하되 주 판정을 바꾸지 않는다.

### 11.2 unconditional 반환 성능

DGP별로 다음을 반드시 함께 보고한다.

- no-return rate.
- singleton-survivor rate.
- unique-largest-component 단일반환 rate.
- tied-largest-component set 반환 rate.
- per-replicate acceptable-return rate: 단일반환이고 $|\widehat\tau-1|\le0.10$인 비율.
- conditional absolute error: 단일반환 replicate에 한정하되 조건부임을 명시.

no-return이나 set-valued return을 제외한 conditional 평균만으로 성공을 주장하지 않는다. paired
confidence bound는 replicate 평균의 suite-level 판정이며 개별 replicate event에 소급 적용하지 않는다.

### 11.3 set-valued 평가

tie가 난 replicate에서는 component median set $\widehat\Theta$에 대해

$$
d(1,\widehat\Theta)=\min_{t\in\widehat\Theta}|t-1|,
$$

set diameter와 $1\in\operatorname{conv}(\widehat\Theta)$ 여부를 보고한다. 이를 point-estimate 성공으로
바꾸지 않는다.

### 11.4 coverage

selection 뒤 개별 Wald interval을 그대로 최종 coverage로 부르지 않는다. 주 simultaneous radius는
Section 8.10의 shared-E multiplier bootstrap을 사용한다. 전체 algorithmic selection까지 포함한
coverage는 replicate bootstrap 또는 독립 outer Monte Carlo에서 별도로 평가한다.

## 12. failure controls와 진단

### 12.1 구조·정보 control

1. $C$ outcome path 제거: $-0.8C$를 $0$으로 바꾼다.
2. $C$ treatment path 제거: $+0.6C$를 $0$으로 바꾼다.
3. $C$ replicate 제거는 DGP별로 정의한다: A/B는 세 번째 coordinate를 독립 noise로 바꾸고,
   C는 모든 $\kappa_r=0$으로 두며, D는 $H_3$의 $C$ 의존성만 제거하고 같은 noise를 유지하며,
   E는 $R_j$를 독립 noise로 바꾼다.
4. instrument 제거: $2(I_1+I_2)$를 $0$으로 바꾼다.
5. proxy null: good block의 $U,C$ loading을 $0$으로 바꾼다.

이 중 $C$ outcome ablation은 해당 mechanism의 기여를 보는 진단이지 일반적 falsification test가 아니다.

### 12.2 학습 control

1. true null은 fit/validation/audit 각각에서 $A$를 모든 predictor와 독립적으로 permutation하여
   audit gap의 null 동작을 확인한다.
2. held-out $W$ row permutation은 $W$의 추가 기여를 보는 diagnostic이다. $T$에 $X$도 들어가므로
   augmented advantage가 반드시 0이 된다고 주장하지 않는다.
3. D image matched-filter vs CNN.
4. E corrupted-class oracle vs CNN.
5. encoder output permutation before nuisance fit.
6. optimizer restart dispersion.

screen 통과를 causal validity test라고 부르지 않는다. structural-valid 후보를 simulation에서 알고 있을 때만
TPR/FPR를 진단적으로 계산한다.

### 12.3 aggregation boundary

30개 후보 중 구조적으로 valid/invalid 후보 수가 15:15인 사실은 target plurality를 보장하지 않는다.
각 DGP와 $n$에서 다음 population-oriented quantity를 별도로 계산한다.

- 각 $S$의 large-$n$ learned candidate $\theta_S$.
- population/audit discrepancy proxy.
- target band 안 multiplicity.
- 최대 invalid component 크기.
- valid와 invalid component separation.

이 결과가 실패하면 DGP는 theorem success example이 아니라 aggregation boundary stress test로 보고한다.

### 12.4 강한 raw comparator 맥락

기존 s10, s11과 이전 MNIST pilot에서 강한 raw learner가 PROBE를 이긴 경험은 learner choice가 결론을
바꿀 수 있음을 보여 주는 project context다. 새 A--E가 같은 순위를 보인다는 evidence는 아니다.
따라서 weak tree나 PCA baseline만으로 raw 열세를 주장하지 않는다.

## 13. 예상 순위와 confidence

다음은 결과가 아니라 사전 hypothesis다.

아래 부등호는 **절대오차가 작을 것으로 기대되는 순서**다. 중괄호는 순서가 불확실한 묶음이다.

| DGP | 사전 기대 ordering | confidence | 핵심 위험 |
|---|---|---|---|
| A Gaussian | oracle $<$ PROBE $<$ raw $<$ $(X,C)$ $<$ $X$ $<$ naive | PROBE/raw는 중간, 나머지는 높음 | direct $C$ 때문에 과거 s9 전이가 깨질 수 있음 |
| B polynomial | oracle $<$ {PROBE, raw} $<$ $(X,C)$ $<$ {$X$, naive} | 낮음 | nonlinear encoder/critic optimization |
| C Poisson | oracle $<$ {PROBE, raw} $<$ $(X,C)$ $<$ {$X$, naive} | 낮음--중간 | count scaling과 finite-sample rate 정보 |
| D synthetic image | oracle $<$ {raw, PROBE} $<$ $(X,C)$ $<$ {$X$, naive} | 낮음; raw를 약간 선호 | CNN이 raw-channel 정보를 버릴 수 있음 |
| E MNIST | oracle $<$ {raw, PROBE} $<$ $(X,C)$ $<$ {$X$, naive} | 가장 낮음; raw를 선호 | raw CNN 승리 또는 corruption 정보 부족 |

대화 중 exploratory 계산은 A에서 Simpson reversal과 raw residual bias 가능성, E에서 $q=0.25$가
$q\le0.20$보다 더 어려운 class-oracle regime일 가능성을 시사했다. 그러나 해당 출력은 저장된 재현
artifact가 아니므로 confirmatory evidence나 성능 보장으로 인용하지 않는다.

## 14. sensitivity와 $\Gamma$

### 14.1 continuous arms

A--D에서 channel injectivity와 유한 stability constant $\Gamma$는 다른 주장이다. Gaussian 또는 image
channel에서 $\Gamma<\infty$가 자동으로 따라오지 않는다. $\Gamma$를 주 성능 설명에 쓰려면 별도 operator
norm 또는 제한된 function class에서 계산해야 한다.

### 14.2 finite-state E의 partial-channel diagnostic

class-oracle E에서는 $10\times10$ corrupted-label channel matrix의 rank, 최소 singular value와 최소 class
probability를 계산한다. 이는 $U$-class 부분의 channel-conditioning diagnostic이다. $C$는 연속 Gaussian
channel이므로 이를 full E-DGP의 $\Gamma$라고 부르지 않는다. full $\Gamma$에는 별도의 outcome-specific
bridge/operator bound가 필요하며 실제 image CNN arm에도 그대로 이전하지 않는다.

### 14.3 approximate linking

$b$-expanded radius 분석은 empirical sensitivity다. Section 4의 안정성 문장이 arbitrary learned neural
bias를 자동으로 덮는다고 해석하지 않는다.

## 15. seed와 재현 규율

### 15.1 namespace

```text
root             190909
coefficient      1
data             2
unit split       5
encoder          6
critic           7
nuisance         8
multiplier       11
MNIST pool       10
subset ablation  12
```

pilot phase id는 0, confirmatory phase id는 1로 고정한다. 일반 실행 seed는
`SeedSequence((root, namespace, phase_id, dgp_id, n, replicate, rotation, subset_code, restart))`로 만든다.
DGP id는 A=1, B=2, C=3, D=4, E=5다. subset code는 5-bit integer다. 공통 계수는 phase와 무관하게
`(root, coefficient)`만 사용하고, 고정 MNIST pool도 phase와 무관하게 `(root, MNIST pool)`만 사용한다.

### 15.2 저장할 metadata

- git commit와 dirty status.
- Python, numpy, scipy, scikit-learn, torch version.
- CPU/GPU와 deterministic-algorithm flag.
- full coefficient arrays.
- source data digest와 MNIST pool indices.
- unit split indices 또는 재생 가능한 seed.
- 모든 hyperparameter와 checkpoint epoch.
- 후보별 gap, SE, overlap, estimate, radius, component id.
- no-return/tie flags.
- runtime와 peak memory.

### 15.3 planned files

아래 이름은 미래 구현 제안이며 아직 존재한다고 가정하지 않는다.

```text
code/probe_dgp_ae.py
code/probe_brier_neural.py
code/run_dgp_ae.py
code/summarize_dgp_ae.py
results/dgp_ae_pilot.json
results/dgp_ae_confirmatory.json
results/dgp_ae_confirmatory_runs.csv
```

## 16. 구현 acceptance checklist

### 16.1 생성 unit tests

- [ ] $\|b_x\|=\|a_x\|=\|y_x\|=\|yt_x\|=1$.
- [ ] $b_{\mathrm{good}}$가 Section 2.1의 여덟 값과 일치.
- [ ] Monte Carlo에서 $\operatorname{Var}(U),\operatorname{Var}(C)\approx1$,
      $\operatorname{Corr}(U,C)\approx0.5$.
- [ ] 모든 DGP에서 empirical $Y(1)-Y(0)=1$.
- [ ] structural propensity가 $[0.05,0.95]$ 안.
- [ ] B의 첫 conditional mean derivative lower bound가 양수.
- [ ] C의 loading matrix rank가 2.
- [ ] D의 세 template가 orthonormal.
- [ ] E의 class frequency, corruption rate와 pool provenance가 맞음.
- [ ] E의 class-conditional empirical pixel-distribution rank 또는 exact-support collision audit 저장.

### 16.2 split·leakage tests

- [ ] D/N/E unit index가 서로소이고 합집합이 전체 sample.
- [ ] 각 unit이 정확히 한 rotation에서 E가 됨.
- [ ] E 접근 없이 screen survivor가 결정되고, component는 E AIPW 값에 사전 규칙만 적용해 결정됨.
- [ ] D standardizer가 N/E에서 refit되지 않음.
- [ ] MNIST source image 재사용 여부와 무관하게 unit split leakage가 없음.

### 16.3 algorithm tests

- [ ] augmented critic의 residual branch zero가 base critic과 동일 output.
- [ ] audit risk는 $D_{\mathrm{audit}}$에서만 계산.
- [ ] 30개 nonempty proper subset 전부 실행.
- [ ] no known-valid pruning.
- [ ] tie이면 모든 max-size component median 반환.
- [ ] no survivor이면 no-return.
- [ ] multiplier가 후보 전체에 unit별로 공유됨.
- [ ] raw와 PROBE nuisance/CNN budget audit 통과.

### 16.4 reporting tests

- [ ] primary $n=12{,}000$가 다른 $n$보다 먼저 판정됨.
- [ ] 50 replicate가 모두 complete일 때만 headline table 생성.
- [ ] 15개 paired contrast의 Bonferroni-$t$ upper bound 저장.
- [ ] unconditional return/failure rate 포함.
- [ ] conditional-only 결과에 명확한 조건 표시.
- [ ] theory-certified, operational, preliminary, empirical result label 분리.

## 17. 중단 규칙과 해석

다음이면 해당 DGP에서 성공 주장을 중단한다.

1. $X$-only가 Simpson sign reversal을 보이지 않는다.
2. oracle AIPW가 평균 절대오차 $0.10$을 넘는다.
3. raw learner가 population diagnostic조차 근사하지 못한다.
4. pilot에서 no-return 또는 tied-set rate가 20%를 넘으면 confirmatory 전에 diagnostic hard stop한다.
5. screen의 structural-valid TPR이 80% 미만이거나 FPR이 20%를 넘는다.
6. D/E에서 matched-filter/class-oracle은 작동하지만 CNN만 실패한다.

confirmatory의 반환 성공 기준은 Section 1.3의 단일반환율 95%이며, 4의 20%는 성공 기준이 아니라 pilot
debug trigger다. 1은 DGP calibration failure, 2--3은 nuisance/implementation failure, 4--5는 search/aggregation failure,
6은 representation-learning failure로 분류한다. 실패한 DGP를 삭제하지 않고 사전 기준에 따라 negative
result 또는 stress test로 보고한다.

## 18. 이 문서가 고정하는 결론의 범위

이 설계가 성공하면 지지하는 것은 다음의 제한된 주장이다.

> 사전 정의된 다섯 synthetic/semi-synthetic 환경에서, unknown contaminated block을 포함한 후보군을
> 전수 탐색하고 독립 audit, honest nuisance fitting과 held-out AIPW를 사용했을 때, PROBE가 raw proxy
> adjustment보다 작은 오차를 보이며 oracle에 가까운 값을 반환할 수 있다.

성공하더라도 모든 clean proxy에서 raw보다 항상 우월함, arbitrary neural network의 consistency,
finite-sample exact causal certification 또는 모든 image distribution의 completeness를 뜻하지 않는다.
실패하면 어느 단계가 원인인지 Section 12와 17의 진단으로 분해하여 보고한다.

## 19. 현재 authority와 연결

- 이론과 Algorithm 1의 현재 authority: `manuscript/main/2.tex`, `manuscript/main/3.tex`,
  `manuscript/main/4.tex`.
- 현재 실험 상태와 과거 s9 recipe: `EXPERIMENTS.md`.
- 기존 RFF reference 구현: `code/probe_e1_pipeline.py`.
- 기존 SCM 계수와 seed 규칙: `code/probe_scms.py`.
- canonical artifact 지위: `README.md`, `ARTIFACTS.md`.

이 문서는 위 자료를 대체하지 않는다. 충돌이 생기면 현재 manuscript Section 2--4의 estimand와
algorithmic contract를 우선하고, operational 차이는 결과 보고서에 명시한다. 사용자 지시에 따라
현재 abstract, Section 1과 Section 5는 이 실험 설계의 authority에서 제외한다.
