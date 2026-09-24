# Shapes3D 원본 이미지 PROBE 실험 PRD v1

작성일: 2026-09-18  
상태: **PRD COMPLETE / IMAGE FEASIBILITY OPEN / 실행 미승인**  
대상 독자: 이 문서를 받아 구현할 COA와 결과를 원고에 사용할 연구자  
적용 범위: `research/papers/single_proxy_balancing/` 안의 향후 Shapes3D 실험

## 0. 이 문서가 승인하는 것과 승인하지 않는 것

이 문서는 구현 명세다. 이 문서의 완성은 DGP, pixel 식별성, CNN 성능이 검증됐다는 뜻이 아니다.

- 현재 별도 실행 승인을 받아 시작할 수 있는 단계는 **P0 feasibility audit뿐**이다.
- P0가 통과하기 전에는 학습 코드, GPU smoke, development, confirmation을 실행하지 않는다.
- P0의 bounded mask menu가 certificate를 만들지 못하면 `NO_CERTIFICATE/STOP`으로 종료한다.
- 그때 결과는 “Shapes3D에서 전역적으로 불가능”이 아니라 “사전 지정한 메뉴에서 certificate를 만들지 못함”이다.
- approximate empirical experiment로 조용히 낮추거나 다른 benchmark로 이동하지 않는다.
- 이 문서는 코드, 결과, 원고, registry, criteria, index 또는 memory를 수정하지 않는다.

## 1. 실제 목표

한 장의 **변형하지 않은 Shapes3D 이미지**를 high-dimensional proxy $W$로 사용한다. 관측 공변량 $X$와 이미지 $W$에서 CNN 또는 ViT representation $Z=\phi(X,W_{-S})$를 학습한다. 학습된 $Z$가 held-out pixels $W_S$에 남아 있는 처치 정보를 줄이는지, 그리고 그 representation으로 계산한 ATE 후보들이 Algorithm 1의 screening과 aggregation을 통해 참값 $\tau=1$에 가까워지는지를 검증한다.

이 실험의 image-level 주장은 다음 두 개로 제한한다.

1. P0에서 실제 pixel support에 대해 확인한 **known witness의 식별 가정**.
2. 이후 fresh data에서 관찰한 **learned CNN/ViT의 empirical performance**.

원고 Algorithm 1 finite-sample theorem을 문자 그대로 검증하는 역할은 기존 exact control이 맡는다. 이 image 실험은 그 exact control을 대체하지 않는다.

## 2. 사용자 결정 ledger

아래 20개 결정은 이 PRD의 고정 입력이다.

| 번호 | 결정 | 구현 의미 |
|---:|---|---|
| 1 | Shapes3D primary | MNIST를 primary로 유지하지 않는다 |
| 2 | 실제 image 가정과 good $\phi$를 P0에서 검증하고 CNN 성능은 empirical claim | literal theorem control은 기존 별도 실험 |
| 3 | Shapes3D infeasible이면 중단 | 다른 benchmark나 approximate claim으로 자동 전환 금지 |
| 4 | controlled semisynthetic 해석 | 교육·의료 서사를 억지로 붙이지 않음 |
| 5 | diverse geometry from start | P0부터 $8\times4\times15=480$ geometry 조합 사용 |
| 6 | 먼저 $J=16$ regular $4\times4$ grid | 필요하면 성능을 보기 전에 정한 adjacency grouping menu를 순서대로 검사 |
| 7 | 각 hue attribute의 10값을 두 그룹으로 사용 | 낮은 5개와 높은 5개를 이진 channel로 사용 |
| 8 | latent four states | $(U_c,U_t)\in\{-1,1\}^2$ |
| 9 | $X$는 두 좌표 | $X_1$은 noisy $U_c$, $X_2$는 독립 Rademacher |
| 10 | DGP 하나 | 결과를 본 뒤 coefficient variant 추가 금지 |
| 11 | CNN과 ViT 모두 | CNN primary, ViT secondary |
| 12 | end-to-end encoder update | frozen feature substitute 금지 |
| 13 | random initialization, balance-only | pretraining, SSL, reconstruction, $A/Y$ warm start 금지 |
| 14 | core comparator 5개 | PROBE, untrained, raw image, $X$-only, oracle |
| 15 | one honest split | rotation 없이 $D/N/E$ 분리 |
| 16 | CNN primary | CNN 실패 시 ViT를 primary로 교체하지 않음 |
| 17 | same 20 fresh datasets | 방법 간 paired dataset IDs |
| 18 | 총 $n=24{,}000$ | $n_D=6000,n_N=6000,n_E=12000$ |
| 19 | 총 10 GPU-hour | 두 architecture의 모든 job을 순차 실행 |
| 20 | 이 PRD 한 파일만 저장 | 현재 턴에 다른 mutation 없음 |

## 3. Source dataset contract

### 3.1 공식 source

대상은 Google DeepMind의 3D Shapes dataset이다. 표준 factor grid는 다음과 같다.

공식 source 설명은 [Google DeepMind 3D Shapes README](https://github.com/google-deepmind/3d-shapes/blob/master/README.md)를 authority로 삼는다. 실행자는 URL만 기록하는 데 그치지 않고 실제 사용한 archive/HDF5 bytes의 digest를 별도로 봉인해야 한다.

- floor hue: 10
- wall hue: 10
- object hue: 10
- scale: 8
- shape: 4
- orientation: 15

전체 조합 수는 $10^3\times8\times4\times15=480{,}000$이다.

실행 시 다음을 protocol에 기록한다.

- 공식 README URL과 retrieval date
- 원본 archive와 추출된 HDF5의 SHA-256
- image dataset key, factor-label key, dtype와 shape
- factor ordering과 공식 factor-value arrays
- source byte count

현재 문서에는 확인하지 않은 SHA를 쓰지 않는다. SHA가 없는 결과는 `complete=false`다.

### 3.2 원본 이미지 불변 조건

$W$는 dataset에 저장된 한 장의 원본 RGB image다.

허용되는 입력 처리는 다음뿐이다.

- 정수 pixel을 $[0,1]$로 나누는 dtype scaling
- retained/held-out 좌표를 나타내는 별도 binary mask channel
- batching을 위한 tensor layout 변경

금지되는 처리는 다음과 같다.

- crop, resize, rotation, affine transform
- hue 변경, color jitter, normalization으로 색 정보 제거
- 새로운 renderer로 재생성
- pixel noise, blur, quantization
- learned preprocessing을 P0 exact certificate에 사용

Scaling은 원래 정수 pixel을 정확히 역복원할 수 있는 dtype을 사용한다. Source image bytes와 mask는 별도 보관한다.

## 4. Scalar reference SCM

### 4.1 잠재변수와 관측변수

$$
U_c,U_t\overset{\mathrm{iid}}{\sim}\operatorname{Unif}\{-1,1\}.
$$

$X_1$은 $U_c$를 확률 $0.65$로 올바르게 측정한다.

$$
P(X_1=U_c\mid U_c)=0.65.
$$

$$
X_2\sim\operatorname{Unif}\{-1,1\},
\qquad X_2\perp(U_c,U_t,X_1).
$$

세 scalar proxy channel은

$$
P(V_1=U_c\mid U_c)=0.85,
\qquad
P(V_2=U_c\mid U_c)=0.70,
$$

$$
V_3=\mathbf 1\{U_c=1\ \text{or}\ U_t=1\}.
$$

모든 Bernoulli measurement error는 $(U_c,U_t,X_2)$ 조건에서 서로 독립이다.

### 4.2 Treatment와 outcome

$$
e(X,V,U_t)
=
P(A=1\mid X,V,U_t)
=
0.5+0.25V_1+0.15U_t+0.05X_2.
$$

따라서 overlap은

$$
0.05\le e\le0.95.
$$

$$
Y(a)
=
a-2.5U_c+0.3X_1+0.4X_2+\varepsilon_Y,
\qquad
\varepsilon_Y\sim N(0,0.3^2).
$$

$\varepsilon_Y$는 나머지 모든 변수와 독립이다. 참 ATE는

$$
\tau=E\{Y(1)-Y(0)\}=1.
$$

### 4.3 이미 계산된 scalar reference

64-state exact enumeration의 reference 값은 다음과 같다.

| functional | 값 |
|---|---:|
| naive difference | $-0.607000$ |
| $X=(X_1,X_2)$ adjustment | $-0.627450261141$ |
| good $\phi=(X,V_1)$ adjustment | $1.000000$ |
| raw $(X,V_1,V_2,V_3)$ adjustment | $1.291167121762$ |
| oracle | $1.000000$ |

이 숫자는 future implementation이 독립적으로 재현해야 한다. 현재 대화의 stdout은 durable evidence가 아니다.

특히 raw CNN이 $1.291167$을 자동으로 목표로 한다고 가정하지 않는다. 그 값은 full image가 scalar $(V_1,V_2,V_3)$와 정보적으로 동등하다는 P0 certificate가 있을 때만 population reference가 된다.

## 5. Scalar state를 원본 Shapes3D image에 매핑하는 규칙

### 5.1 Hue grouping

각 hue factor의 공식 ordered index를 $0,\ldots,9$로 쓴다.

$$
G(h)=
\begin{cases}
-1,&h\in\{0,1,2,3,4\},\\
+1,&h\in\{5,6,7,8,9\}.
\end{cases}
$$

조건부로 요구된 sign group 안의 다섯 hue index를 균등하게 뽑는다.

- wall hue group encodes $V_1$
- floor hue group encodes $V_2$
- object hue group encodes $2V_3-1$

Hue의 정확한 값은 group 안의 관측 nuisance다. Learner에는 $V_j$, hue group, factor label을 주지 않는다.

### 5.2 Geometry

$$
(\text{scale},\text{shape},\text{orientation})
$$

은 $8\times4\times15=480$개 조합 전체에서 균등하게 뽑으며, scalar 변수와 독립이다. P0 첫 단계부터 480개를 모두 사용한다. Geometry를 고정한 쉬운 pilot은 없다.

### 5.3 한 row의 생성

1. $(U_c,U_t,X_1,X_2,V_1,V_2,V_3)$를 scalar SCM에서 생성한다.
2. 세 hue group 안에서 각각 hue index를 균등 추출한다.
3. 480 geometry 조합에서 하나를 균등 추출한다.
4. 여섯 factor index로 HDF5의 기존 image row를 찾는다.
5. 해당 image bytes를 그대로 $W$로 반환한다.
6. $A$와 $Y$는 scalar structural equations에서 생성한다.

같은 factor row가 여러 dataset row에서 다시 나올 수 있다. 이는 finite empirical image distribution에서의 i.i.d. 복원추출이다.

## 6. Data identity와 honest roles

### 6.1 한 dataset replicate

각 replicate는 총 $24{,}000$개의 독립 row draw를 만든다.

| role | 크기 | 용도 |
|---|---:|---|
| $D_{\mathrm{fit}}$ | 4000 | alternating encoder/critic optimization, primary checkpoint와 screen |
| $D_{\mathrm{select}}$ | 1000 | 고정된 $\phi$의 diagnostic critic checkpoint만 선택 |
| $D_{\mathrm{audit}}$ | 1000 | 고정된 $\phi$와 critic의 untouched 평가만 수행 |
| $N_{\mathrm{fit}}$ | 4800 | propensity/outcome nuisance fit |
| $N_{\mathrm{val}}$ | 1200 | nuisance early stopping |
| $E$ | 12000 | AIPW scores와 최종 aggregation |

따라서 $n_D=6000,n_N=6000,n_E=12000$이다.

### 6.2 중복과 leakage

동일한 source image identity가 서로 다른 독립 row draw에서 우연히 반복될 수 있다. 이것은 row leakage가 아니다. 다음은 반드시 달라야 한다.

- row ID
- RNG draw position
- outcome noise draw
- role assignment

동일 row ID가 둘 이상의 role에 나타나면 즉시 실패다.

Image identity-disjoint split은 기본값이 아니다. 이를 강제하면 finite-bank 분포와 support를 바꾼다. 따라서 이 실험은 unseen-image generalization을 주장하지 않는다.

## 7. Pixel block menu

### 7.1 고정 검사 순서

P0는 아래 메뉴를 정확히 이 순서로 검사한다.

1. `grid16`: $64\times64$ 좌표를 regular $4\times4$로 나눈 $J=16$ partition
2. `quad4`: grid16의 인접 $2\times2$ cells를 합친 $J=4$ supercell partition
3. `hband4`: 높이가 같은 horizontal bands $J=4$
4. `vband4`: 폭이 같은 vertical bands $J=4$

각 partition은 모든 pixel을 정확히 한 번 덮고 서로 겹치지 않는다. 경계는 integer array-split 규칙으로 정한다.

첫 번째로 모든 P0 조건을 통과한 partition을 선택한다. ATE error나 CNN 성능을 보고 선택하지 않는다.

### 7.2 Algorithm split universe

선택된 $J$에 대해

$$
\mathfrak S=\{S:S\subset\{1,\ldots,J\},\ 0<|S|<J\},
\qquad T=2^J-2.
$$

실제 학습 budget은

$$
m(J)=\min(128,T)
$$

이며, split-seed로 uniform without replacement 추출한다. P0 known-witness 검사는 이 학습 budget과 구분한다.

## 8. P0: one-hour pixel feasibility certificate

### 8.1 P0의 질문

P0는 “모든 CNN이 성공하는가”를 묻지 않는다. 다음 존재 명제를 검사한다.

> 사전 지정한 mask menu 중 하나에 대해 retained pixels의 genuine function인 known witness $\phi^*(X,W_{-S})$가 존재하며, held-out raw pixels가 원고 Section 3의 outcome-relevant operator 조건을 만족하고, $A\perp(X,W_S)\mid Z^*$가 성립하는가?

### 8.2 Known witness 구성

Generator metadata는 **oracle preflight에서만** 사용할 수 있다. 예를 들어 retained-pixel exact hash가 $V_1$을 유일하게 결정하면

$$
\phi^*(X,W_{-S})=(X,V_1)
$$

lookup을 witness로 사용할 수 있다.

필수 조건은 동일 retained pixel hash가 서로 다른 $V_1$ 값에 나타나지 않는 것이다. 이 조건이 성립해야 $V_1$ lookup이 metadata oracle이 아니라 retained pixels의 실제 함수다.

이 lookup, $V_1$ label, factor metadata는 CNN training에 공급하지 않는다.

### 8.3 Lossless pixel-state compression

각 mask view의 원본 pixel bytes를 SHA-256으로 묶는다. Hash collision 방지를 위해 같은 digest를 가진 payload는 byte equality로 다시 확인한다.

동일 byte payload끼리만 하나의 observed state로 압축한다. 근사 clustering이나 floating tolerance는 사용하지 않는다.

### 8.4 Conditional operator certificate

각 representation stratum $Z=z$에서 원고 표기와 동일하게

- latent/support column을 $r=(X,W_{-S},U)$,
- observed target을 $T=(X,W_S)$

로 둔다. Sparse channel operator를

$$
K_z(t,r)=P(T=t\mid r,Z=z)
$$

로 정의한다.

$W_{-S}$는 factor label로 바꾼 surrogate가 아니라 §8.3의 lossless retained-pixel state다. Scalar $(V_1,V_2,V_3,U)$로 열을 줄이려면 모든 retained-pixel state에서 conditional channel과 outcome mean을 보존한다는 별도 sufficiency proof가 필요하다. 그 증명 없이 scalar support로 대체하지 않는다.

Outcome mean vector를

$$
\mu_a(z,r)=E\{Y(a)\mid Z=z,r\}
$$

라 한다. Exact outcome-relevant completeness의 computable finite-support 조건은

$$
\mu_a(z,\cdot)
\in
\operatorname{rowspan}
\begin{bmatrix}
K_z\\
\mathbf 1^\top
\end{bmatrix}
$$

가 $a=0,1$에 대해 성립하는 것이다.

검사는 rational DGP probabilities와 exact pixel-state counts를 사용한다. Exact ORC PASS에는 exact rational row-space membership certificate 또는 rigorous interval rank/membership proof가 필요하다. Interval이 0이 아닌 residual 가능성을 남기거나 lossless retained-pixel columns를 한 시간 안에 처리하지 못하면 `INCONCLUSIVE/STOP`이다. 단순 SVD residual $<10^{-10}$만으로 exact proof라고 부르지 않는다.

### 8.5 원고 conditional-independence gates

Generator와 pixel channel은 각 후보 $S$에 대해 원고 조건을 직접 검사하고 기록한다.

$$
Y(a)\perp A\mid(U,X,W_{-S}),
\qquad
W_S\perp A\mid(U,X,W_{-S}).
$$

첫 조건은 structural equations와 독립 outcome noise에서 증명한다. 둘째 조건은 단순한 시간 순서만으로 증명하지 않는다. 선택된 witness에서 retained pixels가 $V_1$을 결정하므로 $(U,X,W_{-S})$가

$$
e=0.5+0.25V_1+0.15U_t+0.05X_2
$$

를 결정하고, 독립 Bernoulli treatment noise로부터 $W_S\perp A\mid(U,X,W_{-S})$를 도출해야 한다. 이 도출이 성립하지 않는 $S$는 자동 PASS하지 않는다. ORC와 observable balance만으로 이 두 조건을 대신하지 않는다.

### 8.6 Double-leakage 검사

후보 split은 다음을 모두 통과해야 한다.

1. retained pixels가 $V_1$ known witness를 정의한다.
2. held-out pixels의 operator가 outcome row-space를 복원한다.
3. held-out pixels가 $V_3$ 또는 $U_t$의 추가 treatment 정보를 누출하여 balance를 깨지 않는다.
4. retained witness가 held-out raw pixels를 직접 복사하지 않는다.
5. overlap $[0.05,0.95]$가 representation strata에서도 유지된다.

### 8.7 Population balance

Known witness에 대해

$$
D_{\mathrm{res},S}^2(\phi^*)
=
E\left[
\{P(A=1\mid Z^*,X,W_S)-P(A=1\mid Z^*)\}^2
\right]
$$

를 exact finite enumeration으로 계산한다. $D^2=0$을 수치 반올림으로 선언하지 않는다.

### 8.8 Geometry 전수 조건

P0 pass는 480 geometry 조합 전체를 포함한 sparse support로 계산한다. Geometry sampling은 빠른 reject 또는 runtime estimate 진단에만 쓸 수 있다. Sampled audit는 exact pass certificate가 아니다.

### 8.9 P0 시간과 상태

- wall-clock cap: CPU 1시간
- memory cap: 실행 host에서 사전 측정 후 protocol에 기록
- 메뉴 하나 실패: 다음 메뉴로 이동
- 메뉴 전체 실패: `NO_CERTIFICATE/STOP`
- 시간 또는 memory 안에 결론 불가: `INCONCLUSIVE/STOP`

한 witness가 실패했다고 모든 neural representation이 불가능하다고 주장하지 않는다.

## 9. P1 구현 smoke

P0 pass와 별도 실행 승인이 있을 때만 P1을 시작한다.

P1은 한 개발 dataset에서 다음만 검증한다.

- source hash와 factor index replay
- mask coverage와 split draw
- $D/N/E$ row disjointness
- CNN/ViT forward와 backward
- augmented critic이 실제 held-out pixels를 읽음
- $E$ row가 encoder, critic, nuisance fitting에 사용되지 않음
- AIPW와 aggregation JSON schema
- peak GPU memory와 one-split timing

P1은 성능 검증이 아니다. P1 GPU cap은 30분이다.

## 10. Representation architectures

### 10.1 공통 입력과 출력

Encoder 입력은 다음 네 channel이다.

- RGB image divided by 255
- retained binary mask 한 channel

Held-out pixels는 encoder RGB 입력에서 0으로 가리고 mask channel로 missingness를 명시한다. 원본 bytes는 바꾸지 않는다.

출력은 continuous code

$$
R_W\in\mathbb R^{16}
$$

이고 최종 representation은

$$
Z=(X_1,X_2,R_W)
$$

다. $X$는 별도 passthrough로 보존한다.

### 10.2 CNN primary

기본 CNN은 다음으로 고정한다.

1. Conv $4\to32$, kernel 3, stride 2, padding 1
2. Conv $32\to64$, kernel 3, stride 2, padding 1
3. Conv $64\to128$, kernel 3, stride 2, padding 1
4. Conv $128\to128$, kernel 3, stride 2, padding 1
5. 각 stage에 GroupNorm(8 groups)와 GELU
6. global average pooling
7. Linear $128\to64$, GELU, Linear $64\to16$

Dropout은 사용하지 않는다. 초기화는 PyTorch default를 seed로 고정한다.

### 10.3 ViT secondary

기본 ViT는 다음으로 고정한다.

- patch size 8
- embedding width 64
- transformer blocks 4
- attention heads 4
- MLP hidden width 128
- learned positional embedding
- LayerNorm pre-norm
- mean-token pooling
- Linear $64\to16$

CNN이 실패해도 ViT를 primary로 승격하지 않는다.

### 10.4 금지된 학습

- ImageNet 또는 Shapes3D pretraining
- digit/factor labels
- reconstruction loss
- contrastive/self-supervised loss
- $A$ 또는 $Y$ warm start
- oracle $U,V$ labels
- effect truth를 사용한 checkpoint selection

Encoder update는 balance objective에서만 온다.

## 11. Critics와 empirical discrepancy

### 11.1 Base critic

Base critic은 $(X,Z)$를 입력으로 받는 MLP다.

- widths: 64, 64
- activation: GELU
- output: sigmoid probability

### 11.2 Augmented critic

Augmented critic은 다음을 모두 입력으로 받는다.

- $(X,Z)$
- 실제 held-out RGB pixels
- held-out binary mask

Held-out pixel branch는 해당 architecture의 독립 CNN 또는 ViT trunk를 사용한다. Encoder frozen features, PCA, scalar summaries로 대체하지 않는다.

Augmented critic은 base critic을 정확히 포함한다. Base critic의 logit network를 별도로 복사하고 held-out pixel residual logit을 0으로 초기화한다. Residual이 0이면 augmented prediction이 copied base prediction과 정확히 같다. 두 critic이 parameter를 공유할 필요는 없다. 매 critic fit에서 이 unoptimized lifted-base candidate도 비교하여 augmented fitted risk가 fallback보다 커지지 않게 한다.

### 11.3 Primary objective

같은 row set에서 실제 optimizer가 반환한 critic을 $\widehat q_0,\widehat q_1$이라 하고

$$
\widehat R_0
=
\frac1n\sum_i\{A_i-\widehat q_0(Z_i)\}^2,
$$

$$
\widehat R_1
=
\frac1n\sum_i\{A_i-\widehat q_1(Z_i,X_i,W_{S,i})\}^2
$$

를 계산한다.

Encoder objective와 primary screen statistic은 모두

$$
\widehat D^2
=
\max\{\widehat R_0-\widehat R_1,0\}
$$

이다. 이 값들은 neural class의 infimum이 아니라 **fitted empirical risks**다. 알 수 없는 critic optimization error는 별도 제한이며 0이라고 선언하지 않는다. Signed gap, absolute gap, independently evaluated gap으로 바꾸지 않는다.

### 11.4 Alternating optimization

한 encoder epoch의 순서는 고정한다.

1. $\phi$를 고정하고 $D_{\mathrm{fit}}$에서 base와 augmented critic을 각자의 Brier loss로 적합한다.
2. Augmented fit은 copied-base logit과 zero residual fallback을 반드시 포함한다.
3. Critic parameter를 고정하되 $Z=\phi(X,W_{-S})$를 통한 gradient graph는 끊지 않는다.
4. Encoder parameter만 update하여 $D_{\mathrm{fit}}$ clipped gap을 줄인다.
5. Update 후 full $D_{\mathrm{fit}}$에서 두 fitted risks와 clipped gap을 다시 계산한다.

Critic weight freezing은 encoder로 가는 gradient를 막는 `detach`가 아니다. Gradient는 fixed critic의 입력 $Z$를 거쳐 encoder로 흐른다.

### 11.5 Primary checkpoint와 독립 diagnostic

- **Primary checkpoint와 Algorithm 1 screen:** 각 checkpoint의 full $D_{\mathrm{fit}}$ fitted clipped Brier gap을 사용한다. 같은 statistic으로 encoder를 선택하고 screen한다.
- **Independent diagnostic:** primary $\phi$를 고정한 뒤 fresh critic candidates를 $D_{\mathrm{fit}}$에서만 적합한다. $D_{\mathrm{select}}$는 critic checkpoint만 선택한다. 선택된 critic과 $\phi$를 모두 고정하고 $D_{\mathrm{audit}}$에서는 risk와 SE만 평가한다.
- $D_{\mathrm{select}}$와 $D_{\mathrm{audit}}$는 primary encoder checkpoint, primary screen, $t$를 바꾸지 않는다.

Audit rows에서는 어떤 parameter도 적합하거나 update하지 않는다. Audit gap은 observational diagnostic이며 encoder 선택이나 screen을 소급 변경하지 않는다.

## 12. Optimization defaults

### 12.1 공통 optimizer

- optimizer: AdamW
- encoder learning rate: $3\times10^{-4}$
- critic learning rate: $10^{-3}$
- weight decay: $10^{-4}$
- batch size: 256
- gradient norm clipping: 5
- maximum encoder epochs: 60
- critic refresh: 매 encoder epoch
- critic inner epochs: 최대 5
- checkpoint interval: 5 encoder epochs
- patience: 4 checkpoints
- random restarts: architecture별 1

이 값은 미검증 default다. Development 전에 protocol에 freeze한다. 결과를 본 뒤 epoch, width, code dimension을 바꾸지 않는다.

### 12.2 Primary checkpoint rule

Full $D_{\mathrm{fit}}$ fitted clipped gap이 가장 작은 checkpoint를 primary representation으로 선택한다. 동률 허용오차는 $10^{-12}$이며, 동률이면 더 이른 epoch를 선택한다. 선택 후 §11.5의 독립 diagnostic critic을 별도로 적합한다.

Optimizer success는 objective value와 finite gradient만 뜻한다. Global ERM certificate를 주장하지 않는다.

## 13. Screen threshold와 split budget

### 13.1 Development threshold grid

허용 threshold grid는

$$
\mathcal T=
\{0,2.5\times10^{-5},5\times10^{-5},10^{-4},2.5\times10^{-4},5\times10^{-4},10^{-3},2\times10^{-3}\}
$$

다.

한 development dataset에서 $m(J)$개 split을 같은 draw order로 평가한다. Primary $D_{\mathrm{fit}}$ fitted gaps만 사용하여 다음 조건을 만족하는 가장 작은 $t$를 선택한다.

1. retained count가 최소 1,
2. 모든 retained candidate의 fitted propensity가 $[0.05,0.95]$ 안.

이 규칙은 $E$의 effect estimates, graph, Algorithm 1 반환값, $\tau$, $U$, factor labels, oracle error를 사용하지 않는다. Eligible $t$가 없으면 development FAIL/STOP이다. 선택한 $t$는 development 직후 봉인하며 confirmation replicate마다 다시 선택하지 않는다.

### 13.2 Linking radius

Operational linking radius는

$$
\rho=0.1
$$

로 고정한다. 이는 population uniform-radius certificate가 아니다. Confirmation 뒤에 $\rho$를 재조정하지 않는다.

## 14. Nuisance와 honest AIPW

각 retained split마다 $N_{\mathrm{fit}}$에서 다음을 적합한다.

- propensity MLP: widths 64, 64, GELU, sigmoid
- arm-specific outcome MLP 두 개: widths 64, 64, GELU
- optimizer: AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$
- batch size: 256
- max epochs: 100
- early stopping: $N_{\mathrm{val}}$, patience 10

Propensity prediction은 평가 시 $[0.05,0.95]$로 clip하고 clip fraction을 저장한다.

$E$에서

$$
\widehat\psi_i
=
\widehat\mu_1(Z_i)-\widehat\mu_0(Z_i)
+\frac{A_i\{Y_i-\widehat\mu_1(Z_i)\}}{\widehat e(Z_i)}
-\frac{(1-A_i)\{Y_i-\widehat\mu_0(Z_i)\}}{1-\widehat e(Z_i)}
$$

를 계산하고 $\widehat\theta_S=|E|^{-1}\sum_i\widehat\psi_i$로 둔다.

## 15. Algorithm 1 aggregation

1. $m(J)$개 split을 uniform without replacement로 뽑는다.
2. $\widehat D_S^2\le t$이고 overlap을 통과한 split만 유지한다.
3. 각 retained split의 honest $\widehat\theta_S$를 계산한다.
4. 두 후보가
   $$
   |\widehat\theta_S-\widehat\theta_{S'}|\le2\rho
   $$
   이면 연결한다.
5. largest connected component를 찾는다.
6. 유일하면 그 component의 median을 반환한다.
7. largest component tie면 candidate set을 반환하고 해당 replicate는 performance gate에서 non-return으로 센다.

Image learned plurality는 empirical claim이다. Scalar 4-channel $2{:}1$ census를 Shapes3D geometry나 pixel partition에 자동 승계하지 않는다.

## 16. 비교 방법 다섯 개

### 16.1 PROBE

위의 end-to-end balance encoder, screen, AIPW, aggregation 전체다.

### 16.2 Untrained same-initial representation

PROBE와 같은 initialization의 encoder를 update하지 않는다. 같은 split draw, critic, nuisance, $t$, $\rho$를 사용한다. Random weights는 warm start가 아니다.

각 architecture·dataset에서 첫 optimizer step 전에 encoder checkpoint를 저장한다. PROBE가 실제 사용하는 restart의 initial checkpoint와 untrained comparator의 checkpoint가 byte-identical해야 하며, 양쪽 결과에 같은 SHA-256을 기록한다. 현재 protocol은 restart 1개이지만 이 검사는 future restart 수 변경에도 fail closed로 유지한다.

### 16.3 Strong raw-image baseline

Full original RGB image와 $X$를 직접 입력받아 propensity와 outcome을 학습한다. Primary architecture와 같은 CNN backbone을 사용한다. ViT secondary에는 같은 ViT raw baseline을 둔다.

Raw model은 $N_{\mathrm{fit}}$에서 supervised objective

$$
\mathcal L_{\mathrm{raw}}
=
\operatorname{BCE}\{A,\widehat e(X,W)\}
+\sum_{a\in\{0,1\}}\mathbf 1(A=a)\{Y-\widehat\mu_a(X,W)\}^2
$$

로 backbone과 propensity/outcome heads를 공동 적합한다. $N_{\mathrm{val}}$의 같은 supervised loss로 checkpoint를 선택한다. PROBE encoder의 balance-only 제한은 이 명시적으로 supervised인 strong raw baseline을 금지하지 않는다.

Raw budget은 PROBE와 같은 optimizer hyperparameter menu를 쓰되 128배의 무의미한 restart를 강제하지 않는다. Freeze 전에 다음을 ledger로 저장한다.

- optimizer steps
- pixel examples seen in forward/backward
- restart 수
- checkpoint 선택 횟수
- wall time와 peak memory

결과 표에는 statistical performance와 compute fairness를 별도 열로 쓴다.

### 16.4 $X$-only

$(X_1,X_2)$만 사용하는 동일 nuisance family의 AIPW다.

### 16.5 Oracle

$(X_1,X_2,U_c,U_t)$를 사용하는 AIPW다. Oracle은 evaluator baseline이며 learner나 selection에 들어가지 않는다.

Naive difference와 analytic scalar values는 diagnostic으로만 기록한다. 새로운 tuned baseline으로 세지 않는다.

## 17. Staged execution과 stop rules

### Stage 0: P0

- 별도 실행 승인 필요
- CPU 1시간
- 결과: `PASS`, `NO_CERTIFICATE/STOP`, 또는 `INCONCLUSIVE/STOP`

### Stage 1: P1 smoke

- P0 PASS 뒤 별도 승인 필요
- GPU 0.5시간 이내
- correctness만 검사

### Stage 2: one full development dataset

- CNN과 ViT 모두 $n=24{,}000$
- threshold 선택, finite-loss/gradient, timing, memory와 반환 상태 확인
- 점추정과 diagnostics는 보고하지만 한 development dataset으로 20-replicate confirmation gate를 PASS/FAIL 판정하지 않음
- nonfinite output, correctness failure, P0 위반 또는 compute forecast failure면 STOP/REPORT
- architecture rescue, coefficient 변경, 새 loss 추가 금지

### Stage 3: seal

다음을 immutable protocol에 고정한다.

- source SHA와 source snapshot SHA
- DGP coefficients와 factor map
- mask partition과 split draw rule
- architecture와 optimizer
- $t$, $\rho$, $m$
- development IDs와 confirmation namespace
- five methods와 gates

### Stage 4: fresh confirmation

- fresh independent dataset IDs 20개
- 모든 방법이 동일 dataset ID를 사용
- CNN 20개를 primary로 완료
- ViT 20개를 secondary로 완료
- 결과를 보며 중단하거나 threshold를 바꾸지 않음

10 GPU-hour forecast를 넘으면 confirmation을 시작하지 않는다. CNN만 실행하고 ViT를 버리거나 replicate 수를 줄여 성공 주장을 만들지 않는다.

## 18. GPU-hour budget

| 단계 | 상한 |
|---|---:|
| P1 smoke: 모든 critic/baseline 포함 | 0.5 GPU-h |
| full development CNN: 모든 critic/baseline 포함 | 0.75 GPU-h |
| full development ViT: 모든 critic/baseline 포함 | 0.75 GPU-h |
| confirmation CNN 20: 모든 critic/baseline 포함 | 4.0 GPU-h |
| confirmation ViT 20: 모든 critic/baseline 포함 | 4.0 GPU-h |
| aggregation/summary allowance | 0.0 GPU-h |
| 합계 | 10.0 GPU-h |

P1과 full development의 실제 elapsed time으로

$$
\widehat H
=
H_{\mathrm{used}}
+20\widehat h_{\mathrm{CNN}}
+20\widehat h_{\mathrm{ViT}}
$$

를 계산한다. 이 합계는 raw baseline, untrained comparator, 모든 critic fit, nuisance fit, 실패한 job과 재시도까지 포함한다. $\widehat H>10$이면 confirmation은 `COMPUTE_FORECAST_FAIL/STOP`이다.

## 19. Frozen empirical gates

### 19.1 Population/P0 gates

- $E\{Y(1)-Y(0)\}=1$
- scalar naive $<0$
- scalar $X$ adjustment $<0$
- scalar good $\phi$ adjustment $=1$
- image known-witness balance와 operator certificate
- raw image functional은 pixel-information equivalence가 증명된 경우에만 scalar raw reference와 비교

### 19.2 Confirmation gates

20개 dataset replicate를 inference unit으로 한다.

Primary full-performance PASS에는 다음이 모두 필요하다.

1. CNN이 20/20 replicate에서 단일 값을 반환
2. PROBE MAE 점추정 $\le0.05$
3. paired absolute-error difference `PROBE - raw`의 one-sided 95% upper bound $<0$
4. paired absolute-error difference `PROBE - X`의 one-sided 95% upper bound $<0$
5. paired oracle-excess의 one-sided 95% upper bound $\le0.05$

$k$개 반환값에 대한 conditional MAE의 양측 95% Student-$t$ interval은

$$
\overline L_k
\pm
t_{0.975,k-1}\frac{s_k}{\sqrt k}
$$

다. Paired one-sided 95% upper bound는

$$
\overline d_k+t_{0.95,k-1}\frac{s_{d,k}}{\sqrt k}
$$

를 사용한다. 두 임계값을 혼용하지 않는다. Conditional summaries는 별도 보고할 수 있지만 non-return을 제거하고 primary gate를 계산하지 않는다.

19/20 반환은 return rate $0.95$를 만족하지만 full-performance claim은 `INCOMPLETE/NOT PASS`다. Non-return에 0이나 oracle 값을 대입하지 않는다.

이 20-replicate Student-$t$ bound는 finite empirical benchmark gate이며 distribution-free theorem certificate가 아니다.

## 20. Mechanism diagnostics

다음은 설명용이며 primary gate가 아니다.

- retained count와 return kind
- split별 fit/select/audit Brier gaps
- screen pass overlap
- component sizes와 purity evaluator labels
- learned representation의 factor-prediction accuracy evaluator diagnostic
- raw/untrained/PROBE compute ledger
- propensity clip fraction
- candidate estimate dispersion
- mask별 factor leakage table

Factor labels와 $U$는 이 diagnostics 계산에서만 사용하고 학습·threshold·checkpoint 선택에는 사용하지 않는다.

## 21. Future implementation file map

아래는 **제안 경로**이며 현재 존재를 주장하지 않는다.

| 경로 | 역할 |
|---|---|
| `code/shapes3d_original_image.py` | source loader, factor SCM, masks, row ledger |
| `code/shapes3d_pixel_certificate.py` | P0 hash/operator/balance certificate |
| `code/shapes3d_probe_models.py` | CNN, ViT, critics, nuisances |
| `code/run_shapes3d_probe.py` | stage-aware immutable runner |
| `code/summarize_shapes3d_probe.py` | paired gates와 diagnostics |
| `code/test_shapes3d_probe.py` | unit/integration/adversarial tests |
| `results/shapes3d_probe/p0_protocol_v1.json` | P0 frozen inputs |
| `results/shapes3d_probe/p0_certificate_v1.json` | P0 output |
| `results/shapes3d_probe/dev_protocol_v1.json` | development protocol |
| `results/shapes3d_probe/dev_v1.json` | one-dataset development |
| `results/shapes3d_probe/confirm_protocol_v1.json` | sealed confirmation |
| `results/shapes3d_probe/confirm_shards/` | write-once replicate shards |
| `results/shapes3d_probe/confirm_summary_v1.json` | 20-replicate summary |
| `memo/2026-09-18-shapes3d-original-image-results.md` | human-readable result memo |

새 디렉토리와 파일은 각각 별도 승인을 받은 뒤에만 만든다.

## 22. Future JSON schemas

### 22.1 P0 certificate

필수 top-level fields:

```text
protocol_id, complete, generated_at_utc,
source_path, source_sha256, source_shape, source_dtype,
factor_values, dgp, mask_menu, chosen_mask_or_null,
geometry_count, pixel_hash_method,
witness_checks, operator_checks, balance_checks,
status, stop_reason, timing, memory, source_hashes
```

### 22.2 Learned replicate shard

필수 fields:

```text
protocol_id, dataset_id, method, architecture, complete,
row_ledger_digest, image_draw_digest, split_draw_order,
fit_select_audit_ids, model_digests, optimizer_status,
threshold, radius, retained_splits, screen_statistics,
candidate_estimates, aipw_score_digest, aggregation,
baselines, diagnostics, compute_ledger, timing, peak_memory,
source_hashes, input_hashes
```

Raw score arrays은 별도 compressed artifact로 저장하고 JSON에는 SHA와 shape를 둔다.

## 23. Required tests before any scientific run

### 23.1 Source와 generator

- official factor grid가 정확히 $480{,}000$
- factor index→image row round trip
- source SHA mismatch fail closed
- scalar 64-state values 재현
- empirical generator가 독립 large-$n$ check에서 scalar moments에 수렴
- 원본 image bytes가 generator 전후 동일

### 23.2 Masks

- 각 partition의 masks가 disjoint
- 모든 pixels를 정확히 한 번 덮음
- complement 연산 정확
- block-order permutation equivariance
- uniform split sampler가 nonempty proper split만 반환
- $m(J)$ unique draws

### 23.3 Role honesty

- $D/N/E$ row ID intersection이 공집합
- $D_{\mathrm{select}}$가 primary encoder checkpoint, primary screen 또는 threshold selection에 들어가면 실패
- $D_{\mathrm{audit}}$에서 parameter fit/update를 하거나 checkpoint, screen 또는 threshold selection에 사용하면 실패
- $E$가 encoder, critic, nuisance gradient에 들어가면 실패
- latent labels가 learner view에 있으면 실패

### 23.4 Models와 critics

- encoder 입력의 held-out pixels가 0이고 mask channel이 정확
- augmented critic이 raw held-out tensor를 실제 사용
- held-out pixel shuffle이 augmented prediction을 바꾸는 adversarial test
- base parameters를 lifted augmented fallback으로 복사 가능
- augmented empirical loss가 fallback보다 크면 fallback 선택
- synthetic positive-gap fixture에서는 end-to-end balance step이 encoder parameter를 변경
- 실제 cell에서 clipped gap이 0이면 zero gradient와 unchanged encoder를 legitimate outcome으로 기록하고 구현 실패로 판정하지 않음
- untrained comparator parameter는 변경되지 않음
- PROBE의 실제 선택 restart initial checkpoint와 untrained checkpoint의 SHA가 같음

### 23.5 AIPW와 aggregation

- hand-computed tiny AIPW case
- propensity clipping과 clip count
- graph threshold가 정확히 $2\rho$
- unique largest component median
- tied-largest candidate-set 반환
- non-return가 denominator에서 사라지지 않음

### 23.6 Provenance

- protocol이 result보다 먼저 존재
- source와 input hash 전부 검증
- result overwrite 거부
- atomic temp-to-final rename
- `allow_nan=false`
- `complete=true` 전에 모든 shard와 checksum 존재

## 24. Proposed commands — NOT RUN

다음은 future interface 예시다. 현재 명령을 실행하지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B code/test_shapes3d_probe.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B code/shapes3d_pixel_certificate.py \
  --protocol results/shapes3d_probe/p0_protocol_v1.json \
  --output results/shapes3d_probe/p0_certificate_v1.json
```

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B code/run_shapes3d_probe.py \
  --protocol results/shapes3d_probe/dev_protocol_v1.json
```

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B code/run_shapes3d_probe.py \
  --protocol results/shapes3d_probe/confirm_protocol_v1.json \
  --dataset-id 0
```

Runner는 protocol에 없는 seed, architecture, phase, sample size override를 거부해야 한다.

## 25. Proposed seed namespaces

아래 숫자는 future protocol 후보이며 아직 사용되거나 봉인됐다고 주장하지 않는다.

| 목적 | namespace |
|---|---:|
| P0 deterministic ordering | 84,000,000 |
| P1 smoke | 84,100,000 |
| development dataset | 84,200,000 |
| confirmation datasets 0–19 | 84,300,000–84,300,019 |
| split draw | dataset seed + 10,000 |
| CNN initialization | dataset seed + 20,000 |
| ViT initialization | dataset seed + 30,000 |
| nuisance initialization | dataset seed + 40,000 |

동일 dataset ID 안에서 방법들은 image rows, $X,A,Y$, split draw를 공유한다. Model initialization은 protocol에 따라 matched한다.

## 26. Reporting language

P0가 통과해도 다음 표현은 금지한다.

- “Shapes3D가 원고 가정을 일반적으로 만족한다.”
- “CNN이 ORC를 학습했다.”
- “$m=128$로 $2^{16}-2$ universe를 검증했다.”
- “raw population target은 반드시 $1.291167$이다.”
- “20 datasets가 return probability $0.95$를 증명했다.”
- “ViT 성공이 CNN 실패를 대체한다.”

허용되는 표현은 다음과 같다.

- “사전 지정한 pixel partition과 known witness에 대해 finite-support operator certificate를 확인했다.”
- “동일한 20개 fresh dataset에서 random-init balance-only CNN의 empirical error를 평가했다.”
- “이 image arm은 기존 literal theorem control과 분리된 learned empirical validation이다.”

## 27. 완료와 중단의 정의

이 프로젝트의 단계별 상태는 다음 중 하나만 쓴다.

- `PRD_COMPLETE_IMAGE_FEASIBILITY_OPEN`
- `P0_PASS_IMPLEMENTATION_NOT_STARTED`
- `NO_CERTIFICATE_STOP`
- `INCONCLUSIVE_STOP`
- `P1_CORRECTNESS_PASS`
- `DEVELOPMENT_FAIL_STOP`
- `COMPUTE_FORECAST_FAIL_STOP`
- `SEALED_CONFIRMATION_RUNNING`
- `CONFIRMATION_COMPLETE_PASS`
- `CONFIRMATION_COMPLETE_NOT_PASS`

현재 상태는

```text
PRD_COMPLETE_IMAGE_FEASIBILITY_OPEN
```

이다. 다음 행동은 자동 실행이 아니라 **P0 실행에 대한 별도 사용자 승인 요청**이다.

## 부록 A. 독립 $N$을 추가한 scalar 진단

별도 scalar 계산에서는 독립 Rademacher $N$을 네 번째 channel로 추가했을 때 exact screen이

$$
\{V_2\},\quad\{V_2,N\},\quad\{N\}
$$

세 split을 유지했다. 앞의 두 후보는 ORC로 $	au=1$이고, $N$-only 후보는 ORC가 없어 wrong value를 가질 수 있다. 따라서 restricted scalar model에는 $2{:}1$ majority가 있다.

이 계산은 diagnostic appendix다.

- Shapes3D geometry가 자동으로 $N$이 아니다.
- 독립 noise를 pixels에 제조하지 않는다.
- 같은 channel을 여러 patch에 복제하여 plurality를 만들지 않는다.
- image experiment의 target/wrong count는 P0와 learned results에서 새로 평가한다.

## 부록 B. 제1원리 점검

실제 목표는 예쁜 CNN을 학습하는 것이 아니다. 필요한 최소 결과는 다음 순서다.

1. 원본 image pixel channel에 good witness가 존재한다.
2. held-out pixel operator가 outcome-relevant row-space를 보존한다.
3. learned encoder가 관측 데이터만으로 balance objective를 줄인다.
4. honest AIPW와 Algorithm 1이 fresh datasets에서 안정적으로 반환한다.
5. strong raw-image baseline보다 paired error가 작고 oracle에 가깝다.

1–2가 없으면 3–5에 GPU 시간을 쓰지 않는다. 이것이 이 PRD의 가장 중요한 stop rule이다.
