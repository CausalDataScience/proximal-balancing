# MNIST 원본 다중 이미지 실험 PRD v1

작성일: 2026-09-20  
상태: **설계 문서 / 구현·실험 미실행**  
독자: COA 구현 담당자, 실험 검토자, 원고 작성자  
Canonical project: /Users/yonghanjung/paios/research/papers/single_proxy_balancing/

## 0. 목표와 승인 범위

**목표는 원본 이미지 네 장에서 균형 목적만으로 표현을 배우고, 그 표현을 사용한 인과 효과 추정이 단순 전체 이미지 조정보다 정확하며 oracle에 가까운지 검증하는 것이다.**

이 문서는 생성식부터 결과 판정까지 독립적으로 읽을 수 있는 구현 명세다. 알려진 좋은 표현의 존재, CNN이 그 표현에 필요한 정보를 실제로 학습하는지, 최종 효과가 정확한지는 각각 별도 관문이다. 성공은 가정하지 않는다.

- 이번 산출물은 이 PRD와 짝을 이루는 다른 benchmark PRD 두 파일뿐이다.
- 실제 코드·자료·결과·checkpoint·원고를 이 문서 작성 과정에서 변경하지 않는다.
- 구현·로컬 실행·새 산출물 위치는 다음 실행 승인에서 확인한다.
- 다운로드, 외부 upload, remote GPU job은 대상·payload·action을 구체적으로 제시하고 별도 승인받는다.
- 총 20 GPU-hour는 사용자가 승인한 **계획 상한**이며 현재 계정의 남은 quota나 특정 job의 실행 승인을 뜻하지 않는다.
- 기존 SCM-1–3와 기존 이미지 실험의 자료·봉인 기록·실패 결과는 그대로 보존한다.

## 1. 쉽게 읽는 시나리오: 처치 전 검사와 부품 품질

가상의 생산라인에서 부품 하나를 한 관측치로 둔다. 추가 강화 공정을 할 것인지가 처치이고, 공정 뒤의 품질 단위로 표시한 파괴검사 점수가 결과다. 추가 강화는 모든 부품의 점수를 정확히 1만큼 올린다. 예를 들어 같은 부품의 비강화 점수가 2라면 강화 점수는 3이다. 그러나 결함 신호가 높은 부품이 강화를 더 자주 받으므로, 단순 비교에서는 강화한 부품의 평균 품질이 더 낮게 나올 수 있다.

| 변수 | 시나리오에서의 뜻 | 관측 여부 |
|---|---|---|
| $U_c$ | 숨은 재료 취약성. 높을수록 최종 품질이 낮음 | 잠재 |
| $U_t$ | 기록되지 않은 추가검사·escalation 압력. 강화를 더 유도하지만 품질에는 직접 영향 없음 | 잠재 |
| $X_1$ | 기존 기본 conditioning 기록. 취약한 부품에서 더 자주 나타나며 품질에 작은 직접 효과 | 관측 |
| $X_2$ | 기록된 유리한 공정 설정. 품질과 강화 배정 모두에 작은 영향 | 관측 |
| $W_1$ | 불완전한 주 검사 결과를 담은 원본 이미지 | 관측 |
| $W_2$ | 취약성을 독립적으로 다시 측정한 보조 검사 이미지 | 관측 |
| $W_3$ | 취약성 또는 escalation 중 하나가 높으면 켜지는 결합 경보 이미지 | 관측 |
| $W_4$ | 결과·처치와 관련 없는 참조 검사 이미지 | 관측 |
| $A$ | 추가 강화 공정의 시행 여부 | 관측 |
| $Y$ | 공정 뒤 품질 점수 | 관측 |

MNIST 이미지는 작업자가 기록한 검사 수치의 손글씨 스캔을 나타내는 **시각적 대용물**이다. 숫자0–4와5–9가 이진 검사값의 두 그룹이다. 그룹 안에서는 다섯 숫자를 모두 쓰고, 같은 숫자의 필체도 다양하게 뽑는다. 숫자 자체를 숨은 재료 취약성이라고 부르지 않는다. Benchmark가 실제 공장 기록이라는 주장도 하지 않는다.

이 시나리오는 controlled semisynthetic 연구다. Benchmark 이미지가 실제 재료 취약성을 측정한 자료라고 주장하지 않는다. 실제 원본 이미지 선택 분포를 통제해 인과 구조를 실험한다.

네 이미지는 모두 처치 이전이다. 검사 결과가 이미지 선택과 처치 배정에 함께 사용된다. 처치나 결과를 이미지에 새겨 넣지 않는다. 학습기는 검사 이름이나 역할을 모르며 block 번호만 받는다.

## 2. 최소 설계와 원고에 충실한 주장

| 항목 | 사양 |
|---|---|
| $W$ | 원본 MNIST grayscale 이미지4장, 각28×28 |
| block 수 | $J=4$, 이미지 한 장이 block 하나 |
| split | 14개 비어 있지 않은 proper subset 전수 |
| 표현 | $Z=(X,H)$, CNN 몸통과 head 모두 학습 |
| 초기화·목적 | 무작위 초기화, balance-only |
| 핵심 방법 | PROBE, 동일 초기화 untrained, 강한 raw CNN, X-only, oracle |
| 행 수 | replicate당 24,000 |
| 반복 | smoke 1개, 개발 3개, fresh confirmation 20개 |
| 주 모델 | CNN |
| ViT | secondary 후보로 보류. 자동 추가·CNN 대체 금지 |
| 실행 순서 | Shapes3D 먼저, 그동안 MNIST CPU 준비 |
| 공유 예산 | 두 benchmark의 모든 GPU 작업 합계 20시간 |

기존 Shapes3D 결정표의 random-init/balance-only를 유지한다. Label supervision, channel regression, A/Y warmstart, reconstruction, SSL, pretrained backbone은 primary PROBE에서 금지한다. ViT 보류는 계산 범위를 줄인 명시적 제안이며, 실행하려면 별도 범위·예산 결정을 받는다.

원고 authority는 [Section 2](../manuscript/main/2.tex), [Section 3](../manuscript/main/3.tex), [Section 4](../manuscript/main/4.tex)다. 옛 abstract·Introduction·Section 5를 가정의 근거로 삼지 않는다.

| 이론·알고리즘 | 이 실험의 증거 |
|---|---|
| Section 2 관측 구조 | learner가 $(X,W,A,Y)$만 사용 |
| Section 3 식별 | 실제 image support에서 알려진 witness의 가정 확인 |
| Section 3 잘못된 조정 | full-W 모집단 조정값과 좋은 표현의 조정값이 다름 |
| Section 4 표현 학습 | 지정한 CNN/critic로 Brier-risk 기반 학습을 실제 실행 |
| Algorithm 1 | split, screen, honest AIPW, 반경 연결, 최대 성분 중앙값 |

**Faithfulness 경계:** 원고의 경험 Brier 통계는 적합표본의 두 위험 차이다. 여기서는 최종 독립 audit 행의 차이를 primary screen으로 사용한다. 따라서 operational empirical variant로 명시하고 fit/select/audit 세 통계를 모두 저장한다. Global approximate ERM, learned-CNN ORC, stability 상수, plurality, finite-sample 확률 하한을 인증했다고 쓰지 않는다. Literal theorem audit는 기존 positive control의 역할이다.

## 3. 완전한 자료 생성식

### 3.1 잠재 상태, 측정값, 공변량

$U=(U_c,U_t)$는 항상 잠재다. $U_c,U_t,X_2,N_0$는 독립이며 각각 $-1,+1$을 같은 확률로 갖는다. $N_0$는 네 번째 이미지 선택용 잡음이다.

$$
P(X_1=U_c\mid U_c)=0.65,\quad
P(V_1=U_c\mid U_c)=0.85,\quad
P(V_2=U_c\mid U_c)=0.70.
$$

세 flip error는 서로 및 다른 exogenous 난수와 독립이다. $X=(X_1,X_2)$다. 이미지 선택용 부호를

$$
V_3=\mathbf1\{U_c=1\ \text{or}\ U_t=1\},\qquad
M=(V_1,V_2,2V_3-1,N_0)
$$

로 둔다. $V_3$는 0/1이고 $M_3$는 반드시 -1/+1로 변환한다. $V_j,M_j$와 각 측정 역할은 learner 입력이 아니다.

### 3.2 처치, 잠재 결과, 관측 결과

$$
e=0.5+0.25V_1+0.15U_t+0.05X_2,\qquad A\sim\operatorname{Bernoulli}(e).
$$

$$
Y(a)=a-2.5U_c+0.3X_1+0.4X_2+0.3\epsilon_Y,\quad
\epsilon_Y\sim N(0,1),\quad Y=Y(A).
$$

처치 난수와 결과 잡음은 나머지 난수와 독립이다. 두 잠재 결과에는 동일한 $\epsilon_Y$를 사용한다. 참 평균 처치효과는 $\tau=1$이고 $e\in[0.05,0.95]$다.

시나리오의 직접 효과 부호는 식과 맞춘다. 취약성 $U_c$는 Y를 낮춘다. $X_1,X_2$는 Y를 높인다. $U_t$는 A와 경보를 바꾸되 Y에 직접 들어가지 않는다.

### 3.3 원본 이미지 선택

이미지 $W_j$는 $M_j$만을 조건으로 kernel $K(\cdot\mid M_j)$에서 독립 복원추출한다. 네 kernel의 스타일 난수 stream을 분리한다. Source row 선택이 X,U,A,Y 또는 다른 image ID에 추가로 의존하면 구현 FAIL이다.

$M_j=-1$이면 숫자0–4, $M_j=+1$이면5–9에서 숫자를 균등하게 선택한다. 이어 선택한 숫자의 원본 이미지 중 한 장을 균등하게 선택한다. 숫자별 bank 크기가 달라도 먼저 숫자를 균등 선택하므로 그룹 내 digit 확률은각1/5다. 네 block 모두 같은 규칙을 쓰고 라벨은 생성기만 읽는다.

같은 원본 image ID가 여러 관측 행에서 반복될 수 있다. 이는 고정 bank를 조건으로 한 iid 복원추출이다. 실험 행과 실험 seed는 독립이며, bank 공유를 실험 행 누출과 혼동하지 않는다. 이 실험의 범위는 고정 bank에서 새로 만든 causal dataset이다. Unseen-image/OOD 일반화를 주장하지 않는다.

## 4. Source contract와 저장·메모리

기존 원본은 materials/benchmarks/mnist/mnist.npz다. 2026-09-20 read-only 확인값은11,490,434 bytes, SHA256=731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1이다. x_train은60,000×28×28 uint8, x_test는10,000×28×28 uint8이고 대응 label이 있다.

이번 고정 source bank는 x_train60,000장이다. x_test는 이 causal 실험에 자동 추가하지 않는다. Source kernel은 y_train을 이미지 선택에만 사용하고 learner에 제공하지 않는다. 실행 전 SHA를 다시 확인한다. 파일이 다르면 source 변경 승인·새 manifest가 필요하다.

원본 uint8 bytes는 불변이다. 입력 float32/255와 tensor layout 변환만 허용한다. Crop, resize, hue 변경, color jitter, rotation, affine transform, pixel noise, interpolation, marker, local deformation을 금지한다. 가림은 representation의 block 접근 mask로 구현하고 source image를 수정하지 않는다.

Manifest 필수 항목은 source 경로·URL, retrieval date 또는 기존 파일 확인일, bytes, SHA256, array keys/shape/dtype, factor/label 순서, 각 조건부 그룹의 row 수다. 확인하지 않은 SHA를 만들지 않는다.

원본60,000장 uint8는47,040,000 bytes다. 기존 NPZ를 한 번 읽고 source row indices로 조회한다. 24,000행×4이미지는75,264,000 bytes이지만 이미지 복제 대신 index와 batch tensor를 사용한다. GPU에 source bank 전체를 상주시킬 필요가 없다.

새 dependency를 설치해야 하면 별도 승인받는다. 원본 자료 다운로드와 외부 작업에 이 문서만으로 실행 권한이 생기지 않는다.

## 5. P0: 좋은 표현이 이미지에서도 존재하는가

P0는 학습 전에 하는 CPU 검사다. 학습 seed를 열기 전에 다음을 완료한다.

1. Scalar 확률표의 128개 상태를 가중 전수 열거한다. 결과 잡음은 평균0으로 적분하고 A는 참 확률로 합산한다.
2. 아래 functional을 독립 함수로 재현한다. 절대 허용오차는 $10^{-9}$다.
3. 원본 이미지 그룹 간 pixel-byte 동일성을 hash로 검사하고, 같은 hash는 실제 bytes로 확인한다.
4. 반대 M 그룹의 동일 이미지가 하나라도 있으면 exact decoding certificate FAIL이다.
5. Kernel 코드가 M 및 독립 이미지 난수만 읽는지 fixture와 접근 경계를 검사한다.
6. Source bytes와 learner 입력에서 역변환한 bytes가 일치하는지 검사한다.
7. Known witness의 balance·target·공통 held-out channel·outcome-relevant completeness를 확인한다.

| 조정 정보 | 모집단 functional |
|---|---:|
| 무조정 | −0.607000 |
| X | −0.627450261141 |
| $(X,V_1)$ | 1.000000 |
| $(X,V_1,V_2,V_3,N_0)$ | 1.291167121762 |
| oracle $(X,U)$ | 1.000000 |

Pixel group이 정확히 구분되고 kernel이 M에만 의존할 때 full image의 조정 functional은 full channels와 같다. Raw CNN의 실제 학습 결과가 이 functional을 회복한다는 보장은 별도다.

Known witness는 held-out $S=\{2\}$ 또는 $\{2,4\}$에서 $Z^\star=(X,V_1)$다. 남긴 첫 이미지가 $V_1$을 알려 주고, 두 번째 이미지가 $U_c$를 새로 측정한다.

$$
P(V_2\mid U_c)=
\begin{pmatrix}0.7&0.3\\0.3&0.7\end{pmatrix},\qquad \det=0.4.
$$

결과 평균은 $U_c,X$에만 의존한다. 이 full-rank 채널로 outcome-relevant completeness를 확인한다. 전체 $U_c,U_t$에 대한 full completeness라고 부르지 않는다. 가정은 이 witness에 대해 확인하며 모든 learned CNN으로 확대하지 않는다.

$S=\{4\}$에서는 held-out 이미지가 독립 잡음이므로 balance가 공허하게 성립한다. 그 후보가 반드시 오답이 되거나 다른 후보와 분리된다고 예고하지 않는다. P0의 제한된 scalar library census와 자유 CNN의 empirical 후보는 구별한다.

P0 실패 시 GPU 단계로 가지 않는다. 이미지 전수 support hash와 128-state 계산이면 충분하며 모든 이미지쌍의 제곱수 비교를 하지 않는다.

## 6. Honest data roles와 난수 관리

Replicate당 독립 관측 행 24,000개를 생성하고 한 번만 나눈다.

| 역할 | 행 수 | 용도 |
|---|---:|---|
| Dfit | 4,000 | encoder·critic gradient |
| Dselect | 1,000 | encoder·critic checkpoint 선택 |
| Daudit | 1,000 | 최종 독립 screen, 한 번만 |
| Ntrain | 4,800 | propensity·outcome nuisance |
| Nvalidation | 1,200 | nuisance checkpoint |
| E | 12,000 | 최종 AIPW·보고 |

모든 역할 row ID 교집합은 0이다. 같은 source image ID가 반복되는 것과 구별한다. E의 결과·참값을 hyperparameter 선택, screen, early stop에 사용하지 않는다.

Learner batch 허용 필드는 row_id, X, images, A, Y, block_mask다. U,M,label,factor,exact propensity,true effect를 별도 evaluator 객체로 분리한다. Representation과 critic 함수에는 Y 접근을 허용하지 않는다. X는 Z에 그대로 붙인다.

Seed manifest는 dataset, row split, image sampling, encoder, critic, nuisance, raw CNN의 stream을 분리한다. 같은 replicate의 방법은 같은 데이터와 N/E 행을 공유한다. 개발·확증 및 benchmark 간 dataset seed는 겹치지 않는다. 구현 때 기존 결과와 충돌 검사한 뒤 값을 봉인한다. 이 문서는 검증하지 않은 seed 범위를 미사용이라고 선언하지 않는다.

## 7. CNN, critic, 손실의 정확한 명세

### 7.1 공통 encoder

원본 해상도 grayscale28×28, C=1를 유지한다. 각 이미지에 공유 CNN을 적용한다.

Conv(C,16,k3,s2,p1) → ReLU → Conv(16,32,k3,s2,p1) → ReLU → Conv(32,32,k3,s2,p1) → ReLU → global average pooling → Linear(32,16).

Encoder는 남긴 이미지의 픽셀만 fetch하고 처리한다. 네 block feature를 index 순으로 concatenate하되 held-out slot은 입력을 읽지 않고 상수0으로 채운다. 별도4차원 mask를 붙여 68→32→8 MLP(ReLU)로 H를 만들고 $Z=(X,H)$, 차원10으로 둔다. CNN 몸통과 head를 모두 갱신한다.

Dropout/BN 없음. Framework 기본 initialization과 version을 기록한다. Encoder restart는 split당1개다. 같은 초기 checkpoint를 untrained 비교에 보존한다. Role label은 금지하며 block index·mask만 허용한다.

### 7.2 두 critic

Critic은 처치 A의 확률을 예측하는 함수다. Base $q_0(Z)$는 10→32→32→1 ReLU MLP와 sigmoid다.

Augmented $q_1(X,W_S,Z)$는 독립 base branch의 logit과 residual logit을 더해 sigmoid를 취한다. Residual은 자기 별도 CNN으로 held-out 이미지의 **모든 픽셀만** 읽고 나머지 slot은 상수0으로 채운다. 64개 masked features, 4 mask, Z10을 합친 78→32→1 MLP를 사용한다.

Residual 마지막 layer를0으로 두고 base parameters를 복사한 lifted 후보를 보존한다. 이때 augmented가 base 함수와 정확히 같다. Parameter object는 공유하지 않으며 augmented 전 파라미터를 자기 Brier loss로 학습한다. CNN 특징을 미리 고정한 augmented critic으로 대체하지 않는다.

두 critic loss는 각각 mean $(A-q_b)^2$다. 적합·검증·예측에서 같은 확률을 사용한다. Sigmoid는 (0,1) 출력을 쓰며 사후 bounded-map 변환을 학습 손실과 다르게 적용하지 않는다.

### 7.3 bounded optimizer

| 설정 | 값 |
|---|---|
| optimizer | Adam, betas(.9,.999), eps $10^{-8}$ |
| critic lr / encoder lr | $10^{-3}$ / $3\times10^{-4}$ |
| batch | 128, fit행에서 복원추출 |
| weight decay / dropout / BN | 0 / 없음 / 없음 |
| grad clip | norm5 |
| precision | float32, mixed precision off |
| 초기 critic pair 적합 | 100 steps |
| outer iteration | 20 |
| outer당 critic / encoder | 25 / 5 steps |
| encoder 총 갱신 | 100 steps |

Encoder를 고정해 critic을 적합한 뒤 critic parameters를 고정해 encoder를 갱신한다. Z를 통한 gradient는 유지한다. Encoder minibatch loss는 $\max(\widehat R_0-\widehat R_1,0)$다.

Step0과 매 outer에서 fit/select 위험·signed gap·clipped gap·SE·parameter delta를 기록한다. Critic checkpoint는 현재 고정된 encoder를 사용한 해당 적합 구간 안에서 각자의 Dselect Brier risk로 고른다. 서로 다른 encoder에서 기록한 critic 위험을 직접 비교하지 않는다. Encoder는 Dselect clipped gap이 best보다 $10^{-6}$ 초과 개선할 때만 교체한다. 동점은 먼저 나온 checkpoint다. Signed gap이 더 음수라는 이유로 선호하지 않는다.

Step0 선택을 허용한다. 총100번 갱신했다는 사실과 선택된 checkpoint가 실제 갱신되었다는 사실을 따로 기록한다. 처음부터 clipped0이면 gradient0인지 기록한다. 이를 자동 학습 성공으로 판정하지 않는다.

## 8. 최종 audit screen과 honest 효과 추정

### 8.1 독립 재적합과 screen

선택된 encoder를 freeze하고 새 critic pair를600 steps 재적합한다. Dfit만 gradient에 쓰고 Dselect에서25 steps마다 각자의 위험을 평가한다. 최저 개별 위험 checkpoint를 선택하고 동점은 이른 것을 유지한다. Augmented 선택에는 base의 lifted 후보도 포함한다.

그 뒤 encoder·critic을 모두 고정하고 Daudit를 한 번만 읽는다.

$$
d_i=(A_i-q_0(Z_i))^2-(A_i-q_1(X_i,W_{S,i},Z_i))^2,\qquad
G_S=\max(\overline d,0).
$$

Signed gap, clipped gap, $\operatorname{sd}_{\mathrm{ddof}=1}(d_i)/\sqrt{n_B}$를 역할 B별로 저장한다. $n_{\mathrm{fit}}=4000$, $n_{\mathrm{select}}=n_{\mathrm{audit}}=1000$이며 실제 n도 함께 기록한다. Fit/select의 SE는 적합·선택에 재사용된 행의 기술적 진단값이다. 모델을 고정한 뒤 독립적으로 읽은 audit에만 독립 평가 해석을 붙인다. Primary threshold는 $t=0.0005$다. $G_S\le t$이면 balance pass다. NaN/Inf, 음수 SE, 없는 checkpoint는 fail-closed다. 음수 gap으로 통과한 비율을 반드시 별도 보고한다.

### 8.2 Nuisance와 overlap

Frozen Z에서 propensity e와 두 outcome regression $m_0,m_1$을 N으로 적합한다. 모두 input→64→64→1 ReLU MLP다. Propensity는 sigmoid/Brier, outcome은 arm별 MSE다.

Adam lr.001, batch128, weight decay0, max1,000 steps다. Nvalidation을25 steps마다 확인하고 최소100 steps 이후 patience8, min_delta $10^{-5}$로 종료한다. 각 head는 자기 위험으로 checkpoint를 고른다. X/oracle에도 같은 N 분할·절차를 사용한다.

Propensity 예측은 $\eta=0.05$로 $[\eta,1-\eta]$에 clip한다. Clip 전 범위 밖 비율과 clip 후 범위를 모두 기록한다. 이 overlap 확인은 모형의 범위 확인이며 후보가 옳다는 증거가 아니다. 별도1% 탈락 규칙을 만들지 않는다. 비유한 예측은 clipping으로 숨기지 않고 실패 처리한다.

PROBE와 untrained의 대응 split은 nuisance 초기 seed·batch 순서·N 분할을 같게 한다. Encoder digest는 nuisance 적합 전후 같아야 한다.

### 8.3 E에서 AIPW와 집계

E에서만 다음 점수의 평균을 계산한다.

$$
\psi_i=m_1(Z_i)-m_0(Z_i)
+\frac{A_i(Y_i-m_1(Z_i))}{e(Z_i)}
-\frac{(1-A_i)(Y_i-m_0(Z_i))}{1-e(Z_i)},\qquad
\widehat\theta_S=\frac1{12000}\sum_{i\in E}\psi_i.
$$

Split은 14개 전수라 Algorithm 1의 budget은 $m=T=14$다. 보류된 후보끼리 $|\widehat\theta_S-\widehat\theta_{S'}|\le2\rho$이면 연결한다. Primary $\rho=0.10$은 operational 값이다.

유일한 최대 연결 성분의 중앙값을 반환한다. 최대 크기 동점은 candidate set, 빈 retained set은 no-return이다. 0이나 oracle로 대체하지 않는다. 20개 평가 자료 사이에서 최종값을 다시 투표하지 않는다.

## 9. 다섯 비교군과 공정성

| 방법 | 입력·절차 |
|---|---|
| PROBE | balance-only CNN, audit, nuisance, graph |
| Untrained | 동일 초기 encoder. 자체 audit·동일 nuisance·graph |
| Raw image CNN | 모든 원본 이미지를 사용하는 강한 end-to-end 조정 |
| X-only | X만으로 N nuisance/E AIPW |
| Oracle | $(X,U_c,U_t)$로 N nuisance/E AIPW |

Raw는 각 이미지16차원 feature 전체64개와 $X$의 두 좌표를 concatenate한다. PROBE의8차원 bottleneck을 강제하지 않는다. 66→128→64 shared MLP 뒤 e,m0,m1 heads를 둔다. Treatment e는 sigmoid, outcome heads는 linear다.

Raw objective는 Brier(A,e)+관측 arm outcome MSE다. Dfit에서 최대10 epochs supervised pretrain하고 Dselect의 같은 joint risk로 선택한다. 이어 Ntrain에서 CNN을 포함해 end-to-end 최대20 epochs fine-tune한다. Nvalidation joint risk, patience5, min_delta $10^{-5}$, Adam lr.001, batch128, no dropout/BN, weight decay0다. E는 손대지 않는다. 표준화 없이 Y 원척도 MSE를 사용한다. 매 epoch 모든 해당 train행을 shuffle해 한 번 읽는다.

Raw 전체 픽셀 입력과 full-W functional의 완벽한 학습은 별개다. 추정 평균과 reference1.291167의 차이를 함께 보고한다. Raw가 참값에 가깝더라도 나쁘게 만들거나 실험을 무효화하지 않는다. Raw를 포함한 모든 AIPW 방법에 동일한 propensity clipping $\eta=0.05$와 clipping 전 범위 밖 비율 기록을 적용한다.

모든 방법의 parameter 수, gradient steps, CNN forward/backward 수, GPU wall time을 기록한다. Compute-matched라는 주장은 하지 않는다. Scalar good-witness/true-nuisance 계산은 evaluator-only 진단이며 primary 비교군을 대체하지 않는다.

## 10. 단계별 실행과 중단

### P0: source·SCM·witness

§4–5를 CPU에서 완료한다. 실패하면 GPU를 사용하지 않는다. Source 문제, kernel 문제, 수학 문제를 구분해 보고한다. 좋은 결과가 나오도록 coefficient를 자동 탐색하지 않는다.

### P1: smoke와 실제 시간 측정

256행 fixture로 shape·gradient·누출·graph를 검사한다. 그 수치는 성능 근거로 쓰지 않는다. 별도 smoke seed로24,000행의14 splits와 다섯 방법을 전부 실행해 시간 상한을 측정한다.

Trunk가 실제 변하는지, critic이 lifted base와 일치하는지, Daudit/E가 선택에 쓰이지 않는지, 중단 파일을 scorer가 거부하는지 확인한다. 구현 FAIL이면 개발로 가지 않는다.

### P2: 개발3개

고정된 hyperparameter로 독립 dev dataset3개를 실행한다. 확증 진입 기준은 구현 PASS, 3/3 finite 단일 반환, dev MAE≤.10, dev PROBE MAE가 raw/X보다 작음, budget forecast PASS다. 개발3개는 통계적 우월성 확인이 아니다.

기준 미달이면 confirmation seed를 열지 않는다. 자동 warmstart·ViT·channel decoder·threshold rescue를 금지한다. 설계를 바꾸려면 이유와 영향을 적고 새 version 승인 뒤 별도 개발로 간다.

### P3: 봉인과 confirmation20개

P0, source, 코드, scorer, criteria, seed manifest를 hash로 봉인한다. 모든 expected replicate/method/split key를 미리 정한다. Confirmation 자료20개를 모두 실행한다.

실패한 seed를 교체하거나 중간 성공률로 종료하지 않는다. 기술적 재시작은 같은 seed·같은 protocol이고 실패 이력과 비용을 보존한다. 결과를 보고 hyperparameter를 바꾼 실행은 별도 version이며 원래 결과를 남긴다.

## 11. 통계 관문

독립 단위는 dataset20개다. Split14개나 평가행12,000개를 독립 replicate로 세지 않는다. Replicate b, 방법 k의 오차는 $e_{bk}=|\widehat\tau_{bk}-1|$다. MAE는20개 평균이고 p90은 NumPy quantile(method=linear)로 계산한다.

Paired 차이 $d_b=e_{b,\mathrm{PROBE}}-e_{bk}$의 일방95% 상한은

$$
\overline d+t_{0.95,19}\operatorname{sd}_{\mathrm{ddof}=1}(d_b)/\sqrt{20}.
$$

이 t 상한은20개 독립 자료에 대한 통상적인 근사 추론이다. 임의 분포에서 정확한 finite-sample coverage가 인증되었다고 주장하지 않는다. Naive 차이의 상한에도 같은 ddof와 t 규칙을 적용한다.

| Primary gate | 기준 |
|---|---|
| 구현·provenance | 필수 검사와 모든 expected cell 검증 |
| 단일 finite 반환 | PROBE·raw·X·oracle 각각20/20 |
| PROBE MAE | ≤.05 |
| PROBE p90 | ≤.10 |
| Oracle 초과 paired UCB | ≤.05 |
| Raw 대비 paired UCB | <0 |
| X 대비 paired UCB | <0 |
| Simpson reversal | replicate별 naive 차이 평균의 일방95% UCB<0 |

운영상 PROBE 반환율≥.95는 별도 참고값이다. Primary paired 판정에는 PROBE·raw·X·oracle 각각20/20을 더 강하게 요구한다. 이 방법들에 미반환·비유한 값이 있으면 joint performance는 FAIL 또는 실행미완료면 INCOMPLETE다. Returned-only 표는 분모를 명시하고 같은 확인 판정을 내리지 않는다. Untrained 미반환은 별도 기록하며 secondary balance-attribution을 판정불가로 만든다. 그것만으로 primary 우월성 판정을 자동 FAIL로 바꾸지 않는다.

모든 gate를 통과해야 공동 성공을 주장한다. 이는 conjunctive rule이며 각 confidence bound 전체의 simultaneous95% coverage를 주장하지 않는다. P90/반환율 관문도 각각의 모집단 신뢰보증이 아니다.

Untrained 대비 paired improvement는 별도 balance-attribution 판정이다. 통과하지 못하면 balance 자체의 개선을 입증했다고 쓰지 않는다. 두 benchmark 결과는 별도로 채점하며40개 iid로 합치지 않는다.

## 12. 추가 학습 없는 진단

- 초기/final gap, selected step, parameter delta, critic risk와 SE.
- Signed gap<0 통과 비율, retained 수, component 구성, no-return 이유.
- Known witness split, noise split의 실제 추정값과 screen 결과.
- $t\in\{.00025,.0005,.001\}$, $\rho\in\{.05,.10,.20\}$의3×3 sensitivity.
- Sensitivity를 위해14개 split 모두의 nuisance/ATE를 저장한다. Primary는 보류된 후보만 쓴다.
- 이9개 조합 중 최선으로 primary를 교체하지 않는다.
- Block permutation은 mask·input columns·parameters·seed mapping을 함께 바꾸는 구현 동변성 검사다.
- 새로 무작위 학습한 두 CNN의 bitwise 동일성을 요구하지 않는다.

## 13. 공유20 GPU-hour 예산과 실행 순서

두 PRD가 사용하는 예산은 하나다. 각각20시간이 아니다.

| 단계 | Shapes3D | MNIST |
|---|---:|---:|
| Smoke | .75h | .50h |
| Development | 1.50h | 1.00h |
| Confirmation | 8.00h | 5.25h |
| benchmark 소계 | 10.25h | 6.75h |

공유 reserve3.00h를 더해 총20.00h다. 위 숫자는 allocation이며 runtime 예측이나 실제 남은 GPU quota가 아니다. 이전 프로젝트 사용량과 계정 잔여량을 실행 전 확인한다.

순서는 Shapes3D P0→smoke→dev→confirmation, 이후 MNIST smoke→dev→confirmation이다. Shapes3D GPU가 도는 동안 MNIST source·CPU P0·loader·256행 CPU smoke를 진행할 수 있다. GPU 작업은 항상 동시에 하나다.

보수적 replicate 시간은 최근 정규 dev2개 runtime 최대값×1.25다. Smoke 하나뿐이면×1.5를 쓴다. Confirm20개가 allocation에 들어가려면 Shapes≤24분/replicate, MNIST≤15.75분/replicate여야 한다. 시간에는14 splits, untrained, raw, X, oracle, 평가·저장이 포함된다.

총 사용시간+모든 잔여 작업 forecast가20시간을 넘으면 confirmation 전에 BUDGET_BLOCKED로 중단한다. Reserve 이체는 확인 전에 기록한다. N,replicate 수,비교군,image size를 몰래 줄이지 않는다. Budget이 두 배가 됐어도 표본수24,000과20회를 자동 확대하지 않는다.

Ledger는 benchmark,phase,jobID,GPU model,시작/종료,allocated GPU wall time,billed time,실패·재시작을 기록한다. GPU가 idle해도 할당 job wall time에 포함한다. Queue 대기는 별도 표시한다. 각 job 시작 전 잔여 시간 예약과 hard timeout을 둔다.

CPU-only 대체의 wall-clock은 별도다. 실측 없이 하룻밤 완료를 약속하지 않는다. 20시간 초과, remote job, dependency 추가는 새 승인 사항이다.

## 14. 재사용과 공통 구현 backend

기존 자산을 활용하되 두 benchmark 알고리즘을 따로 만들지 않는다.

| 기존 파일·위치 | 재사용 |
|---|---|
| code/sim5_cnn_algorithm1.py:39,50 | AIPW와 graph aggregation |
| code/sim5_digit_proxy.py:198,214,692 | paired gap, digest, nuisance 초기화 매칭 |
| code/sim5_cnn_models.py:42 | 작은 CNN 구조 참고 |
| code/sim5_cnn_algorithm1.py:14 | J64 sampler는 새 J4 전수열거로 교체 |
| 기존 renderer·I/C 결합·frozen trunk | 재사용 금지 |

기존 AIPW/graph에 대한 empty/nonfinite/tie/boundary/chain/정답 점수6개 read-only 검사는2026-09-20에 통과했다. 새 backend 통합 뒤 다시 검사한다. 이전 SCM7 성능을 새 원본 이미지 실험 결과로 옮기지 않는다.

향후 제안 파일은 다음과 같다. 현재 존재·구현·실행을 뜻하지 않는다.

- code/multi_image_scm.py: SCM·finite evaluator·privileged-state 분리.
- code/multi_image_banks.py: 원본 source loader·kernel·collision audit.
- code/multi_image_probe.py: CNN·critics·학습·screen·nuisance·집계.
- code/run_multi_image_suite.py: 단계별 runner·seed·timing·protocol.
- code/score_multi_image_suite.py: raw cell만 읽는 독립 scorer.
- code/test_multi_image_suite.py: 회귀 검사.
- results/multi_image/mnist/<run_id>/: 승인 후 생성할 결과 위치.

## 15. 필수 검사, provenance, 결과 schema

필수 검사는 scalar 확률/overlap/ATE, source bytes/group 분리, kernel독립성, witness 가정, learner 정보 누출, row역할 교집합, held-out pixel 비접근, lifted critic 동일성, parameter 분리·실제 갱신, checkpoint tie/min_delta, NaN/Inf/SE, AIPW·graph 경계/동점/빈집합을 포함한다.

파일 누락·중복 cell·checksum 변경·protocol 변경·없는 checkpoint·미완료 manifest를 주입하면 scorer가 모두 거부해야 한다. Source image 반복은 실험 row 중복과 구별한다. A/Y warmstart나 channel label loss가 숨겨져 있으면 구현 FAIL이다.

필수 schema:

- run_id, phase, benchmark, complete, expected_cell_keys, failure_reason.
- source_sha, source_bytes, protocol_sha, criteria_sha, code_sha, scorer_sha, dirty_diff_sha.
- dataset_seed_manifest, algorithm_seed_manifest, row_role_digest.
- framework/device versions, optimizer constants, checkpoint paths/digests.
- per_split: mask,fit/select/audit risks/gaps/SE,selected step,delta,screen,overlap,ATE,score SD.
- per_method: estimate,error,return kind,component members,nuisance seeds/timing.
- gpu_ledger, elapsed CPU wall time, restart history.

원본 bank는 한 번 저장하고 replicate마다 source row index·bank hash를 저장한다. 이미지 전체 tensor를 반복 영구 저장하지 않는다. Audit 행별 d_i와 모든 split의 E AIPW score를 저장하여 재채점할 수 있게 한다.

Atomic shard 저장 후 모든 expected key와 digest가 검증돼야 complete=true다. Source·protocol이 달라진 결과를 같은 run으로 합치지 않는다. Scorer도 hash 봉인 대상이다. Confirmation summary를 신뢰하기 전에 raw cell에서 재계산한다.

## 16. Plot·table과 최종 보고

표1은 source shape,block 수,행 수,replicate 수,반환율,실제 compute다. 표2는5개 방법의 mean estimate,bias,MAE,medianAE,p90,paired UCB다. 표3은 모든 gate의 기준·실측·분모·판정이다.

그림1은 각 부품의 원본 이미지4장과 block 번호다. 역할 설명·latent 상태는 evaluator-only panel로 분리한다. 그림2는20개 dataset별 추정값 dot plot이며 참값1과 full-W reference1.291167을 표시한다. 그림3은 초기/final gap과 후보ATE, 최대성분 구성을 보여준다. 부록은 sensitivity,negative-gap,selected-step,compute ledger다.

Source bank에서 숫자0–9 모두를 사용한다. Replicate별 digit histogram과 같은 label 안에서 사용한 서로 다른 image ID 수를 보고한다. 두 숫자만 골라 난도를 낮추거나 MNIST test label로 CNN을 사전학습하지 않는다.

최종 보고는 P0 가정 검증, 구현 검증, fresh 성능, balance 기여를 따로 판정한다. 실패 원인을 특정하지 못했으면 추측과 확인된 사실을 구분한다. Budget 중단을 통계적 FAIL로, 성능 FAIL을 구현 성공으로 상쇄하지 않는다.

## 17. COA 실행 체크리스트

1. 두 PRD와 Sections2–4를 읽고 승인된 코드·출력 경로를 확인한다.
2. Source availability, SHA, dependency, 계정 quota를 read-only로 확인한다.
3. 외부 다운로드/upload/job마다 정확한 action 승인을 받는다.
4. 공통 backend를 구현하고 CPU 검사 및 P0를 통과시킨다.
5. Shapes smoke 시간으로 전체 forecast를 다시 계산한다.
6. Shapes dev3개를 채점하고 통과·예산 확인 뒤20개 confirmation을 봉인한다.
7. GPU 실행 중 MNIST는 CPU 준비만 한다.
8. Shapes 종료 뒤 MNIST smoke/dev/confirmation 순서를 따른다.
9. 독립 scorer로 raw 결과를 재계산하고 plot/table을 만든다.
10. 결과와 사용시간, 실패·미완료, 주장 가능한 범위를 보고한다. Git/mirror/원고 반영은 별도 승인이다.

향후 CLI 예시이며 현재 미구현·미실행이다:

    python code/run_multi_image_suite.py --benchmark mnist --phase p0 --protocol <approved_protocol>
    python code/run_multi_image_suite.py --benchmark mnist --phase smoke --protocol <approved_protocol>
    python code/run_multi_image_suite.py --benchmark mnist --phase dev --protocol <approved_protocol>
    python code/run_multi_image_suite.py --benchmark mnist --phase confirm --protocol <sealed_protocol>
    python code/score_multi_image_suite.py --benchmark mnist --protocol <sealed_protocol> --results <complete_manifest>

## 18. 근거와 변경 이력

- [기존 Shapes3D PRD](2026-09-18-shapes3d-original-image-prd-v1.md): scalar 구조·random-init 합의. 한 이미지 spatial partition에서 네 원본 이미지로 변경했다.
- [기존 CNN 개발 결과](2026-09-18-sim5-cnn-dev-result.md): raw .989,warm .956,final .937의 음성 결과. 새 성능으로 재사용하지 않는다.
- [기존 P0](2026-09-18-sim5-cnn-p0-feasibility.md): full-U 이미지 설계의 raw 비교 한계.
- [짝 PRD](2026-09-20-shapes3d-multi-image-prd-v1.md): 동일 backend·SCM·통계·공유예산을 사용한다.
- v1: 사용자가 요구한 생산라인 시나리오, Shapes선행, 공유20GPU-hour, 원본 다중이미지 사양을 문서화했다. 구현·학습·새 성능 수치는 아직 없다.
