# PRD: Tiny16 autoencoder 초기화 후 직접 CNN 균형학습

상태: **설계 문서만 작성 승인됨. 구현·실행·성공은 아직 확인하지 않았다.**

작성일: 2026-09-22. 대상: 구현 담당자와 실험 결과를 판단할 연구자.

이번 질문은 하나다. **이미지를 재구성하도록 먼저 배운 CNN에서 출발하면, 같은 균형 목적이 더 유용한 표현을 찾는가?** 초기화 외에는 같은 두 실험을 끝까지 비교한다. 나쁜 중간 성능을 이유로 중단하지 않으며, 수치 오류와 계산 시간 상한만 중단 사유다.

## 1. 산출물과 주장 범위

Shapes3D 자료 하나, 분할 S1 하나에서 다음 두 arm을 새로 실행하는 구현 계획이다.

| arm | 표현 CNN 초기값 | 이후 학습 |
|---|---|---|
| R | 무작위 초기값 | CNN 전체를 균형 목적으로 갱신 |
| AE | 라벨 없는 이미지 재구성으로 배운 초기값 | CNN 전체를 동일한 균형 목적으로 갱신 |

Autoencoder(AE)는 이미지를 작은 특징으로 압축한 뒤 원래 이미지를 복원하는 모형이다. 이번에는 압축하는 encoder만 초기값으로 가져오고 복원하는 decoder는 균형학습 전에 버린다. AE와 CNN은 대안 관계가 아니다. **CNN 구조를 autoencoder의 encoder로 먼저 학습한다.**

완료 산출물은 두 arm의 초기·선택·최종 표현, 독립 균형 통계, 인과효과 추정값, 실제 업데이트 기록, 시간·해시가 들어 있는 비교 보고서다. 두 arm 모두 선택이 끝나기 전에는 독립 검사와 효과 결과를 공개하거나 선택에 사용하지 않는다.

하지 않을 일:

- 새 SCM, MNIST, 전체 30분할, 다수결 집계, 확증 seed 확대.
- 각도·숫자·잠재변수 라벨을 이용한 초기화 또는 표현 선택.
- decoder 복원 손실이나 결과 예측 손실을 balancing에 혼합.
- 기존 v3 결과를 수정하거나 그것을 수정된 R arm의 대조로 대체.
- 기존 파일·봉인 결과 덮어쓰기, dependency 설치, DeltaAI 작업, Git 변경.

이는 여러 번 열어 본 동일 자료에서의 **개발 진단**이다. 새 자료에 대한 성능 확증, Algorithm 1 전체 실행, 정리의 문자 그대로 검증이라고 부르지 않는다.

## 2. 기존 기록에서 무엇을 고치는가

기존 v3는 두 가지를 그대로 비교 기준으로 사용할 수 없다.

1. CNN 업데이트 뒤 정규화 통계가 한 단계 늦게 critic에 전달되는 경로가 있다. 두 arm 모두 이 수명주기를 먼저 고친다.
2. 과거 감소 진단은 실제 업데이트에 사용한 critic이 아니라 새로 적합한 critic을 고정했다. 또한 실제 업데이트가 없는 구간까지 분모에 넣었다. 이번에는 실제 optimizer 업데이트만 구분한다.

따라서 R도 다시 실행한다. 과거 v3와 새 AE의 차이를 AE 효과라고 해석하지 않는다.

기존 64×64 AE는 복원은 잘했지만 고정 특징에서 각도를 선형으로 읽는 검사에서 실패했다. 기록된 선형 설명력은 0.675, 이차식 진단은 0.978이었다. 이는 **고정된 AE 특징 + 선형 판독**의 실패이지, AE 초기값에서 CNN 전체를 균형 목적으로 재학습하는 이번 절차의 실패를 뜻하지 않는다. 반대로 이번 절차의 성공 근거도 아니다.

그 64×64 encoder는 Tiny16 구조와 다르므로 가중치를 로드하지 않는다. 새 Tiny16 AE를 정해진 자료 안에서 학습한다. 각도 설명력을 초기화의 통과 관문으로 두지 않는다.

근거: [기존 AE 기록](2026-09-21-scm4e-v2-results-log.md), [v3 구현](../code/direct_balance_v3.py), [v3 실행기](../code/run_direct_balance_v3.py), [v3 저장 결과](../results/direct_balance/v3_ab/summary_v1.json).

## 3. 고정 자료와 정보 접근

| 항목 | 고정값 |
|---|---|
| benchmark / SCM | 기존 Shapes3D / 기존 SCM-1 이미지 생성 경로 |
| 자료 seed | 96900800 — 신규 확증 seed가 아님 |
| 분할 | S1, 코드의 split `(0,)` |
| 이미지 | 관측대상당 6장, 논리 블록 5개 |
| 남긴 이미지 | 코드 열 1,2,3,4,5 |
| 제외한 이미지 | 코드 열 0 |
| 해상도 | 원본 64×64를 4×4 영역 평균으로 16×16로 축소 |
| 참 평균 처치효과 | 1, 평가기에만 사용 |

기존 생성식, 15단계 관측화, 원본 이미지 선택, 행 역할을 바꾸지 않는다. 같은 seed만 믿지 말고 기존 산출물의 source-index 배열과 역할별 row-index 해시를 대조한다. 일치하지 않으면 다른 자료로 계속하지 않고 원인을 보고한다.

| 행 역할 | 행 수 | 사용 |
|---|---:|---|
| represent | 4,800 | AE 이미지 추출, CNN·critic 교대학습 |
| select | 1,200 | fresh critic으로 checkpoint 선택 |
| check | 6,000 | 선택 고정 후 독립 균형 검사 |
| nuisance | 6,000 | 최종 추정의 처치·결과 모형 적합 |
| evaluate | 12,000 | 효과 추정 및 점수 오차 계산 |
| 합계 | **30,000** | 역할 간 행 교집합 0 |

AE는 represent 행의 여섯 이미지 열에서 **고유 원본 이미지 ID**만 모은다. 정렬된 ID에 고정 목적 seed의 permutation을 적용하고 앞 floor(0.8×개수)를 AE-fit, 나머지를 AE-validation으로 둔다. 중복 ID가 양쪽에 들어가지 않아야 한다. 수치 좌표, 각도, A,Y,U,X는 AE loader가 반환하지 않는다.

각 고유 이미지는 epoch당 한 번 읽는다. 역할별 이미지 빈도를 학습 가중치로 사용하지 않는다. 원본 ID는 분리·재현 metadata이지 모형 입력이 아니다.

Tiny16 픽셀은 기존 `resize16`과 represent 행에서 정한 채널 평균·표준편차를 그대로 사용한다. 이 통계는 AE-validation의 라벨 없는 픽셀도 포함할 수 있으므로 AE-validation은 고정 전처리를 조건으로 한 복원 선택용이다. check/evaluate 자료로 전처리를 다시 맞추지 않는다.

독립 행끼리 같은 은행 이미지를 뽑을 수 있다. AE-fit/validation 간 ID 중복은 금지하지만, 인과자료의 다른 역할과 원본 ID가 겹치는 것은 별도 기록한다. 이번 결과를 처음 보는 이미지에 대한 일반화라고 부르지 않는다.

## 4. AE와 두 초기값을 만드는 방법

### 4.1 네트워크와 복원 학습

Encoder는 기존 Tiny16 `ImageTower`와 parameter-key까지 맞춘다.

- 입력: 표준화된 RGB 3×16×16.
- Conv2d(3,16,kernel=3,stride=2,padding=1), ReLU.
- Conv2d(16,32,kernel=3,stride=2,padding=1), ReLU.
- Flatten(512), Linear(512,32).
- Decoder: Linear(32,512), ReLU, reshape(32,4,4).
- ConvTranspose2d(32,16,kernel=4,stride=2,padding=1), ReLU.
- ConvTranspose2d(16,3,kernel=4,stride=2,padding=1), **identity output**.

픽셀 표준화 뒤에는 음수도 있으므로 출력에 sigmoid를 붙이지 않는다. BatchNorm, Dropout, augmentation, skip connection은 넣지 않는다.

복원 손실은 입력과 출력의 모든 픽셀에 대한 평균제곱오차다. Adam learning rate 0.001, 기본 betas (0.9,0.999), eps 1e-8, weight decay 0, batch 256, 최대 10epoch. 마지막 작은 batch도 포함한다. gradient clipping은 추가하지 않는다.

epoch 0과 완료된 각 epoch를 저장한다. validation 전체의 픽셀 평균제곱오차가 기존 최선보다 1e-8 이상 작을 때만 교체하고 동점은 빠른 epoch를 택한다. 시간 제한이면 완료된 epoch까지의 최선을 사용하고 10epoch 완료라고 쓰지 않는다. 최소 한 epoch도 끝나지 않으면 AE arm을 실행 가능한 초기화로 확보하지 못한 것으로 보고한다.

### 4.2 두 arm의 차이를 초기화 하나로 제한

표현 head는 retained 이미지별 32차원 특징 다섯 개와 X의 두 좌표를 받아 162→64→2로 계산한다. 마지막 출력은 X를 그대로 포함한 Z=(X,h)로 만든다.

- R의 각 tower는 무작위로 초기화한다.
- AE의 각 tower에는 선택된 하나의 AE encoder를 복사한다. 다섯 복사본은 독립 parameter storage를 가지며 이후 서로 다르게 갱신될 수 있다.
- 두 arm의 head는 동일한 원본 state를 복사한다.
- AE decoder는 균형 optimizer에 포함하지 않는다. 모든 표현 tower와 head는 `requires_grad=True`다.
- **기본·증강 critic은 두 arm 모두 무작위 초기화한다.** 표현 AE를 critic에 복사하지 않는다.

AE 학습의 RNG 소비가 R/AE critic을 다르게 만들지 않도록 전역 RNG 순서에 의존하지 않는다. 목적별 RNG를 별도로 생성한다. 권장 고정 규칙은 `SeedSequence([96900800, purpose_id, fold, iteration])`에서 uint32 seed 하나를 얻는 것이다. purpose_id는 AE-ID분리=101, AE초기화=102, AE배치=103, R-tower=201, 공통head=202, 공통critic=301, 공통critic배치=302, fold분리=303, 선택=401, 검사16=402, 검사64=403, nuisance=404, refit진단=501로 봉인한다.

공통 목적 seed에는 arm 이름을 넣지 않는다. 같은 iteration/fold의 초기 난수와 minibatch row 순서는 두 arm에서 같다. 단, 학습된 critic 가중치까지 계속 같아야 한다는 뜻은 아니다. 초기 state와 batch-index 해시로 구현을 검사한다.

## 5. 균형 목적과 고정 학습량

Z는 X와 학습한 두 좌표 h를 함께 담는다. 모집단 목표는 다음 잔여 처치 정보다.

$$
D_{\mathrm{res},S}^2(\phi)=\mathbb E\left[\{P(A=1\mid X,W_S,Z)-P(A=1\mid Z)\}^2\right].
$$

이 실험은 이를 직접 계산하지 않는다. 기본 critic q0(Z)와 증강 critic q1(Z,W_S)의 Brier 손실 차이를 근사로 쓴다. represent 행을 고정 3fold로 나누고 각 1,600행을 scoring에, 나머지 3,200행 중 2,560행을 critic-fit, 640행을 critic-validation에 쓴다. 이 역할은 100회 동안 고정한다.

$$
d_i=\{A_i-q_{0,-f(i)}(Z_i)\}^2-\{A_i-q_{1,-f(i)}(Z_i,W_{S,i})\}^2,
\qquad L(\theta)=\max\left\{\frac1{4800}\sum_i d_i,0\right\}.
$$

세 fold의 signed gap을 합친 뒤 **한 번만** clipping한다. fold별 clipping이나 minibatch별 clipping을 평균하지 않는다. pooled gap이 0 이하이면 optimizer step을 하지 않고 skip 사유를 저장한다. 작은 음수나 0은 모집단 균형 인증이 아니다.

기본 critic은 4→64→64→1 ReLU MLP, 출력은 0.1+0.8×sigmoid다. 증강 critic은 같은 Z branch와 held-out Tiny16 tower의 32차원 특징을 읽는 36→64→1 추가 branch로 구성한다. 두 logit을 더한다. 추가 branch의 마지막 weight와 bias를 0으로 두면 기본 critic을 정확히 재현한다. 표현 CNN과 critic CNN은 parameter를 공유하지 않는다.

| 설정 | 두 arm 공통 |
|---|---:|
| outer iteration | 계획 100 |
| critic 초기 적합 | 각 fold, 각 critic 최대 300step |
| 매 outer critic 재적합 | 각 fold, 각 critic 최대 100step |
| critic own-validation 검사 | step 0 및 매 10step |
| critic 선택 | 자기 Brier risk 최소, 개선량 1e-7, 동점 빠른 step |
| critic / CNN optimizer | Adam, lr 각각 0.001, weight decay 0 |
| critic batch / encoder 역전파 batch | 256 / 512 |
| checkpoint | 0,10,20,40,70,100 |
| CNN gradient | 전체 4,800행 목적에서 누적 후 1step |

Critic weights는 재적합 간 warm-start하되 Adam은 기존 v3와 같이 각 refit에서 새로 만든다. lifted-base 대안과 fitted augmented 중 선택도 고정 critic-validation risk로만 한다. lifted base가 선택되면 명시적으로 표시한다. 모든 seed·배치 규칙과 하이퍼파라미터를 protocol에 풀어 저장하여 import된 상수의 사후 변경에 영향받지 않게 한다.

이 cross-fitted 학습 목적은 원고 Section 4의 같은 표본 경험위험 최소값을 문자 그대로 계산한 것이 아니라 **모집단 잔여 불균형을 겨냥한 operational surrogate**다. 공유 encoder가 반복해서 A를 사용하므로 내부 scoring fold 전체가 적응적 학습으로부터 독립이라는 주장도 하지 않는다.

근거: [원고 3절](../manuscript/main/3.tex), [원고 4절](../manuscript/main/4.tex), [Tiny16 코드](../code/direct_balance_tiny16.py), [신경망 코드](../code/direct_balance_net.py).

## 6. 정규화 수명주기: 실행 전 반드시 고칠 부분

두 arm 모두 h의 좌표별 평균과 표준편차로 표준화한다. 통계는 represent 전체 4,800행에서 계산하고, 표준편차는 `unbiased=False`, 하한 0.001이다. X에는 기존 represent 기준 전처리만 적용한다. 행별 LayerNorm이나 차원을 줄이는 변환은 사용하지 않는다.

$$\bar h_\theta=(h_\theta-\mu_\theta)/s_\theta,\qquad Z_\theta=(X,\bar h_\theta).$$

CNN step에서 평균과 표준편차도 h의 함수로 미분한다. batch별 독립 정규화나 통계를 detach한 다른 목적을 쓰지 않는다. 전체 Z를 한 번 계산하여 dL/dZ를 얻고 CNN batch로 다시 역전파하는 기존 경로를 쓸 수 있다.

각 iteration의 순서를 고정한다.

1. 현재 encoder 상태 θt와 통계 nt가 critic이 사용한 좌표와 일치하는지 검사한다.
2. critic을 고정하고 **동적 통계가 포함된** 현재 목적을 미분한다.
3. pooled gap이 양수이면 CNN optimizer를 한 번 갱신한다.
4. **갱신 후 encoder θt+1로 전체 represent의 통계 nt+1을 다시 계산한다.**
5. 다음 critic 적합 전에 기존 critic의 입력층을 nt→nt+1 좌표로 옮긴다.
6. post-step encoder와 nt+1을 사용하여 critic을 재적합한다.
7. encoder, critic, 통계가 같은 시점임을 검사하고 checkpoint를 저장한다.

4번에서 pre-step 통계를 반환하거나 다음 iteration에서만 통계를 갱신하면 안 된다. skip이면 θ와 통계는 같아야 한다. 평가 wrapper, fresh selection, 독립 check, nuisance가 모두 저장된 같은 Z를 읽어야 한다.

좌표 운반은 원시 h에 대한 critic 함수를 보존한다. h 열에 해당하는 첫 층에서:

$$W_{\rm new}=W_{\rm old}\operatorname{diag}(s_{\rm new}/s_{\rm old}),\quad b_{\rm new}=b_{\rm old}+W_{\rm old}\left[(\mu_{\rm new}-\mu_{\rm old})/s_{\rm old}\right].$$

마지막 식의 나눗셈은 좌표별이다. bias 계산에는 변경 전 W를 쓴다. 기본 critic, 증강 Z branch, 증강 extra branch의 Z 입력 열을 모두 변환하며 픽셀 특징 열과 X 열은 건드리지 않는다. critic optimizer를 refit마다 새로 만들므로 이 변환에 이전 Adam moment를 잘못 이어 붙이지 않는다.

## 7. 실제 업데이트가 무엇을 했는지 검사

매 outer iteration마다 raw/clip gap, fold별 두 risk, 실제 step 여부, 모든 encoder layer의 gradient norm과 parameter 변화량, h 평균·표준편차, critic 선택 step, lifted-base 선택 여부를 남긴다.

실제 CNN optimizer 업데이트의 순서 번호 **1,3,7**에서만 상세 상태를 저장한다. outer iteration 번호와 구별한다. 일곱 번 미만 움직이면 없는 진단은 `not_reached`다. no-op을 감소 실패 1건으로 세지 않는다.

저장 내용: pre/post encoder, **실제 학습에 사용한** 두 critic의 각 fold state, CNN Adam pre/post state, critic 적합 종료 optimizer metadata, fold indices, 배치 목적 seed, pre/post 정규화 통계, 실제 행별 손실 차이.

### 7.1 실제 critic을 고정한 국소 변화

가능하면 모든 실제 step에 대해, 해당 step의 critic을 그대로 고정하고 전후 목적을 계산한다. 전후 각각의 encoder에서 μθ와 sθ를 다시 계산한다. 이것이 실제 gradient가 미분한 동일 함수다.

**이 비교 중에는 critic을 원시 h 좌표에 맞춰 재조정하지 않는다.** 그런 운반을 하면 전후 서로 다른 함수를 비교한다. 6절의 운반은 이 진단을 끝낸 뒤 다음 critic refit을 준비하는 별도 동작이다.

raw signed gap과 clipped gap을 모두 저장한다. Adam 한 번이 언제나 목적을 감소시킨다고 보장하지 않는다. 증가가 있으면 증가로 보고하며 학습률을 바꾸는 새 실험을 자동으로 시작하지 않는다.

### 7.2 새로 적합하면 감소가 유지되는가

실제 업데이트 1,3,7의 전후 표현에 각각 **새로운 critic**을 같은 초기 seed·행·300step·validation 규칙으로 적합한다. 이는 7.1과 다른 질문이므로 열을 분리한다. 가능한 rowwise gap을 함께 보관한다. 이 진단의 결과로 checkpoint를 바꾸지 않는다.

시간 부족이면 이 오프라인 진단부터 미완료로 남긴다. 실제 수행하지 않은 항목을 0 또는 성공으로 채우지 않는다.

## 8. 선택과 평가: 먼저 잠그고 나중에 본다

### 8.1 Checkpoint 선택

각 저장 checkpoint에서 표현을 고정하고 represent 행에 fresh critic을 맞춰 select 1,200행에서 평가한다. 기존 fresh evaluator의 사양을 고정한다: 기본·증강 각각 300step, batch256, lr0.001, fit의20% own-validation, 매25step 확인, 개선량1e-7, 빠른 동점. checkpoint마다 같은 목적 seed로 초기화한다.

선택 기준은 select의 **clipped gap 최소**, 동점은 빠른 checkpoint다. 비교 허용오차는 1e-6으로 두고 새 값이 기존 값보다 그 이상 작을 때만 바꾼다. 더 큰 음수를 보상하지 않는다. ATE·U 설명력·독립 check로 선택하지 않는다.

**두 arm 모두의 선택 checkpoint와 해시를 잠근 뒤에만** 초기·선택·최종의 check와 ATE를 계산한다. AE의 balancing 전 checkpoint0도 이 시점에 평가한다.

### 8.2 항상 보고할 세 상태

- 초기0: R은 random, AE는 복원 사전학습만 완료한 상태.
- 선택: fresh select gap이 고른 상태.
- 최종: 계획100 완료 시100; 시간 상한이면 마지막 완료 iteration을 명시.

동일 encoder·통계 해시이면 평가를 재사용하고 동일 상태임을 표시한다. 선택0이어도 마지막 상태를 숨기지 않는다.

독립 check는 기존6,000행 반반 교차 적합을 유지한다. 각 절반3,000 중 critic-fit2,400/validation600, 다른 절반 scoring3,000이다. Tiny16 held-out critic과 원본64 held-out critic을 별도로 새로 맞춘다. 64 검사는 표현 입력을64로 바꾸는 것이 아니라 **같은 Tiny16 표현을 더 풍부한 held-out 픽셀로 검사하는 진단**이다.

기존 CHECK 사양(각300step, 매25step, batch256, bounded link)을 그대로 쓰고 각 fold의 fitted-base와 augmented step을 저장한다. exact lifted-base 선택으로 gap=SE=0이면 그 사실을 표시하며 완전한 균형 인증이라고 읽지 않는다.

ATE는 기존 nuisance MLP64/64, 최대200epoch, patience15, batch256, lr0.001, weight decay0.0001, nuisance 내부 validation20%를 재사용한다. 모든 상태·arm에 같은 seed와 행을 쓴다. 성향점수 clipping [0.05,0.95]와 clipping 전 범위 밖 비율을 저장한다. evaluate12,000행은 적합·선택에 사용하지 않는다.

## 9. 통계와 결과 해석

행별 gap di에서 `gap=mean(di)`, `SE=sd(di,ddof=1)/sqrt(n)`를 계산한다. 기존 화면 문턱과 비교하기 위한 descriptive upper는:

$$\max(\widehat{gap},0)+\Phi^{-1}(1-0.05/30)\widehat{SE}.$$

이번에는 S1 하나만 보지만 이전 진단과 연결하기 위해 30 기준을 유지한다. upper≤0.0015 여부는 **표시만** 하며 학습을 중단시키지 않는다. 교차적합 nuisance 공유와 반복 개발을 무시한 정확한 유한표본 신뢰보증이라고 부르지 않는다.

효과 추정값은 evaluate의 AIPW 점수 평균이다. 점수 배열을 지우기 전에 `SE=sd(scores,ddof=1)/sqrt(12000)`를 저장한다. 참값1과의 차이는 **절대오차**다. 자료 하나이므로 MAE라고 부르지 않는다.

전후 비교는 같은 행의 점수 또는 gap 차이를 먼저 만들고 그 차이의 SE를 계산한다. 정규근사 구간은 차이평균±1.96×SE로 제시하되, 이 자료와 적합된 모형을 조건으로 한 진단이지 새 자료들에서 AE가 우월하다는 신뢰구간이 아니다. checkpoint를 독립 반복으로 세지 않는다.

선택을 잠근 후 선택적으로 U의 선형 설명력을 기록할 수 있다. 입력은 **Z=(X,h)** 전체이고 nuisance에서 회귀를 맞춰 evaluate에서 R²를 계산한다. h만의 수치를 Z의 설명력이라고 부르지 않으며 선택·중단에 사용하지 않는다.

| 관측된 결과 | 결론과 다음 판단 |
|---|---|
| AE 초기부터 좋고 balancing 후 개선 없음 | 재구성 초기화의 효과. balancing 기여는 확인되지 않음 |
| AE 초기→선택에서 독립 gap·효과 절대오차가 모두 감소 | 이 자료에서 balancing의 유용한 변화 관측 |
| gap만 감소, 효과 절대오차 불변/증가 | 균형 통계 개선과 인과 정확도 개선을 분리하여 보고 |
| AE 선택이 R 선택보다 좋음 | 같은 예산에서 초기화 차이에 따른 개발 증거, 일반적 우월성 아님 |
| 선택 절대오차≤0.15, upper≤0.0015 | 사전 지정 진단 표식 두 개 만족, 확증 PASS 아님 |
| 둘 다 실패하거나 미완료 | 수치를 그대로 보고. 모델/자료/관문 자동 변경 없음 |

낮거나 음수인 경험적 gap만으로 조건부 독립·outcome-relevant completeness·식별성을 주장하지 않는다. 이번 실험에는 후보 집계가 없다.

## 10. 계산 순서와 60분 상한

구현·단위검사 준비 시간과 실제 실험 계산 시간을 구분한다. 구현과 리뷰는 약45–90분을 예상하되 약속된 완료시간이 아니다. **실험 계산은 로컬 Mac 한 프로세스, 단조시계 기준60분 상한**이며 AE, 선택·평가, 결과 직렬화까지 포함한다. 단위검사는 실행 전 단계지만 소요시간을 따로 공개한다.

| 계산 항목 | 상한 |
|---|---:|
| 자료·AE 준비 및 AE 학습 | 10분 |
| R 주 학습 | 12분 |
| AE 주 학습 | 12분 |
| 두 arm 선택·독립 검사·효과 평가·저장 | 16분 |
| 실제 업데이트 오프라인 refit 진단 | 10분 |
| 합계 | **60분** |

기존 v3는 약43분이었지만 새 AE와 수정된 경로가60분에 끝난다는 보장은 없다. 상한은 예상 소요시간이 아니다. 순서는 데이터/AE→R학습→AE학습→두 선택 고정→양쪽 평가→오프라인 진단이다. 통계가 나쁘다는 이유로 B,C 단계를 건너뛰지 않는다.

각 학습은 계획100회와 실제 완료 iteration을 함께 저장한다. arm별12분이 가까워지면 마지막 완결 iteration을 저장하고 평가 시간을 남긴다. 100회 완료가 아니면 `partial_time_cap`이다. ATE가 좋다고 완결로 바꾸지 않는다.

deadline은 내부 critic 루프·AE batch·CNN batch·선택·audit·nuisance 루프에서도 확인한다. 최종 저장을 위한 여유를 확보한다. 시작 전에 디스크 여유를 검사하고 부족하면 승인 없는 파일 삭제나 다른 디렉토리 이동 없이 중단한다. 모든 checkpoint를 중복 저장하지 말고 평가·실제 업데이트 진단에 필요한 상태만 남긴다.

새 GPU 작업, 병렬 실행, 설치, hyperparameter 탐색은 없다. 수치 오류, 파일 무결성 오류, 시간·디스크 한계로 못 끝난 부분은 명시적으로 남긴다. 다른 arm의 아직 가능한 평가까지 자동 폐기하지 않는다.

## 11. 구현 경계·검사·재현 기록

아래는 **향후 구현 시 생성할 경로 제안**이며 현재 존재하거나 승인된 실행 결과라는 뜻이 아니다.

- `code/direct_balance_ae_warmstart.py`: AE, paired initialization, 수정된 norm lifecycle, actual-step 진단.
- `code/run_direct_balance_ae_warmstart.py`: 단일 자료 두 arm, deadline, 선택 lock, 평가·기록.
- `code/test_direct_balance_ae_warmstart.py`: 최소 회귀검사.
- `results/direct_balance/ae_warmstart_v1/`: 새 protocol·상태·행별 결과.

기존 `direct_balance_net.py`, `direct_balance_v3.py`, `direct_balance_tiny16.py`, `direct_balance_eval.py`, 데이터와 family 코드들은 읽고 재사용하되 수정하지 않는다. 결함 수정은 새 모듈에서 명시적으로 override하고 source provenance에 기록한다. 기존 결과가 존재하는 경로는 write-once(O_EXCL)로 보호한다. 문서 승인만으로 위 파일 생성·실행까지 승인된 것은 아니다.

필수 구현 검사:

1. 30,000행 역할 분리와 기존 row/source-index 해시 동일성.
2. AE의 고유 ID fit/validation 무중복, represent 밖 ID 접근 금지.
3. AE loader에 A,Y,U,X·각도 라벨 없음; 표현 forward에 held-out 픽셀 없음.
4. Tiny16 encoder와 AE encoder state-key·shape 일치, decoder 출력 shape·finite.
5. AE tower 복사본 parameter storage 독립, 두 arm head 초기 해시 동일.
6. AE RNG 소비 전후 공통 critic 초기 해시·minibatch 인덱스 동일.
7. 전체 gap clipping과 작은 직접 계산의 값·gradient 일치; per-fold clipping 미사용.
8. 동적 mean/std를 포함한 gradient의 작은 float64 수치미분 검사.
9. post-step 통계가 저장 통계와 일치; 다음 critic 입력도 해당 통계 사용.
10. 모든 Z 입력층 운반 전후 raw-h prediction 동일성; skip 시 통계 불변.
11. 실제 step ordinal1/3/7만 진단, no-op은 NA; 실제 critic state 해시 일치.
12. 두 arm 선택 lock 전 check/evaluate 함수 호출 금지; 0/선택/최종 출력 누락 금지.
13. NaN·timeout·불완전 파일·digest 불일치 fail-closed 및 deadline 전파.

`protocol.json`에는 PRD·코드·의존 source SHA256, 기존 입력 해시, seed 목적표, image-ID 분할 해시, 전처리 통계, 모델·optimizer 상수, 기기·버전·시간 상한을 저장한다. 실행 전후 기존 읽기 전용 source/result 해시가 바뀌지 않았는지 확인한다.

`ae.json`에는 ID 개수, epoch별 train/val MSE, 선택 epoch, state 해시와 시간을 기록한다. `arm_R/AE`에는 모든 iteration history, actual-update count, checkpoint와 통계, 선택 표, lock 시각, status를 기록한다. `evaluation`에는 각 상태의 두 해상도 check, raw risks, fitted step, rowwise gap, AIPW scores·SE·절대오차를 저장한다. `diagnostics`에는 실제/새로 적합한 critic 결과를 다른 key로 분리한다.

`summary.json`은 complete/partial/failed 이유, 계획·실제 iteration, 두 선택 lock, 재사용한 동일 상태, 시간 원장, 입력 보존 여부를 포함한다. summary만으로 원시 행별 수치를 버리지 않는다. 단위검사 결과는 실제 exit code와 원문 output을 저장하며 문자열 파싱 실패를 PASS로 바꾸지 않는다.

## 12. 최종 보고와 완료 조건

핵심 표는 여섯 줄이다: R/AE × 초기/선택/최종. 열은 checkpoint, 실제 누적 업데이트, Tiny16 gap±SE와 upper, 64 gap±SE와 upper, 효과 추정±점수SE, 절대오차, 시간, 완료 상태다. AE 초기 결과를 별도 표시하여 사전학습과 balancing의 기여를 구분한다.

보조 그림은 두 개면 충분하다.

1. iteration에 따른 raw/clip gap, q0/q1 risk, 실제 step 표시와 h의 좌표별 표준편차.
2. 각 arm의 초기→선택→최종 독립 gap과 효과 추정의 짝지은 변화. 참값1을 표시하되 그 선으로 checkpoint를 선택하지 않는다.

실제 업데이트 진단은 actual ordinal, outer iteration, 실제 고정 critic의 raw/clip 변화, fresh-refit 변화, 수행/미도달/시간부족을 한 표로 공개한다. `0/5 성공`처럼 no-op을 실험으로 세는 요약은 금지한다.

문서의 구현 완료 조건은 코드와 위 회귀검사를 통과하고 자료·정규화·초기값 통제가 입증된 것이다. **실험 완결 조건**은 두 arm 각각100회와 정해진 선택·초기/선택/최종 평가를 모두 마친 것이다. 시간 상한으로 못 마치면 완료된 결과를 유용하게 보고하되 완결이라고 주장하지 않는다.

판단의 중심은 다음 문장이다.

> 라벨 없이 이미지를 재구성한 초기값이, 동일하고 수정된 균형학습에서 무작위 초기값보다 나은 출발점이었는가? 그리고 그 출발점에서 실제 균형학습이 독립 불균형과 인과효과 오차를 추가로 줄였는가?

답이 아니어도 계획한 수치를 끝까지 보고한다. 답이 좋아도 이 한 자료로 원고 정리나 전체 이미지 방법의 성공을 선언하지 않는다.
