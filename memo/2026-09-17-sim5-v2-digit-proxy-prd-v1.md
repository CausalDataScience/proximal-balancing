# Sim-5 v2 PRD: 역할을 지정하지 않고 숫자 이미지에서 조정 표현 학습하기

Created: 2026-09-17 CDT
Revision: **3, 2026-09-17 CDT. 기존 파일을 개정하며 파일명 v1은 경로 연속성을 위해 유지한다.**
Status: **구현 규약. Revision 3은 작은 고정-split smoke의 구현·실행만 승인하며 full-30과 confirmation은 승인하지 않는다.**
Supersedes: 이 파일의 최초 사양인 support ≤ 2 library, probit 선택/MLP 심사 혼용, 정확한 후보 census·정리 하한·certificate 검증.
기존 SCM-7-v4 결과와 기존 positive-control 정리 감사는 별도 증거로 보존한다.

## 0. 만들 결과와 최소 범위

**목표:** 관측된 이미지의 영역별 역할과 숨은 숫자 정답을 주지 않고 조정 표현을 배우며, 같은 자료에서 균형 학습 전후와 전체 이미지 조정을 비교한다.
최종 산출물은 독립 자료 15개에서의 효과 오차·반환율·비교군 표, 초기 표현 대 최종 표현 그림, 30개 split 검사 기록이다.
Revision 2에서 독립 여백 블록을 사분면에 병합하고 평가행과 head 학습량을 늘렸다.
실행자는 이 PRD로 구현하고, 용한은 작은 개발 실험 결과를 보고 이미지 실험을 본문에 넣을지 판단한다.
성공은 실행 횟수가 아니라 10절의 구현·성능·비교 판정을 각각 계산하고 실패까지 재현 가능하게 보존하는 것이다.

**최소 경로:** 픽셀 특징 학습 → 남긴 모든 특징을 결합한 작은 head → Brier 균형 학습 → 독립 검사 → AIPW → 기존 그래프 집계.
새 DGP 격자, 대형 CNN 탐색, root recovery, 완벽한 숫자 분류, 새로운 정리 감사는 이 실험의 완료 조건이 아니다.
새 GPU 작업이나 dependency를 추가하지 않는다. CPU 실측 예산은 8절에서 정한다.

원고 대응은 Section 3의 근사 균형과 Section 4의 표현 학습·정직한 추정·집계다.
신경망의 전역 최적화 오차, 안정성 상수와 nuisance 오차를 인증하지 않으므로 이미지 결과에 정리의 수치 하한이나 certificate를 붙이지 않는다.
Algorithm 1의 단계 구조를 따르는 **경험적 이미지 구현**이라고 보고한다. 문자 그대로의 정리 검증은 기존 positive-control이 담당한다.

## 1. 관측과 생성식

관측 자료는 $(X,W,A,Y)$뿐이다. 잠재벡터 $U=(U_{\mathrm c},U_{\mathrm t})$의 두 성분은 모두 미관측이다.
$U_{\mathrm c}$는 숫자, $U_{\mathrm t}$는 처치 표식의 원인이다. 아래 계수와 잠재변수는 생성기와 oracle 평가자만 받는다.

$$U_{\mathrm c}\sim\mathrm{Unif}\{1,\ldots,9\},\quad U_{\mathrm t},\eta_X,\varepsilon_Y\sim N(0,1)\quad\text{서로 독립}$$
$$X=0.5(U_{\mathrm c}-5)+2\eta_X$$
$$P(A=1\mid U,X)=0.1+0.8\Phi\{0.5(U_{\mathrm c}-5)+1.5U_{\mathrm t}+0.2X\}$$
$$Y=A-0.8(U_{\mathrm c}-5)+0.2X+0.3\varepsilon_Y,\qquad \tau=\mathbb E[Y(1)-Y(0)]=1.$$

$\Phi$는 표준정규 누적분포함수다. $A$는 위 확률의 Bernoulli 추출이다.
숫자를 환자 중증도의 비유로만 설명하며 실제 의료 데이터라고 쓰지 않는다.
별도 관측 $C$나 $I$ 좌표를 전달하지 않는다. 과거의 $I$는 이 문서에서는 $U_{\mathrm t}$다.
숫자 정답, 잠재 propensity, 생성 계수로 encoder·checkpoint·split·문턱을 선택하지 않는다.
잠재 정답은 실행이 끝난 후 진단과 oracle 비교에만 사용할 수 있다.

## 2. 이미지와 역할을 모르는 입력

MNIST 학습 bank의 숫자 1–9에서 해당 $U_{\mathrm c}$의 견본을 복원추출한다.
bank 경로는 materials/benchmarks/mnist/mnist.npz이며 실행 시 **전체** SHA-256을 protocol에 기록한다.
견본에 회전 $\mathrm{Unif}(-20^\circ,20^\circ)$와 크기 $\mathrm{Unif}(0.9,1.1)$를 독립 적용한다.
쌍선형 보간한 $28\times28$ 이미지를 $36\times36$ 중앙에 놓는다.
표식 16픽셀을 $0.5+0.4\tanh(U_{\mathrm t}/2)$로 채운 후 전체에 독립 $N(0,0.05^2)$ 센서잡음을 더하고 $[0,1]$로 자른다.

| 고정 index | 생성기에서만 쓰는 이름 | 행·열, 0-based inclusive | 크기 |
|---|---|---|---:|
| 0 | $Q_1$ | 0–17, 0–17 | 324 |
| 1 | $Q_2$ | 0–17, 18–35 | 324 |
| 2 | $Q_3$ | 18–35, 0–17 | 324 |
| 3 | $Q_4$ | 18–35, 18–35에서 표식 제외 | 308 |
| 4 | $M$ | 32–35, 32–35 | 16 |

$3\cdot324+308+16=1296$이며 블록은 겹치지 않고 모든 픽셀을 덮는다.
원래의 독립 잡음 블록 R을 네 사분면에 병합했으며 이미지 생성식은 바꾸지 않는다.
학습기 입력은 X, block arrays, A, Y와 anonymous index뿐이다. 역할별 분기나 특수 scalar 추출을 금지한다.
입력 차원 때문에 layer 폭의 첫 항은 달라도 모든 블록의 코드 경로와 손실은 같다.
모든 비어 있지 않은 proper held-out 집합 S를 검토하므로 $2^5-2=30$개다.
표식 M 포함 집합은 15개이고 나머지는 15개다. 이 수는 열거 검사에만 사용한다.
어떤 후보가 균형을 이루거나 집계에서 제외되는지는 사전 확정하지 않는다.
독립 잡음 후보를 제거해도 근사 표현·학습 오차 때문에 잘못된 후보가 생길 수 있으므로 집계는 유지한다.

## 3. 표본 분리: 먼저 나누고 그 뒤 학습한다

한 독립 데이터셋은 총 $n=48,000$행이다. 행 ID와 생성 seed를 먼저 저장한다.

| 행 집합 | 크기 | 허용된 사용 |
|---|---:|---|
| $D_{\mathrm{fit}}$ | 2,400 | 전처리·encoder·fusion·critic 파라미터 적합 |
| $D_{\mathrm{select}}$ | 800 | representation checkpoint와 restart 순위 |
| $D_{\mathrm{audit}}$ | 800 | 최종 선택 표현을 한 번 검사 |
| $N$ | 4,000 | 고정된 표현의 nuisance 적합 |
| $E$ | 40,000 | 효과 점수와 비교군 평가 |

앞의 세 집합을 합쳐 원고의 표현 학습 자료 $D=4,000$행으로 본다.
$D_{\mathrm{fit}}$ 안의 1,920/480행을 고정해 optimizer 적합/early stopping에 쓴다.
아래에서 “fit/validation”은 이 1,920/480행만 뜻한다. 최종 재적합도 같은 예산과 분할을 사용한다.
X·Y 표준화, 픽셀별 표준화가 필요하면 그 평균·분산은 $D_{\mathrm{fit}}$에서만 추정한다.
픽셀 기본 입력은 이미 $[0,1]$이므로 추가 픽셀 표준화는 하지 않는다.
$D_{\mathrm{select}}$·$D_{\mathrm{audit}}$·$N$·$E$에서 표현 또는 critic을 갱신하지 않는다.
모든 baseline도 같은 outer D/N/E 행을 쓴다. 순서를 바꿔도 행 교집합은 항상 0이어야 한다.
어떤 train fold의 feature를 fit 전에 전체 자료로 미리 학습하는 경로를 금지한다.

## 4. 하나의 표현 구조

### 4.1 블록 특징과 초기화

블록별 MLP $g_j:\mathbb R^{p_j}\to\mathbb R^8$는 $p_j\to64\to64\to8$, SiLU다.
각 블록의 특징으로 처치와 결과를 함께 예측해 초기화한다.
처치 head는 $(X,g_j(W_j))$, 결과 head는 $(A,X,g_j(W_j))$를 읽는 폭 32 MLP다.
결과는 $D_{\mathrm{fit}}$의 평균·표준편차로 표준화한 $\widetilde Y$를 쓴다.

$$L_{\mathrm{warm}}=\operatorname{BCE}(A,\widehat e)+(\widetilde Y-\widehat m(A,X,g_j(W_j)))^2.$$

두 항의 가중치는 1이다. Adam $10^{-3}$, weight decay $10^{-4}$, batch 128,
최대 40 epoch, fit 내부 validation의 위 손실로 patience 5, min_delta $10^{-4}$다.
한 데이터셋에서 다섯 블록을 한 번씩 학습한 뒤 고정한다.
숫자 분류 정확도로 epoch나 architecture를 선택하지 않는다.

### 4.2 모든 남긴 블록을 결합

split $S$마다 $f_{-S}=\operatorname{concat}\{g_j(W_j):j\notin S\}$를 만들고
$h_S$는 $(X,f_{-S})\to64\to32\to8$의 SiLU MLP로 둔다.
최종 표현은 $Z_S=(X,h_S(X,f_{-S}))$의 9차원이다. X는 별도 bypass로 보존한다.
남긴 특징을 모두 입력하므로 support ≤ 2 library를 사용하지 않는다.
추론 시 $h_S$나 $g_j$의 입력에 A, Y, 숨은 숫자·표식 값이 들어갈 수 없다.

split별 head restart는 2개로 고정한다. 각 restart에서 같은 A/Y 손실로 head를 먼저 초기화한다.
head warmstart 예산도 최대 40 epoch, patience 5다. 블록 encoder는 이 단계부터 고정한다.
각 restart의 warmstart checkpoint를 digest와 함께 저장하고 균형 학습의 첫 후보(epoch 0)로 포함한다.
마지막에 선택된 restart의 **바로 그 초기 checkpoint**가 “균형 전” 비교 대상이다.
새로 뽑은 초기값이나 다른 restart의 좋은 초기값으로 바꾸지 않는다.

## 5. 같은 Brier 목적을 학습·선택·검사에 사용한다

표현 $Z$만으로 처치를 예측하는 base critic을 $q_0(Z)$,
$Z$와 held-out 원 픽셀을 함께 읽는 augmented critic을 $q_1(Z,W_S)$라고 한다.
X가 Z에 포함되므로 원고의 $(X,W_S,Z)$ 입력이 확보된다.

$q_0$는 $9\to64\to64\to1$ MLP다.
$q_1$는 독립 base trunk와 $(Z,W_S)\to64\to64\to1$ residual logit branch를 가진다.
두 출력은 $0.05+0.90\operatorname{sigmoid}(\text{logit})$다.
residual의 마지막 층을 0으로 시작하면 base 함수가 augmented class 안에 들어간다.
base와 augmented critic은 **파라미터를 공유하지 않고 각각 자기 Brier 손실로 적합**한다.
lifted start는 포함관계와 초기화 장치이며 empirical/global minimum 인증은 아니다.

같은 평가행 $i$에서 $d_i=(A_i-q_0(Z_i))^2-(A_i-q_1(Z_i,W_{S,i}))^2$를 저장한다.
signed gap은 $\bar d$, clipped 목적은 $\widetilde D^2=\max(\bar d,0)$다.
paired SE는 $\operatorname{sd}(d_i)/\sqrt n$다. 음수 gap도 원본 그대로 보존한다.

### 5.1 균형 학습

warmstart 후 블록 encoder를 고정하고 fusion head만 갱신한다.
각 restart에서 fusion head를 정확히 100회 갱신한다. update 0에서 critic을 적합하고 이후 매 5번째 head update 직전에 critic을 새로 적합한다. 각 refresh에서:
1. fusion head를 고정하고 두 critic을 자기 Brier 위험으로 최대 20 epoch 적합한다.
2. critic 파라미터를 고정하고 $D_{\mathrm{fit}}$의 optimizer-fit 1,920행 **전체 gap**을 계산한다.
3. 그 전체 gap을 clip한 목적에 대해 fusion head를 Adam $10^{-4}$로 정확히 1회 갱신한다.

각 critic의 Adam은 $10^{-3}$, batch 128, weight decay $10^{-4}$, validation patience 3이다.
encoder의 clip은 minibatch별로 적용한 뒤 평균하지 않는다. 메모리 때문에 누적하면 signed 전체 mean을 만든 뒤 한 번 clip한다.
checkpoint는 실제 head update가 0, 25, 50, 100에 도달한 직후에만 저장한다. warmstart를 update 0으로 포함해 총 8개(2 restart × 4개)다. 저장 record의 update 수와 optimizer counter가 다르면 구현 실패이며 중간 값을 사후 보간해 checkpoint로 만들지 않는다.
실제 head/critic optimizer 갱신 횟수, outer iteration, zero-clipped 횟수와 파라미터 이동량을 저장한다.
0회는 warmstart다. 중간 종료로 없는 checkpoint를 만들어 보고하지 않는다.
critic이 충분히 적합됐다는 보장은 없으며, optimization residual을 인증했다고 보고하지 않는다.

### 5.2 checkpoint 비교와 최종 검사

저장 checkpoint마다 critic을 **같은 class·예산·fit/validation 행·2개 초기화 seed**로 새로 적합한다.
각 critic은 fit 내부 validation Brier가 낮은 초기화를 고른다.
그 예측으로 동일한 $D_{\mathrm{select}}$행의 gap을 구해 representation 후보를 비교한다.
예산은 critic당 최대 40 epoch, patience 5로 고정한다. epoch 0에도 동일하게 쓴다.

선택 순위는 clipped gap, $10^{-6}$ 이내 동점이면 더 이른 균형 반복, 그다음 더 작은 restart ID다.
AIPW 추정값, oracle 오차, 숫자 분류, 생성식 정보는 순위에 사용하지 않는다.
선택된 표현에 최종 critic을 새 seed의 2개 초기화로 같은 예산에서 재적합하고
$D_{\mathrm{audit}}$의 두 위험·signed gap·paired SE·clipped gap을 **한 번** 저장한다.
audit 결과로 다른 checkpoint로 돌아가지 않는다. 실패하면 해당 split이 실패한다.
형태가 같은 empirical Brier 목적을 쓰지만 평가행 분리와 유한 optimizer를 쓰는 실행 변형이다.

### 5.3 screen과 신뢰성 진단

유한 값 확인 후 $\bar d<-2\,\mathrm{SE}$이면 critic reliability flag를 켜고 그 split을 불확실로 탈락시킨다.
이것은 음수 clip만으로 자동 통과하는 것을 막는 **경험적 guard**이며 통계적 균형 보증은 아니다.
나머지는 $\max(\bar d,0)\le t$이면 유지한다.
구조적 overlap 상수는 $\eta=0.05$다. critic과 nuisance propensity의 실제 예측에 대해 최소·최대와 $[0.1,0.9]$ 밖 비율을 모두 저장한다. $[0.1,0.9]$ 밖 비율은 진단이며 탈락 규칙이 아니다. $[0.05,0.95]$ 밖 값 또는 비유한 값은 구현 실패다.
critic 적합 오차는 paired SE에 포함되지 않는다. 낮은 gap을 참 discrepancy의 상계라고 부르지 않는다.
표식 split 전부 탈락, 잡음 split 반드시 X 선택, 사분면 split 전부 유효 같은 규칙은 사용하지 않는다.

## 6. 효과 추정과 집계

유지된 표현을 고정하고 N 안의 고정 80/20분할에서 propensity·outcome nuisance를 적합한다.
propensity는 $Z\to64\to64\to1$, outcome은 $(A,Z)\to64\to64\to1$ MLP다.
propensity 출력은 $0.05+0.90\operatorname{sigmoid}$, outcome은 선형 출력이다.
Adam $10^{-3}$, batch 128, weight decay $10^{-4}$, 최대 60 epoch, patience 8.
outcome은 학습 시 표준화하고 AIPW 계산 전에 원 단위로 돌린다.
모든 방법이 동일한 nuisance 적합 예산과 N/E 행을 쓴다.

$$\widehat\theta_S=\frac1{|E|}\sum_{i\in E}\left[
\widehat m_1(Z_i)-\widehat m_0(Z_i)
+\frac{A_i(Y_i-\widehat m_1(Z_i))}{\widehat e(Z_i)}
-\frac{(1-A_i)(Y_i-\widehat m_0(Z_i))}{1-\widehat e(Z_i)}
\right].$$

유지 수 $K$에 대해 $\rho=\max_S\widehat\sigma_S\sqrt{K/(|E|\delta)}$, $|E|=40,000$, $\delta=0.05$를 사용한다.
$\widehat\sigma_S$는 E의 AIPW score 표준편차다. 거리 $\le2\rho$인 후보를 연결한다.
unique largest component의 중앙값을 반환한다. K=0, 최대 성분 동점, 비유한 값은 no-return이다.
이 반경은 nuisance 편향과 표현 편향을 보증하지 않는 **plug-in 경험적 반경**이다.
정리의 uniform-radius·separation·plurality 조건을 충족했다고 선언하지 않는다.
$\rho$, 성분 크기, 포함 split, 후보 효과, 단순 전체후보 평균·중앙값은 저장하되 추가 최적화에 쓰지 않는다.

## 7. 비교군과 무엇을 비교하는가

주 표의 다섯 방법은 X 조정, 전체 픽셀 조정, 학습 요약 조정, proximal balancing, oracle다.
전체 픽셀 조정은 X와 36×36 원본 전체를 읽는 CNN propensity·outcome 비교군이다.
CNN은 Conv(1→16, 3×3, padding1)→ReLU→MaxPool2→Conv(16→32, 3×3, padding1)→ReLU→MaxPool2다.
32×9×9 특징을 flatten하고 X를 붙여 128→64→1 head로 보낸다. outcome head에만 A를 함께 넣는다.
propensity와 outcome의 CNN 파라미터는 독립이다. Dfit의 같은 1,920/480행에서 A/Y 예측으로 최대40 epoch 초기화한다.
최종 nuisance는 N의 같은80/20분할에서 모든 CNN 파라미터를 최대60 epoch 미세조정하고 E에서 평가한다.
선정에 E나 잠재 정답을 쓰지 않는다. 이 비교군도 같은 D/N/E 자료와 사전 고정된 학습·선정 예산을 받는다.
학습 요약 조정은 같은 다섯 블록 encoder와 all-block fusion을 A/Y warmstart까지만 학습한 Z를 쓴다.
oracle은 $(X,U_{\mathrm c},U_{\mathrm t})$로 같은 nuisance/AIPW 경로를 쓴다.
naive 차이는 효과부호 진단으로 함께 기록한다. 모든 baseline은 추가 학습행을 받지 않는다.

같은-split 진단에서는 고정 $S=\{0\}$에 대해 선택된 restart의 warmstart Z와 final Z를
동일 N/E 행에서 각각 적합·평가한다. 이것이 균형 학습 전후를 비교하는 최소 대조다.
warmstart의 audit도 final과 같은 독립 critic 재적합 예산·행으로 계산하되, 이 사후 비교값은 checkpoint나 문턱 선택에 사용하지 않는다.
전체-W 요약 대 최종 집계의 차이는 전체 절차의 비교이며 균형 단계 단독 효과라고 하지 않는다.
잠재 숫자 분류는 사후 선택적 진단일 뿐 통과 조건이 아니다. 추가 분류기 연구는 기본 실행에 넣지 않는다.
이미지에 정보가 충분하면 전체 픽셀 조정이 같거나 더 좋을 수 있으며 그 순위를 그대로 보고한다.

## 8. 실행 순서와 예산

**A0. 생성기 preflight:** 먼저 $n=100,000$ 잠재행에서 propensity 범위, 처치율, outcome 유한성, naive·X·oracle 진단을 계산한다. 이미지 전수 materialization은 하지 않고 고정된 일부 행에서 MNIST 복원·변환·블록 덮개를 검사한다.

**A0. 생성식 preflight:** 이미지 없이 잠재벡터와 X,A,Y만100,000행 생성한다. naive 차이<0인지,
X-only OLS(Y를 1,A,X에 회귀한 A 계수), oracle OLS(Y를 1,A,X,Uc,Ut에 회귀한 A 계수)를 보고한다.
X-only는 선형 조정 진단이며 비모수 X 조정의 정확한 표적이라고 부르지 않는다. HC0 표준오차를 함께 저장한다.
그 숫자는 생성식 진단이며 learner 선택에 사용하지 않는다. oracle은 참값1의 ±0.03을 sanity 기준으로 삼는다.

**A. smoke:** 개발 seed 하나에서 축소판과 고정 split $\{0\}$로 실행한다.
분할 비율, 업데이트, warm checkpoint, latent 입력 금지, AIPW·반환 형식을 검사한다.
이어서 본 크기 $D=4,000,N=4,000,E=40,000$에서 split 하나의 시간·peak memory를 측정한다.
30-split 전체 비용과 15 confirmation 비용을 실측에서 추정해 먼저 보고한다.
기존 3시간 CPU는 참고 목표이고 보장이 아니다. 예상 전체 CPU wall time이 6시간을 넘으면
confirmation을 자동 시작하지 않고 병목과 같은 통계 사양의 계산 최적화안을 보고한다.

**B. 고정 split 개발:** 독립 자료 seed 3개, n=48,000, $S=\{0\}$.
같은-split warm/final의 독립 audit gap과 AIPW 오차를 기록한다.
진입 조건은 구현 검사 전부 통과, 3개 중 2개 이상 final critic reliability guard 통과,
그리고 독립 audit clipped gap의 paired 변화 중앙값이 0 이하인 것이다.
전체 DGP를 고치거나 숫자 정답으로 checkpoint를 고르지 않는다.
효과 오차 개선 여부는 별도 진단으로 보고한다. 이 작은 실험으로 유의한 우위를 주장하지 않는다.
warmstart가 계속 선택되면 “균형 최적화의 추가 개선 미확인”으로 표시하고 원인 1개만 점검한다.

**C. 전체 개발:** 같은 독립 자료 3개에서 30 split을 실행한다.
문턱 후보는 $t\in\{2,5,10,20,50\}\times10^{-4}$뿐이다.
저장된 audit 통계에 적용하며 문턱마다 재학습하지 않는다.
자료 3개 중 2개 이상에서 답을 내는 가장 작은 t를 선택한다. 없으면 개발 FAIL로 멈춘다.
이 선택에 ATE 참값·oracle·역할 라벨·기대 census를 사용하지 않는다.
같은 개발 audit를 여러 t에 사용하는 것은 개발 튜닝이며 독립 확증으로 보고하지 않는다.
Simpson 진단은 naive 차이 < 0인 개발 자료가 3개 중 2개 이상일 때 잠정 확인한다.
이를 만족하지 않으면 reversal 주장을 제외하고 생성식은 그대로 유지한다.

**D. 봉인:** 생성식, 행 분할, architecture, 모든 학습 예산·seed, 선택 규칙, t, 반경,
성과 관문, 소스·MNIST 해시를 저장한다. “implementation mode”를 명시한다.
개발 후 수정은 새 revision과 사유가 있어야 하며 confirmation을 본 뒤 수정한 결과와 합치지 않는다.

**E. 확인:** 새 독립 자료 **15개**에서 encoder부터 모든 학습·선별·추정을 새로 실행한다.
각 자료는 48,000행이고 각 방법이 같은 자료를 공유한다. 알고리즘 seed도 사전 고정한다.
학습 상태 5개에서 평가만 반복하는 방식으로 대체하지 않는다.
CPU 순차 실행을 기본으로 하고 실패 dataset도 원인과 함께 남긴다.
확인 결과를 본 뒤 좋은 seed만 다시 뽑거나 실패를 삭제하지 않는다.

## 9. seed와 불확실성 계산

기존 파일에 적힌 미사용 주장은 재사용하지 않는다. 봉인 시 아래 제안 값을 실제 ledger와 대조한다.
smoke: 6,510,000. 개발: 6,520,000+k, k=0,1,2.
confirmation: 6,600,000+k, k=0,…,14.
MNIST bank 추출, rendering, DGP, optimizer, critic, nuisance는 root seed에서 별도 SeedSequence로 분리한다.
사용 이력 충돌 시 아직 실행하지 않은 전체 예약 구간을 바꾸고 봉인 전에 기록한다.

통계 단위는 독립 데이터셋 seed다. split 30개나 row 4,000개를 replicate 수로 세지 않는다.
방법 M과 비교군 B의 paired loss difference는
$d_k=|\widehat\tau_{M,k}-1|-|\widehat\tau_{B,k}-1|$다.
95% 단측 상한은 $\bar d+t_{0.95,n-1}\operatorname{sd}(d)/\sqrt n$으로 사전 고정한다.
15개이므로 정규근사 가정의 small-sample paired t interval임을 밝힌다.
여러 비교는 각각의 구간으로 보고하며 전체 비교를 한꺼번에 유의하다고 주장하려면 Holm 보정을 추가 표기한다.
MAE는 mean absolute error, p90은 numpy quantile(method="linear")다. median과 혼용하지 않는다.
반환율 분모는 항상 15다. MAE·paired 비교는 반환된 자료에서만 계산하고 n_answered를 명시한다.
no-return을 오차 0으로 채우지 않는다. unconditional accurate-return 비율도
#(반환하고 절대오차≤0.10)/15로 별도 보고한다.

## 10. 판정: 구현·성능·비교를 분리한다

| 판정 | 봉인 기준 | 실패의 의미 |
|---|---|---|
| 구현 | 분할·입력·목적·provenance 검사 전부 통과, 15 run 완료 | 결과를 근거로 사용하기 전 수리 필요 |
| 반환 | 반환율 ≥0.95, 즉 15/15 | aggregate 안정성 부족 |
| 성능 | 반환 MAE ≤0.05, p90 ≤0.10 | 개발 목표 미달 |
| oracle 근접 | paired 초과 MAE의 단측 상한 ≤0.05 | oracle 근접 주장 미달 |
| X 대비 | paired 차이 상한 <0 | X 대비 우위 근거 |
| 전체 픽셀 대비 | paired 차이 상한 <0 | 미달하면 우위 주장하지 않음 |
| 요약 대비 | paired 차이 상한 <0 | 미달하면 균형 절차의 추가 이점 미확인 |

“near-oracle performance PASS”는 구현·반환·성능·oracle 근접이 모두 성립할 때다.
X/raw/summary 비교는 각자 따로 판정한다. 좋은 raw 성능을 이유로 baseline을 약화하지 않는다.
Simpson reversal은 자료별 naive 차이와 부호 빈도를 보고한다. <0.5를 reversal이라고 쓰지 않는다.
완벽한 screen census, 특정 wrong 후보 선택, exact ORC, 정확한 근 복원은 관문에서 제외한다.

## 11. 구현 출발점·명령·필수 검사

기존 code/probe_brier_neural.py에는 BlockStack, Representation, 독립 critic mode,
critic refit, AIPW, graph aggregate가 있다. 읽고 작은 adapter를 만든다.
현재 TRAINING_MODE 기본값과 minibatch clipping 경로를 확인하고 이 PRD 사양을 명시적으로 설정한다.
기존 train_representation을 그대로 호출해 다른 목적을 실행하면 구현 FAIL이다.
기존 image renderer는 무게중심 채널용이므로 새 숫자 renderer와 혼용하지 않는다.
새 파일명 제안: code/sim5_digit_proxy.py, code/run_sim5_digit_proxy.py, code/test_sim5_digit_proxy.py.
이 문서는 구현을 지시할 준비물이며, 이 파일들이 이미 존재하거나 실행됐다는 뜻이 아니다.

다음 CLI는 **구현자가 만들 인터페이스 제안**이다. 현재 실행 가능한 명령이라고 보고하지 않는다.

~~~bash
python code/run_sim5_digit_proxy.py --phase smoke --prd memo/2026-09-17-sim5-v2-digit-proxy-prd-v1.md
python code/run_sim5_digit_proxy.py --phase fixed-split-dev
python code/run_sim5_digit_proxy.py --phase full-dev
python code/run_sim5_digit_proxy.py --phase freeze
python code/run_sim5_digit_proxy.py --phase confirm --protocol results/sim5_digit_v2/protocol.json
python code/run_sim5_digit_proxy.py --phase score --protocol results/sim5_digit_v2/protocol.json
~~~

필수 검사는 6개 묶음이다.
1. 1,296픽셀의 무중복 덮개, index 순서, split 30/15/15 계수.
2. train/select/audit/N/E 교집합 0, preprocessing fit 행 확인, overlap 주입 시 거부.
3. learner batch에 latent key 주입 시 거부, response head A가 phi 입력으로 흐르지 않음.
4. 두 critic 파라미터 분리·zero residual 포함관계·전체 gap clipping 대 minibatch clipping의 구별.
5. 같은 restart warmstart digest, 동일 평가행 paired gap/SE, audit 변경이 checkpoint 선택을 바꾸지 않음.
6. 미완료·중복 seed·바뀐 protocol/source/checkpoint 해시 거부와 no-return 분모 유지.

## 12. 산출물과 그림

결과 디렉토리 제안은 results/sim5_digit_v2/, 그림은 figures/sim5_digit_v2/다.
protocol에는 PRD/source/data 해시, constants, seed ledger, dependencies 버전, 예상 15 dataset ID가 들어간다.
dataset record: complete/status, seed, row hashes, timings, selected threshold, all 30 split records, estimates/comparators.
split record: held-out indices, actual inputs, restart/epoch, warm/final checkpoint digest, critic seed·fit/val losses,
selection/audit signed gaps·SE·clipped gaps, reliability flag, retain decision, nuisance losses, AIPW scores digest,
theta_hat, radius, component membership. 생성기 역할 라벨은 별도 evaluator metadata다.
scorer는 원시 record로 모든 숫자를 재계산하고 입력 해시를 검증한다.

그림은 두 개로 끝낸다.
- 주 그림: 다섯 방법의 15개 자료별 절대오차와 평균, 반환 수. 같은 seed끼리 paired 연결 가능.
- 진단 그림: 고정 split의 warm/final audit gap과 효과 오차, 그리고 전체 split의 screen/성분 구성.
gap 대 효과오차 산점도는 empirical association이다. Gamma를 계산하지 않았으므로 certificate 그림이라고 부르지 않는다.
full-pixel 입력은 남긴 블록 encoder와 held-out critic, raw baseline에서 각각 어떤 픽셀을 읽는지 caption에 적는다.
본문 교체와 LaTeX 수정은 이 PRD 구현 범위와 별도로 처리한다.

## 13. 문헌과 해석 경계

Miao, Geng and Tchetgen Tchetgen (2018), “Identifying causal effects with proxy variables of an unmeasured confounder”
는 proxy를 통한 식별에 구조 가정이 필요하다는 배경이다.
Xu et al. (2021), “Deep Proxy Causal Learning and its Application to Confounded Bandit Policy Evaluation”
는 고차원 proxy 특징을 관측자료에서 학습한다는 설계의 참고다.
이 문헌의 bridge estimator 또는 식별 가정을 현재 균형 방법에 자동으로 이전하지 않는다.
A/Y warmstart는 정보 보존을 돕는 구현 선택이며 그 자체가 causal identification을 보장하지 않는다.
이번 이미지 실험의 결론은 실제 출력과 비교표로 정한다. 기존 정리 감사의 PASS를 이 실험에 물려주지 않는다.
평가행 확대는 반경의 표본잡음 항을 약 1/√10로 줄이지만 표현 편향이나 nuisance 편향을 고치지 않는다.

<details>
<summary>Superseded archive: 최초 PRD v1. 실행 사양으로 사용하지 않는다.</summary>

아래는 개정 전 사양을 보존한 기록이다. 경로·명령의 inline-code 표기는 일반문자로 옮겼으며 실행 authority는 위 Revision 2뿐이다.

# Sim-5 v2 PRD v1: 손글씨 숫자를 중증도로 쓰는 단일 이미지 proxy

Created: 2026-09-17 CDT
Supersedes: Sim-5 (SCM-7-v4, 다섯 장 이미지, $C$·$I$ 이름표 전달)
Status: **제안. 검토 대기. 구현하지 않았고 DGP seed를 쓰지 않았다.**

## 0. 통제 문장

단위마다 이미지 **한 장** $W$를 두고, 숨은 중증도 $U$는 그 이미지에 적힌 손글씨 숫자이며, 픽셀을 자료를 보기 전에 정한 여섯 영역으로 나눈 뒤
원고의 Algorithm 1을 문자 그대로 실행한다. 학습기는 어느 영역이 무슨 역할인지 모르고, 모든 영역을 같은 코드 경로로 다룬다.

이 실험의 이름은 **역할 비인지 표현으로 실행한 Algorithm 1의 이미지 실험**이다. 이 실험이 시험하는 것은 근사 균형 regime의
Theorem 3, 4, 5와 심사·연결·다수결의 동작이다. 정확한 균형(Theorem 1)은 이 설계에서 성립하지 않으며 시험 대상이 아니다(12절 R1).

다음은 이 실험의 질문과 무관하므로 뺀다. 다섯 장 이미지, 둘째 교란 $C$, 이름표 달린 숫자 좌표, $96\times96$ 캔버스, 무게중심 역변환,
256차원 벡터의 Sim-4, budget 표집($m<T$), parameter grid, DeltaAI GPU.

## 1. 자료 생성

관측 자료는 $O=(X,W,A,Y)$뿐이다. $U$와 $I$는 미관측이며 시뮬레이터와 평가자만 쓴다. 학습기는 아래 어떤 계수도 받지 않는다.

$$U\sim\mathrm{Unif}\{1,\ldots,9\},\qquad \bar U:=U-5,\qquad I,\ \eta_X,\ \varepsilon_Y\sim N(0,1)\ \text{독립}$$
$$X=0.5\,\bar U+2\eta_X$$
$$P(A=1\mid U,I,X)=0.1+0.8\,\Phi\bigl(0.5\,\bar U+1.5\,I+0.2\,X\bigr)$$
$$Y=A-0.8\,\bar U+0.2\,X+0.3\,\varepsilon_Y,\qquad \tau=1$$

$U$는 환자의 중증도, $I$는 판독의의 처치 결정에는 영향을 주지만 예후에는 무관한 스캔 표식이다.
overlap은 구성상 $0.1\le e\le0.9$이므로 $\eta=0.1$이다. 격자는 없다. 계수는 preflight P4가 요구하는 교란 강도를 만족하는지만 확인하고,
만족하지 못하면 이유를 적고 v2를 낸다.

## 2. 이미지와 블록 분할

캔버스는 $36\times36$, 픽셀 값은 $[0,1]$이다. 한 단위의 이미지는 다음 순서로 만든다.

1. **견본**: 동결된 MNIST 학습 집합(materials/benchmarks/mnist/mnist.npz, sha256 731c5ac6…)에서 숫자 $U$의 이미지 하나를 복원추출한다.
   숫자 1–9는 54,077장, 클래스당 최소 5,421장이다.
2. **비틀기**(잡음, 모든 변수와 독립): 회전 $\theta\sim\mathrm{Unif}(-20^\circ,20^\circ)$, 크기 $s\sim\mathrm{Unif}(0.9,1.1)$, 쌍선형 보간.
   $28\times28$에 적용한 뒤 $36\times36$ 캔버스 중앙에 놓는다. 바깥 4픽셀 여백에는 글자가 닿지 않는다.
3. **표식**: 오른쪽 아래 $4\times4$ 영역 $M$(행 32–35, 열 32–35)의 16픽셀을 $0.5+0.4\tanh(I/2)$로 채운다.
4. **센서 잡음**: 모든 픽셀에 $0.05\,\xi$, $\xi\sim N(0,1)$ i.i.d.를 더하고 $[0,1]$로 자른다.

블록은 $J=6$이다.

| 블록 | 영역 (행, 열) | 픽셀 수 | 설계상 역할 (학습기는 모름) |
|---|---|---|---|
| $Q_1$ | 4–17, 4–17 | 196 | 글자의 왼쪽 위. $U$의 깨끗한 proxy |
| $Q_2$ | 4–17, 18–31 | 196 | 오른쪽 위 |
| $Q_3$ | 18–31, 4–17 | 196 | 왼쪽 아래 |
| $Q_4$ | 18–31, 18–31 | 196 | 오른쪽 아래 |
| $M$ | 32–35, 32–35 | 16 | 표식. $I$만 담고 $U$는 없음 |
| $R$ | 바깥 4픽셀 여백에서 $M$을 뺀 것 | 496 | 잡음만. $U$도 $I$도 없음 (ORC 실패 부류) |

합 $4\cdot196+16+496=1296=36^2$. split은 $2^6-2=62$개이고 전부 열거한다($m=T$).

여백을 캔버스에 붙여 만든 이유는 측정값에 있다. 원본 MNIST $28\times28$의 자체 4픽셀 테두리는 잉크 비율이 평균 3.7%뿐인데도
그 테두리만으로 숫자를 0.567 맞힌다(우연은 0.111). 즉 원본에는 비어 있는 영역이 없다. ORC 실패 부류가 있으려면 정보가 없는 영역을
만들어야 하고, 센서 잡음만 있는 여백이 그것이다. 이 부류를 두지 않으려면 $R$을 빼고 $J=5$로 간다(12절 R4).

원본 $10\times10$ 사분면으로 잰 참고값(선형 MLP, 20,000행 학습, 5,000행 검사): 사분면 하나 0.767, 둘 0.905, 셋 0.946, 넷 0.959, 전체 0.956.
비틀기를 거친 $14\times14$ 사분면의 값은 preflight P2가 잰다.

## 3. 학습기가 받는 것과 표현류

학습기는 $(X,\ W_{(1)},\ldots,W_{(6)},\ A,\ Y)$를 받는다. $W_{(j)}$는 길이 $(196,196,196,196,496,16)$의 픽셀 벡터이고, 순서는 고정이지만
이름도 역할도 없다. 여섯 블록 모두 같은 코드 경로를 지난다. if j == 0 같은 분기는 금지한다.

**1단계, 블록 특징(학습 표본 $\mathcal I_D$에서, 데이터셋마다 한 번).** 블록 $j$마다 encoder $g_j:\mathbb R^{p_j}\to\mathbb R^8$을 둔다.
구조는 여섯 개가 같다: MLP $p_j\to64\to64\to8$, SiLU. 각 $g_j$는 블록 $j$의 픽셀만으로 $A$를 예측하도록 학습한다
(BCE, Adam $10^{-3}$, weight decay $10^{-4}$, 최대 40 epoch, $\mathcal I_D$의 25%를 검증으로 떼어 early stopping). 출력 $f_j=g_j(W_{(j)})$.

**2단계, split마다 표현 선택.** 남긴 블록 집합 $S^c$에 대해 표현류는 유한 library다.

$$\Phi_S=\bigl\{\,Z=(X)\,\bigr\}\ \cup\ \bigl\{\,Z=(X,f_j): j\in S^c\bigr\}\ \cup\ \bigl\{\,Z=(X,f_j,f_k): j<k\in S^c\bigr\}$$

원소 수는 $1+|S^c|+\binom{|S^c|}{2}\le16$이다. Algorithm 1의 $\widehat\phi_S$는 이 library에서 $\widetilde D_S^2$를 최소로 하는 원소다.
동점(차이 $<10^{-6}$)이면 support가 작은 쪽, 그다음 블록 index가 작은 쪽을 고른다. 이 규칙 때문에 잡음만 있는 $R$을 빼면 모든 원소가
0 근처에서 동점이 되어 $Z=(X)$가 선택된다. 그것이 ORC 실패 부류의 목표값 $\theta_R=\theta_X$를 정한다.

이 class는 블록 순열에 닫혀 있고 역할을 참조하지 않는다. positive-control 감사의 "남긴 블록 하나를 통로로" class에 학습된 블록 특징을
얹은 것이다. 연속 head $h_S$를 minimax로 배우는 class는 12절 R5의 대안이다.

## 4. 심사, 연결, 다수결

원고의 Algorithm 1 줄 2–6을 그대로 쓴다.

**심사 통계.** $\mathcal I_D$에서 5-fold cross-fitting으로 nested Brier critic을 맞춘다.
기본 critic $b_0(Z)$는 MLP $2\times64$ SiLU, 출력을 $[0.05,0.95]$로 자른다.
확장 critic $b_1(Z,W_S)$는 $b_0$에 잔차 MLP를 더한 것으로, 잔차 가지는 $(Z,\ W_S$의 원 픽셀$)$을 읽는다.
$b_1$은 lifted $b_0$(잔차 가지 가중치 0)에서 시작하고 restart 2회, early stopping. **뺀 블록의 모든 픽셀이 심사에 들어간다.**
통계는 held-out fold의 $\mathrm{gap}=\overline{(b_0-A)^2-(b_1-A)^2}$이고 $\widetilde D_S^2=\max\{\mathrm{gap},0\}$.

**선택 대 심사.** library 원소 16개마다 MLP critic을 새로 맞추는 비용을 피하려고, 선택은 probit-선형 critic(감사와 같은 것)으로 하고
선택된 원소 하나에만 위 MLP 심사를 적용한다. 선택 critic과 심사 critic이 다르다는 점은 보고한다.

**보류 규칙.** $\widetilde D_S^2+z_{1-\alpha/T}\,\mathrm{SE}<t$ 이고 포화가 없을 때 보류한다. $\alpha=0.05$, $T=62$.
포화는 $b_0$의 1% 초과가 자름 경계 $10^{-3}$ 안에 있을 때이며, 이것이 $\eta\le\widehat e_S\le1-\eta$의 구현이다.

**추정.** 보류된 split마다 $\mathcal I_N$에서 nuisance $\widehat e(Z)$(로지스틱 MLP, $[0.1,0.9]$로 자름)와 $\widehat m_a(Z)$(MLP 회귀)를
맞추고 $\mathcal I_E$에서 AIPW 점수를 낸다. $\mathcal I_D,\mathcal I_N,\mathcal I_E$는 서로소다.

**연결과 반환.** $\rho=\max_S\widehat\sigma_S\sqrt{N/(n_E\delta)}$, $\delta=0.05$, $\widehat\sigma_S$는 $\mathcal I_E$ 점수의 표준편차.
$2\rho$ 안에서 잇고 가장 큰 성분의 중앙값을 반환한다. 동률은 실패로 센다.

## 5. 표본과 분할

| 조각 | 크기 | 쓰임 |
|---|---|---|
| $\mathcal I_D$ | 4,000 | encoder, 선택, 심사 |
| $\mathcal I_N$ | 4,000 | nuisance |
| $\mathcal I_E$ (성능 팔) | 4,000 | AIPW 점수, Figure 5의 $n=12,000$ |
| $\mathcal I_E$ (분리 팔) | $n_E^{\mathrm{big}}$ | Theorem 6 조건 검사 |

$n_E^{\mathrm{big}}$은 preflight가 $g_{\min}>4\rho$를 만족하도록 정한다: $n_E^{\mathrm{big}}\ge16\,\widehat\sigma^2N/(\delta\,g_{\min}^2)$.
$\widehat\sigma\approx3$, $N\approx15$, $g_{\min}\approx0.9$면 약 53,000이고, 두 자리로 올려 동결한다.

## 6. Preflight (개발 seed 6,100,000, 한 데이터셋)

| 관문 | 내용 | 통과 기준 |
|---|---|---|
| P1 | $W_R$만으로 숫자 맞히기 | 정확도 $\le0.15$ |
| P2 | 비튼 $14\times14$ 사분면에서 숫자 맞히기 | 셋 남김 $\ge0.90$, 둘 $\ge0.70$, 하나 $\ge0.40$. 값을 기록 |
| P3 | $W_M$에서 $I$ 복원 | $R^2\ge0.99$ |
| P4 | 교란 강도, $n=100,000$ | naive 대비 $<0.5$, $X$만 조정 $<0.4$, oracle($U,I,X$ 조정) $\in[0.97,1.03]$ |
| P5 | 62개 split census | 아래 |
| P6 | 시간 | 학습 상태 하나 + draw 20개 $\le$ 30분 CPU |
| P7 | 정직한 대비 (보고 관문) | 아래 |

**P5.** 개발 학습 표본 하나에서 62개 split 전부에 대해 잠정 $t$(7절 사다리의 중간값)로 보류 여부, 선택된 library 원소,
$n=100,000$ 평가 표본의 $\widehat\theta_S$를 기록한다. 기대 구조는 셋이다.
(i) $M\ni S$인 32개는 탈락($I$를 잃어 균형 불가). (ii) $S=\{R\}$는 보류되고 $\widehat\theta_R\approx\theta_X$. (iii) 나머지 29개 중
사분면을 둘 이하 뺀 것은 보류되고 $\widehat\theta_S\approx\tau$, 셋 이상 뺀 것은 탈락.
관문: $M$을 포함한 split이 하나라도 보류되면 설계 FAIL. 보류 집합에서 $\Delta\ge0.3$, $L=1$, $g_{\min}\ge0.5$.
(iii)의 보류 split이 4개 미만이면 library support를 3으로 올린 v2를 낸다.

**P7.** 같은 데이터셋에서 $n=12,000$으로 네 값을 잰다. $X$ 조정; $X$+전체 이미지 조정(1296픽셀+$X$를 읽는 CNN propensity·outcome, AIPW);
균형 검사 없는 학습 요약($36\times36$ CNN의 8차원 요약, 뺀 것 없음); PROBE. 순서를 기록한다.
전체 이미지 조정의 오차가 PROBE의 오차$+0.03$ 이내면 12절 R2의 claim boundary가 적용된다. 이 관문은 실험을 막지 않는다.

## 7. Dev study (seed 6,110,000 + k, k = 0…4)

사다리 $t\in\{2,5,10,20,50\}\times10^{-4}$. 각 $t$에서 다섯 개발 seed의 보류 집합을 P5의 기대 구조와 비교한다.
$t$는 기대 구조를 5개 중 4개 이상에서 맞히는 가장 작은 값으로 정한다. 그런 $t$가 없으면 v2를 낸다.
같은 seed에서 encoder epoch 상한과 critic restart 수를 고정한다. 그 뒤 모든 것을 동결한다.
개발 seed의 결과는 본문에 쓰지 않는다.

## 8. Confirmation (봉인)

protocol JSON에 1–7절의 모든 상수, 소스 파일 sha256, seed를 넣고 동결한다. 실행기는 CLI가 동결값과 다르면 멈춘다.

구조는 감사와 같다. 학습 상태 $s=0,\ldots,4$마다 $\mathcal I_D$를 한 번 뽑아 encoder·선택·심사를 돌리고 보류 집합을 고정한다.
그 위에서 평가 draw $r=0,\ldots,19$마다 $(\mathcal I_N,\mathcal I_E)$를 새로 뽑아 AIPW와 반환을 계산한다(성능 팔).
상태마다 분리 팔 draw 하나를 $n_E^{\mathrm{big}}$으로 추가한다.

상태마다 실현된 보류 집합에서 $N,\Delta,L,g_{\min},\rho$를 다시 계산하고 세 조건(다수, 분리, 보류 집합의 기대 구조 일치)을 판정한다.
관측 성공률을 $1-\delta-L\,e^{-N\Delta^2/2}$와 비교한다. 동률 반환은 실패, 기대 밖 보류는 FAIL로 기록하고 그대로 보고한다.

## 9. 보고 항목과 그림

- Figure 5의 Sim-5 열: 다섯 줄($X$ 조정, $X$+전체 이미지, 학습 요약, PROBE, oracle), 데이터셋 20개(상태 5 × draw 4를 쓰거나 draw 전부).
- Theorem 3 패널: 보류된 split마다 $x=\widetilde D_S^2$, $y=|\widehat\theta_S-\tau|$. 잔여 불균형과 오차가 같이 움직이는지.
- 심사 census: 62개 split의 $\widetilde D_S^2$와 문턱 $t$. 흐름도 (c)의 재료.
- 표: 상태별 $N,\Delta,L,g_{\min},\rho$, 조건, 성공률, 하한.
- 흐름도 (a)(b)의 thumbnail을 이 실행의 실제 이미지로 바꿀 수 있다.

## 10. Seed

| 용도 | seed |
|---|---|
| preflight | 6,100,000 |
| dev | 6,110,000 + k, k = 0…4 |
| confirmation 학습 상태 | 6,300,000 + 10,000 s |
| confirmation 평가 draw | 6,400,000 + 10,000 s + r |
| 분리 팔 draw | 6,450,000 + s |

봉인 전에 results/, code/, memo/에서 이 값들이 단어 단위로 나오지 않음을 확인하고 그 명령을 protocol에 적는다.
2026-09-17 확인: 세 디렉토리 어디에도 없다.

## 11. 예산 (측정값 근거)

이 Mac(Apple M4, torch 2.13, CPU 스레드 4)에서 잰 값이다.

| 항목 | 측정 |
|---|---|
| 4,000행 MLP 한 epoch (입력 100–784, batch 128) | 0.01초 CPU (MPS 0.03초, 작은 모델은 CPU가 빠름) |
| 회전+크기 비틀기 | 이미지당 0.04ms, 50,000장에 2초 |
| MNIST 적재와 sha256 | 1초 |

학습 상태 하나의 추정: encoder 6개 × 40 epoch × 0.01초 ≈ 3초. split 하나: probit 선택 ≈ 1초, MLP 심사 5 fold × critic 2 × 40 epoch × 0.01초 × restart ≈ 8초,
합 ≈ 10초. 62개 ≈ 10분. 평가 draw 하나: 보류 split ≈ 15개 × nuisance MLP 2 × 40 epoch × 0.01초 ≈ 12초, 20개 ≈ 4분. 분리 팔 $n_E=60,000$의 추론 ≈ 1분.
**학습 상태 하나 ≈ 15분 CPU.** preflight 1 + dev 5 + confirmation 5 = 11개 상태 ≈ **3시간 CPU. DeltaAI GPU 0분.**
이 추정은 P6이 실측으로 대체한다. 다섯 장 $96\times96$ 설계가 GPU를 쓴 이유는 픽셀 46,080개를 CNN 몸통과 픽셀 critic으로 읽었기 때문이고,
한 장 $36\times36$에 블록별 MLP를 쓰면 두 자릿수 이상 싸다.

## 12. 위험, claim boundary, 미결

**R1. 근사 균형.** 이 설계에서 깨끗한 블록의 정확한 균형은 $Z$가 $U$를 정확히 결정할 때만 가능하다. 합성 설계의 정확한 균형은
$Z=(X,C,\bar K+rI)$의 $r$이 처치 원인 $I$를 지렛대 삼아 잔여 교란을 상쇄하도록 풀렸기 때문에 가능했다. 여기엔 그 지렛대가 없고
critic이 유연하다. 그러므로 보류된 목표 split의 $\theta_S$는 $\tau$에서 잔여만큼 벗어나며, 실험은 Theorem 3의 $\Gamma\cdot D_{\mathrm{res}}$
관계를 보이는 것이다. 정확한 균형을 주장하지 않는다.

**R2. 전체 이미지 조정이 PROBE와 비슷하거나 더 좋을 수 있다.** CNN은 전체 이미지에서 $U$를 0.96 이상, $I$를 정확히 읽으므로
$X$+전체 이미지 조정은 거의 oracle이다. PROBE는 일부 사분면만 남기니 잔여가 더 크다. 이 경우 Sim-5의 주장은
"역할을 모르는 학습기가 실제 픽셀에서 Algorithm 1을 돌리면 심사와 다수결이 이론대로 움직이고 certificate가 오차를 따라간다"이며,
정확도 우위는 지렛대가 있는 Sim-1–3에서 보인다. P7이 이것을 측정해 문장을 정한다. 원고에 그대로 적는다.

**R3. 잘못된 부류가 하나뿐이다.** $\{R\}$ 하나라 다수결이 쉽다. 다수결의 경계를 흔드는 실험은 합성 $J=7$의 몫이다.
$R$을 셋으로 쪼개 $J=8$로 가면 잘못된 부류가 7개가 되지만 v1에서는 하지 않는다.

**R4. 잡음 여백은 설계 장치다.** 없애면 $J=5$, ORC 실패 부류가 사라지고 심사만 일한다. 용한의 결정 사항.

**R5. 표현류.** 유한 library(3절)가 1차다. P5에서 support 2로 보류 목표 split이 4개 미만이면 support 3, 그래도 안 되면 연속 head를
minimax로 배우는 class로 v2를 낸다. 모두 confirmation seed 이전에 정한다.

**R6. $C$가 없다.** Figure 3의 "작용하는 좌표 둘"은 Sim-1–3에만 해당한다. Sim-5 v2에는 작용 좌표가 $I$ 하나이고 둘째 교란이 없다.
원고 도입부에 한 문장으로 적는다.

**R7. 1단계 encoder가 $A$를 쓴다.** Algorithm 1의 $\widehat\phi_S$는 $\mathcal I_D$에서 $A$를 쓰는 균형 목적으로 배우므로 규칙 안이다.
encoder는 모든 블록의 픽셀로 학습되지만 단위 $i$의 $Z_i$는 $i$의 남긴 블록만 읽는다. 흐름도 (b)의 문구는 "uses only"로 둔다.

## 13. 원고에 미치는 변화

Sim-4는 본문에서 뺀다(부록 후보). Sim-5는 이 설계로 바뀌고 도입부에 한 문단을 갖는다. Figure 5의 다섯째 열을 다시 그린다.
5.4의 보라 메모("이름표 달린 좌표")는 사라진다. 흐름도의 thumbnail을 실제 자료로 바꿀 수 있다. caption은 한 줄 규칙 그대로.

</details>
