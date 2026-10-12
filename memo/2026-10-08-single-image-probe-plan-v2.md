# 단일 이미지 PROBE 계획 v2

작성: 2026-10-08 CDT. 작성자: `single-proxy-balancing` (수학 검토: `math-theory-agent`). v1(`2026-10-08-single-image-probe-plan-v1.md`)을 대체한다. v1은 기록으로 남긴다.

v2가 v1과 다른 점은 1차 구현(§15)이 가르쳐 준 네 가지다. 판정 predictor는 사다리 최상단으로 고정한다. 허용 예산 격자를 $\{0.005,0.01,0.02\}$로 내린다. phantom은 image-only 모드가 기본이다. 1차 보고의 오류 하나를 정정해 기록한다.

이 문서는 계획과 실행 계약이다. 1차 구현 코드는 `code/single_image/`, 결과는 `results/single_image/phase1/`에 있다. 2026-10-09 local 구현·실행 범위는 §16에 고정한다. 실제 완료 상태는 새 결과 JSON으로 판정한다.

---

## 0. 한 문장 목표

환자마다 CT 한 장 $W$만 있을 때, $W$에서 남는 쪽 표현 $Z$와 떼어 둔 쪽 $T$를 만들어 PROBE를 적용한다. 두 predictor가 각자의 참 처치확률에 가깝고, 실제 불균형이 허용 예산 안에 있으며, 이미지 변환이 식별에 필요한 proxy 조건을 보존할 때, $Z$로 보정한 효과를 추정하고 남는 편향을 한계와 함께 보고한다.

## 1. 기호

- 관측: $O=(W,A,Y)$. $A\in\{0,1\}$ 처치, $Y\in\mathbb R$ 결과, $W\in\mathbb R^{p}$ CT 한 장의 픽셀 벡터. 이번 실행의 feature는 이미지만이다. 관측 공변량 $X$는 없다.
- 숨은 변수: $U$ 중증도. $G$ 전역 요인(밝기, 획득 조건, 체격). $I$ 처치 전용 원인(장치 표식).
- 잠재결과 $Y(a)$, 목표 $\tau=\mathbb E[Y(1)-Y(0)]$.
- 영역: 분석 전에 고정한 픽셀 분할 $W=(W_0,W_1,\dots,W_J)$, 기본 $J=5$(폐엽). $W_0$는 폐엽 밖의 픽셀이고 항상 남긴다. 영역 $k$와 인접하지 않은 영역의 집합 $N(k)$.
- 분할 $S$: 떼어 둘 폐엽의 집합. 기본은 $|S|=1$인 다섯 개. 남는 쪽 픽셀 $V_S=(W_0,W_{-S})$, 떼어 둔 쪽 $T_S=W_S$는 원 픽셀.
- encoder $\phi_S$: 학습하는 bottleneck은 2차원이다. Local 표현은 $Z_S=(c(W_{\mathrm{ref}}),\phi_S(V_S))\in\mathbb R^3$이다. $W_{\mathrm{ref}}$는 모든 폐엽 밖의 고정 body mask이고 $c$는 영상에서 계산한 밝기 scalar다. Raw/corrected 모두 이 좌표를 보존한다.
- 참 처치확률: $e(z)=P(A=1\mid Z_S=z)$, $p(t,z)=P(A=1\mid T_S=t,Z_S=z)$.
- 실제 불균형: $D^2=\mathbb E[\{p(T_S,Z_S)-e(Z_S)\}^2]$.
- predictor: $\widehat q_0(Z_S)$는 $e$의 추정량, $\widehat q_1(T_S,Z_S)$는 $p$의 추정량. 오차 $\epsilon_0^2=\mathbb E[(\widehat q_0-e)^2]$, $\epsilon_1^2=\mathbb E[(\widehat q_1-p)^2]$.
- Brier 위험 $R_0=\mathbb E[(A-\widehat q_0)^2]$, $R_1=\mathbb E[(A-\widehat q_1)^2]$. 독립 환자에서 계산한 표본 차이 $\widehat g$.
- 허용 예산 $\delta^2$. 조정값 $\tau_Z=\mathbb E[\mathbb E(Y\mid A=1,Z)-\mathbb E(Y\mid A=0,Z)]$.
- 고정 요약 $\chi_l(W_l)\in\mathbb R^{16}$: 영역 $l$의 학습하지 않는 통계량.
- 영역 reader $\psi_k(W_k)\in\mathbb R^{d_\psi}$: 이미지만으로 학습하는 영역별 요약. 보조 부품이다(§6.4).

## 2. 가정

- (M1) 원본 영상의 생성 구조. $W_k=f_k(U,G,I,\varepsilon_k)$, $A=a(U,G,I,\epsilon_A)$, $Y(a)=y(a,U,G,\epsilon_Y)$. 영역 잡음은 $(U,G,I)$가 주어지면 서로 독립이고 처치·결과 잡음과 독립이다. 장치가 없는 영역에서는 $f_k$가 $I$에 의존하지 않는다. 공통 reference로 보정한 영역의 잡음 독립은 자동 승계하지 않는다. Audit channel은 원본 픽셀로 유지한다.
- (M2) 중증도 가시성. 모든 $k$에 대해 $f_k$는 $u$에 대해 상수가 아니다. 미만성 질환을 뜻한다. 국소 질환은 범위 밖이다.
- (A3) outcome-relevant completeness. 논문 가정 3을 $T_S=W_S$에 대해 가정한다. 데이터로 검정하지 않는다.
- (A4) stability. 논문 가정 4의 상수 $\Gamma_S$가 유한하다고 가정한다. 편향 한계에만 쓴다.
- 가정하지 않는 것. 영역 reader나 encoder가 $U$를 복원한다고 가정하지 않는다. 선형 Gaussian 작업 모형의 결과를 신경망 encoder로 확대하지 않는다. 특정 estimator의 값을 "모든 이미지 기반 estimator의 천장"으로 해석하지 않는다.

## 3. 수학적 뼈대

### 3.1 Brier 분해
$$
R_0-R_1=D^2+\epsilon_0^2-\epsilon_1^2 .
$$
1차 구현은 이 식의 위험을 경험적으로 보여 주었다(§15.2). 같은 $Z$, 같은 $q_0$에서 $q_1$을 압축 reader에서 원본 이미지 predictor로 바꾸자 $\widehat g$가 $0.0049$에서 $0.0106$으로 올라갔다. 실제 $\epsilon_1$을 직접 측정해 항등식을 검증한 결과는 아니다. $\epsilon_1$이 크면 불균형이 숨을 수 있다.

### 3.2 통과식
모집단 위험 차이의 상한을 $\widehat g+r$로 통제하고 $\epsilon_1^2\le b_1$의 근거가 있으면 $D^2\le\widehat g+r+b_1$이다. 보수적 통과 조건은
$$
\max(\widehat g,0)+r+b_1\le\delta^2 .
$$
$z\cdot\mathrm{SE}$는 정상근사 진단이다. Finite guarantee와 구분한다. 실제 고정된 predictor와 같은 $Z,T$에 대한 $b_1$의 근거가 없으면 경험적 검사다. 참 L reference의 오차를 raw-image predictor로 옮기지 않는다. 판정 $q_1$은 원본 픽셀을 읽는 최상단이고, 압축 reader gap은 진단에만 쓴다. Local finite bound는 §16의 독립 S와 fresh phantom 계산을 사용한다.

### 3.3 떼어 둔 쪽은 고정한다
$Z$를 고정하고 $T'=h(T,Z)$가 임의의 가측 함수이면 $D^2(T')\le D^2(T)$다. 균형 목적함수로 $T$를 고르면 해는 상수 $T$다. $T_S$는 분석 전에 고정하고 어떤 학습 목적함수로도 바꾸지 않는다. 같은 이유로 판정 predictor의 입력도 거칠게 만들지 않는다.

### 3.4 식별의 승계
장치가 없는 원본 held-out channel과 그 channel을 읽지 않는 retained 변환에 대해 논문 가정 2를 점검한다. 원본 $T$와 disjoint reference를 유지하며, latent $U^*=(U,G)$를 조건으로 채널 독립이 성립하는 경우에만 승계를 주장한다. 모든 분할에 자동 성립하지 않는다. (A3), exact balance, uniform overlap이 추가될 때 논문 정리 1을 적용한다. Image ORC와 stability는 local 학습만으로 검증되지 않는다.

### 3.5 편향 한계
(A3), (A4), overlap $\eta\le e\le 1-\eta$, $D\le\delta$ 아래에서
$$
|\tau_{Z_S}-\tau|\le\frac{\Gamma_S\,\delta}{\eta(1-\eta)} .
$$
허용 예산을 넓히면 이 한계도 비례해 넓어진다. 한계는 추정값과 함께 보고한다.

### 3.6 작업 모형의 닫힌 식 (phantom 평가 전용)
선형 Gaussian 한 요인 모형($\psi_k=\lambda_kU+\sigma_k\xi_k$), probit 처치($A=\mathbf 1\{\gamma U+\epsilon_A>0\}$), 선형 결과($\beta U$), 남는 측정값의 선형 encoder라는 범위에서, 정밀도 $\Omega_K=\sum_{k\in K}\lambda_k^2/\sigma_k^2$, $v_K=1/(1+\Omega_K)$, $s_K=\sqrt{1+\gamma^2v_K}$, $t_K=\gamma\mathbb E[U\mid\psi_K]/s_K$로 둔다. $\varphi$와 $\Phi$는 표준정규 밀도와 누적분포다. 이 범위에서
$$
D^2_{S,\text{posterior reference}}\approx\frac{\gamma^2}{s_{[J]}^2}\cdot\frac{\Omega_S}{(1+\Omega_{[J]})(1+\Omega_{-S})}\cdot\mathbb E[\varphi(t_{-S})^2]>0,\qquad
\tau_{Z_S}-\tau=\beta\frac{\gamma v_{-S}}{s_{-S}}\mathbb E\!\left[\frac{\varphi(t_{-S})}{\Phi(t_{-S})\Phi(-t_{-S})}\right].
$$
기존 X 모드에서 참 L posterior-reference의 no-device $D^2$는 .003263이다. I와 X를 제외한 다른 작업 모형의 delta 근사는 .003807이다. 두 값은 학습한 encoder의 값이나 최소 하한이 아니다. Reference와 실제 learned-Z의 gap 일치는 성공 기준으로 강제하지 않는다.

## 4. 허용 예산

용한의 지시는 "넓게, 유연하게"다. 1차 구현이 위반 크기를 실측했으므로(§15.2) 격자를 그 크기에 맞춘다.

- 격자: $\delta^2\in\{0.005,\ 0.01,\ 0.02\}$, 즉 $\delta\in\{0.07,\ 0.10,\ 0.14\}$.
- 개발 기본값: $\delta^2=0.01$. 2단계의 벌점과 멈춤 조건은 이 값을 쓴다.
- 운영 규칙: Primary .01을 평가 전에 고정한다. .005와 .02는 같은 upper에 적용한 sensitivity로 모두 보고한다. S 통과 수로 primary를 다시 고르지 않는다. 인증·overlap·식별 조건을 충족하지 못하면 returned causal estimate는 없고 candidate diagnostic만 보고한다.
- 상한의 뜻: $\sigma=0.85$ phantom에서 장치 split을 원본 이미지 $q_1$으로 잰 gap은 $0.0106$(상한 $0.0141$)이었다. $\delta^2=0.02$에서는 이 split이 통과한다. 따라서 $\delta^2\ge0.015$에서의 통과에는 "위반을 걸러내지 못할 수 있음" 표시를 붙인다. 이 수치는 이 phantom의 것이고, 다른 설정에서는 다시 잰다.
- 편향: $\delta$가 커지면 §3.5의 한계가 커진다. 격자의 각 값에 한계를 함께 적는다.

## 5. 환자 집단 분할

같은 환자의 모든 조각은 같은 집단에 둔다. 집단은 여섯이다.

| 집단 | 역할 | 쓰는 변수 |
|---|---|---|
| R | 이미지 전용: MAE 사전학습, 고정 요약의 척도, reader | $W$만 |
| D | encoder와 두 predictor의 학습 | $W,A$ |
| C | 후보 선택: checkpoint, $\lambda$, 사다리 단계 | $W,A$ |
| P | 새 predictor 적합, AIPW의 nuisance 적합 | $W,A,Y$ |
| S | 최종 균형·overlap 검사 | $W,A$ |
| E | 효과 추정의 평가 표본 | $W,A,Y$ |

P와 S를 분리하는 이유: 같은 환자로 predictor를 적합하고 평가하면 독립 검사가 아니다. S와 E를 분리하는 이유: 선별에 쓴 자료로 효과를 평가하면 선택 편향이 생긴다. D, C, P, S, E는 논문처럼 회전한다. R은 회전하지 않는다. 1차 구현의 진단(§15)은 같은 scoring 환자를 다시 쓴 development pilot이고, 이 분할을 아직 따르지 않았다.

## 6. 1단계: 이미지만으로 준비한다 (집단 R)

### 6.1 영역과 인접 관계
해부학적 마스크로 $W_1,\dots,W_5$를 고정한다. 1차 구현은 타원 두 개(오른쪽 폐가 영상 왼쪽)를 직선 엽간열로 잘라 폐엽 다섯 개(224에서 363 픽셀)를 만들었고, 인접 관계는 픽셀 변을 공유하는 $\{1\text{-}2,\ 2\text{-}3,\ 4\text{-}5\}$다. 모든 폐엽이 비인접 영역을 둘 이상 가진다.

### 6.2 밝기 보정 (v2에서 추가)
1차 구현에서 모든 reader가 전역 밝기 $G$를 거의 완벽하게 읽었다($R^2=0.99$). reader의 첫 principal component는 $G$와 상관 $0.97$에서 $0.99$, $U$와는 약 $0.10$이었다. 그래서 one-factor 검사와 $\Omega$ 추정이 $U$가 아니라 $G$를 쟀다. 대응은 둘이다.
- 모든 폐엽 밖의 고정 body reference mask를 사용한다. Whole-lung mean을 사용하지 않는다. Raw/corrected 모두 image-derived reference scalar를 Z에 남긴다. Corrected encoder input만 이 scalar로 보정하고 원본 T를 raw audit에 남긴다. CT 단위 보정 자체가 모든 생리적 G 제거를 정당화하지 않는다. G가 confounder인 실제 자료에서는 reference 제거의 인과 조건을 별도로 점검한다.
- 진단은 two-factor로 넓힌다. tetrad 검사는 한 요인만 검정하므로, 보정 뒤에도 남는 공유 성분이 있으면 요인 둘로 적합해 본다.

### 6.3 MAE 사전학습
이미지 일부를 가리고 복원하는 Masked Autoencoder를 R의 모든 이미지로 학습한다. 가림 비율 0.75, patch 8×8. 1차 구현의 모형은 약 14만 모수이고 가린 픽셀의 MSE가 0.087(평균 영상 예측은 0.695)이었다. Local 실행에서는 결과 가중치를 encoder $\phi_S$에 복사한다. Raw predictor는 별도 작은 crop CNN을 새로 적합한다. 출처: He et al., CVPR 2022. 외부 pretrained weights는 받지 않는다.

### 6.4 고정 요약 $\chi_l$
영역마다 16개 숫자: 평균 강도, 분산, 분위수 8개(0.1에서 0.9), 임계값 이상 비율 4개, 고정 랜덤 사영 2개. 척도는 R에서 영역별로 맞추고 고정한다. §6.2의 밝기 보정 뒤에 계산한다.

### 6.5 영역 reader $\psi_k$ (보조)
입력 $W_k$, 출력 $d_\psi=8$. 손실은 비인접 영역의 고정 요약 예측
$$
\mathcal L_k(\psi_k)=\min_\theta\frac1{|R|}\sum_{i\in R}\sum_{l\in N(k)}\|\chi_l(W_{l,i})-\theta_{kl}^\top\psi_k(W_{k,i})\|^2 .
$$
쓰임새는 둘이다. 수준 A encoder(§8.1)의 입력, 그리고 predictor 사다리 1·2단계의 $T$쪽 특징. 판정에는 쓰지 않는다. 1차 구현에서 reader는 자기 영역의 부담 $L_k$를 $R^2$ $0.74$에서 $0.80$으로 읽었고, 다섯 reader를 합치면 $U$를 $R^2=0.79$로 읽었다(참 $L$ 전부로는 $0.87$). 영역 5 reader는 장치 $I$를 AUC $0.68$로 일부 읽었다(다른 영역은 $0.49$). 충분성 확인은 자기 영역의 요약 16개를 기준으로 한다. 1차 구현의 랜덤 특징 기준은 약해서 통과가 쉬웠다.

## 7. 2단계: 분할마다 교대로 학습한다 (집단 D, 선택은 C)

다섯 singleton 분할 각각에 대해 encoder를 따로 학습한다. 초기 가중치는 §6.3의 MAE에서 복사하고, 이후 갱신은 분할마다 독립이다.

### 7.1 구성
- 떼어 둔 쪽 $T_S$: 영역 $S$의 원 픽셀. 고정. 해당 환자의 이 영역 픽셀은 encoder 입력에서 제외한다.
- encoder $\phi_S$: 작은 CNN. 입력은 $V_S$(떼어 둔 영역을 0으로 가린 전체 영상). 학습하는 출력은 2차원이다. 고정 reference 좌표를 앞에 붙여 Local $Z_S$는 3차원이다. 4차원 bottleneck은 후속 비교다.
- decoder $d$: MLP. $Z_S$에서 다섯 영역의 고정 요약 전체($5\times16=80$개)를 복원한다.
- predictor $\widehat q_0$: $Z_S$를 받는 작은 MLP(sigmoid 출력, 범위를 강제하지 않음). $\widehat q_1$: 사다리 최상단, 즉 원본 픽셀 $T_S$를 CNN과 고정 high-pass 특징으로 읽고 $Z_S$를 이어 붙인 모형. $\widehat q_1$은 $\widehat q_0$의 logit에 $T$ 경로의 residual을 더하는 구조라 residual이 0이면 $\widehat q_0$와 같다.

### 7.2 목적함수
- 정보 보존: $L_{\mathrm{keep}}(\phi,d)=\sum_{l=1}^{5}\mathbb E\|d_l(Z_S)-\chi_l(W_l)\|^2$. 원 픽셀 복원은 비교군으로 유지한다.
- 균형 벌점: $\lambda\,(\widehat R_0-\widehat R_1-\delta^2)$, $\delta^2=0.01$.
- overlap 벌점: $\mu\mathbb E[(\widehat q_0-.95)_+^2+(.05-\widehat q_0)_+^2]$. 허용 구간 밖에만 작용한다. Predictor 자체는 proper Brier risk로 적합한다. Empirical tail 비율은 uniform population overlap을 인증하지 않는다.
- B단계 손실: $L_{\mathrm{keep}}+\lambda(\widehat R_0-\widehat R_1-\delta^2)+\mu\cdot\text{overlap}$. predictor는 이 손실로 학습하지 않는다.

### 7.3 고리
1. (A) $\phi,d$ 고정. $\widehat q_0$, $\widehat q_1$을 각자의 Brier 위험으로 D에서 학습한다. Epoch는 D 내부 validation으로 고른다. Local q0/reader는 최대 30 epoch, raw q1은 최대 60 epoch다. 선택 epoch, initial/selected validation risk와 cap을 기록한다. 이 재적합만으로 near-Bayes 정확성을 보장하지 않는다.
2. (B) predictor를 고정하고 encoder/decoder를 짧게 갱신한다. 그 뒤 새 encoder에서 두 predictor를 각각 다시 적합한다. C의 새 risk를 계산한 뒤에만 다음 판정으로 간다. Frozen old-predictor gap으로 멈춤·checkpoint를 정하지 않는다. Local 실행은 round당 20 step이다.
3. $\lambda$ 갱신: $\lambda\leftarrow\max\{0,\ \lambda+\rho_\lambda(\widehat R_0-\widehat R_1-\delta^2)\}$, 제약을 만족하면 $\lambda\leftarrow0.9\lambda$.
4. 멈춤: C에서 (i) $\widehat R_0-\widehat R_1\le\delta^2$, (ii) 범위 밖 비율 $\le1\%$, (iii) $L_{\mathrm{keep}}$이 세 바퀴 연속 $1\%$ 넘게 줄지 않음. 셋이 함께 성립하면 멈춘다. 최대 바퀴 수는 30이다.
5. 후보 선택: C에서 (i), (ii)를 만족하는 checkpoint 중 $L_{\mathrm{keep}}$이 가장 작은 것을 고른다. 선택 뒤 $\phi_S$를 고정한다.

### 7.4 기록
바퀴마다 $\widehat R_0$, $\widehat R_1$, $\widehat g$, $L_{\mathrm{keep}}$, $\lambda$, 범위 밖 비율, $\widehat q_1$의 검증 위험을 D와 C 각각에 대해 기록한다.

## 8. encoder 두 수준

### 8.1 수준 A (기준선)
$Z_S=B\,\psi_{-S}(W_{-S})$, $B$는 rank 2 선형. 논문의 함수군이고, 논문의 PROBE 코드를 고치지 않고 reader 출력을 블록으로 넣어 부른다. 1차 구현 결과는 §15.1이다. 이 수준은 논문의 bounded critic을 쓰므로 overlap 검사가 자동 통과한다. 그 통과를 overlap 인증으로 적지 않는다.

### 8.2 수준 B (이 계획의 대상)
§7의 신경망 $\phi_S$. 수준 B가 수준 A보다 낫다는 것은 통과 조건이 아니다. 신경망의 이점은 측정이 비선형인 별도 phantom(§10.3)에서 검사한다.

## 9. 3단계: 새 환자와 새 predictor로 검사한다 (집단 P, S)

### 9.1 절차
1. P에서 $\widehat q_0$, $\widehat q_1$을 새로 적합한다. $\phi_S$는 고정이다. epoch 선택은 P 안의 내부 validation으로 한다. S는 scoring에만 쓴다.
2. S에서 $\widehat g$와 $\mathrm{SE}$를 계산한다. 일반적인 다섯 검사에는 $z=\Phi^{-1}(1-0.05/5)$를 쓴다. 이번 Local 실행은 raw5/corrected5/trained1/low-sigma1의 $M=12$개이므로 $z=\Phi^{-1}(1-0.05/12)$다. $z\cdot\mathrm{SE}$는 정상근사 진단이다. Finite risk bound는 §16에 별도로 기록한다. S에서 critic이나 checkpoint를 다시 고르지 않는다.
3. 실제 고정 raw predictor를 fresh phantom에서 평가한다. $\pi=P(A=1\mid U,I)$와의 제곱오차 평균은 actual $\epsilon_1^2$의 보수적 상한이고 sampling radius를 추가한다. Reference epsilon은 승계하지 않는다. 이 상한은 irreducible proxy noise를 포함하여 느슨할 수 있다. 실제 자료에서는 matching bound가 없으면 경험적 검사다.
4. 격자의 각 $\delta^2$에 대해 통과 여부 $\max(\widehat g,0)+r+b_1\le\delta^2$와 overlap(범위 밖 비율 $\le1\%$)을 계산한다. 판정 $\widehat q_1$은 사다리 최상단이다.

### 9.2 predictor 사다리
- 1단계: 선형 지수의 probit. $T$쪽 입력은 reader 특징 $\psi_S(W_S)$.
- 2단계: 좌표별 spline(매듭 5개)의 additive probit. 입력은 1단계와 같다.
- 3단계(판정): 원 픽셀 $T_S$를 읽는 CNN과 고정 high-pass 특징에 $Z_S$를 이어 붙인 모형. high-pass 특징은 mask 정규화 blur를 뺀 뒤 경계 3픽셀을 제외한 제곱평균과 고정 분위수다. 특징은 이미지와 mask만으로 계산한다.
- 세 단계는 자동으로 포함 관계가 아니다. 각 단계의 $R_0$, $R_1$, $\widehat g$를 따로 기록한다. 1·2단계는 진단이고, 1차 구현에서 1단계(reader)의 gap이 3단계의 절반 이하였다. 모든 단계가 같은 신호를 놓쳤을 가능성은 남는다.

## 10. phantom

### 10.1 생성기 (Gaussian 기본형, image-only)
- 영상 $64\times64$, 2D. 다섯 영역은 §6.1의 마스크. 몸통 기준 조직 영역을 추가한다(§6.2).
- 중증도 $U\sim N(0,1)$. 영역 $k$의 병변 부담 $L_k=U+\sigma\varepsilon_k$. 병변은 Poisson($5e^{0.45L_k}$)개의 Gaussian 얼룩, 불투명도 $1.5\,\mathrm{sigmoid}(L_k)e^{0.35\xi}$, 위치와 크기는 무작위, 겹침은 포화한다.
- 전역 밝기 $G\sim N(0,1)$. 모든 픽셀에 $0.3G$를 더한다.
- 장치 표식 $I\sim\mathrm{Bern}(0.3)$. 영역 5에만 선 모양으로 그린다.
- 처치 $A=\mathbf 1\{1.5U+1.0I+\epsilon_A>0\}$. 결과 $Y=\tau A+1.0U+\epsilon_Y$, $\tau=1$. $X$ 항은 없다.
- 각 무작위 성분은 독립된 난수 흐름을 쓴다. 장치를 끄면 다른 것은 바뀌지 않는다.
- $n=24{,}000$. $\sigma=.3$에서 참 L 전부의 최적 U 예측 $R^2$는 $5/(5+.09)=.9823$이다. 실제 image→U 성능은 따로 측정한다. Local control은 §16의 n6000이며 전체 반복 grid는 아직 실행하지 않는다.

### 10.2 참값 계산
Fresh joint sample만으로 rawT/learnedZ의 조건부 e,p가 자동 계산되지 않는다. Local에서는 실제 predictor의 pi-error를 계산하고 finite sampling radius를 추가한다. 참 L의 analytic e,p는 별도 reference에만 쓴다. 기존 image-only S5 pilot의 gap은 raw .0106, 참 I positive control .0139였다. 이는 empirical gap이며 oracle D2 자체가 아니다.

### 10.3 비선형 변형
병변의 가시성이 포화하는 변형($L_k$ 대신 $\tanh(L_k)$로 그림)과, 결과가 $U$에 비선형인 변형($U$ 대신 $U+0.5U^2$). 수준 B의 이점은 여기서 본다.

### 10.4 비교군
수준 A, 수준 B(요약 복원), 수준 B(원 픽셀 복원), 전체 픽셀 CNN 보정(균형 없음), AIPW-(reader 전체), AIPW-(참 $L$ 전체), 역할을 준 proximal 2SLS(영역 두 묶음), oracle $U$, 단순 비교. AIPW-(참 $L$)은 참 영역 부담을 선형으로 보정한 한 estimator이고, 이미지 기반 estimator의 천장이 아니다.

### 10.5 성공 기준 (Gaussian 기본형)
1. Actual frozen predictor의 fresh pi-error upper, observational risk upper와 radius를 각각 기록한다. Bound가 느슨해 실패하면 predictor 실패와 구분한다.
2. 거짓 통과: 영역 5를 떼어 둔 분할의 통과율이 $\delta^2\le0.01$에서 5% 이하. $\delta^2=0.02$에서의 통과율은 보고만 한다.
3. 유효 분할의 통과: 영역 1부터 4를 떼어 둔 분할이 $\delta^2=0.01$에서 통과한다. 1차 구현은 이것을 아직 재지 않았다(§15.3).
4. 재적합 안정성: P에서 새로 적합한 predictor의 $\widehat g$가 D의 마지막 $\widehat g$와 $0.002$ 안에서 일치.
5. 효과 추정: $|\widehat\tau-\tau|$가 §3.5의 한계 안에 있고, 수준 B의 MAE가 수준 A의 1.2배 이하.
6. 닫힌 식은 같은 true-L reference 범위에서만 대조한다. Learned image encoder와 20% 일치를 요구하지 않는다.

## 11. 4단계: 추정, 집계, 보고 (집단 P, E)

- 통과한 분할마다 P에서 결과모형과 성향모형을 맞추고, 성향은 $[\eta,1-\eta]$, $\eta=0.05$로 자르며, E에서 AIPW 점수의 평균 $\widehat\theta_S$를 낸다.
- Plurality 집계는 실제 split의 common-target와 동시 radius 근거가 있을 때 별도로 실행한다. Reference bias 차이를 실제 neural split의 radius로 승격하지 않는다. Local 한 split loop에서는 집계를 실행하지 않고 각 candidate와 sampling SE를 보고한다. Propensity clipping은 수치 안정화이며 uniform overlap 보장이 아니다.
- 집계는 올바른 효과가 가장 큰 고유 무리를 이룰 때만 맞다. 여러 분할이 같은 방향으로 치우치면 집계가 그 치우침을 지우지 못한다. 1차 구현의 singleton 추정값은 모두 $1.30$에서 $1.45$로 같은 쪽에 있었다. 그래서 §3.5의 한계를 함께 보고한다.
- 보고 항목: 격자의 $\delta^2$마다 통과 분할과 $\widehat\tau$, 각 분할의 $\widehat g$, $r$, $b_1$(있으면), 범위 밖 비율, 사다리 단계별 $R_0,R_1,\widehat g$, $L_{\mathrm{keep}}$, 편향 한계, 그리고 phantom에서는 참값 대조.

## 12. 비용과 순서

1. 보정 전 다섯 singleton의 raw audit를 새 환자 P→S로 실행한다.
2. 같은 환자와 분할에서 fixed body-reference 보정 전후를 paired 비교한다. Stage1은 각 input mode의 R에서 따로 적합한다.
3. 2단계와 3단계 코드. 분할 하나로 고리를 검증한 뒤 다섯으로 늘린다. Mac에서 한다.
4. §10.5를 통과하면 §10.3의 비선형 변형.
5. 전체 격자($\sigma$ 다섯 값 × 반복 20회)는 DeltaAI에서 한다.

1차 구현의 시간(Mac M4, MPS): 본 실행 $n=24{,}000$이 675초였다(생성 4초, reader 80초, MAE 132초, 진단 77초, 수준 A 192초, oracle 계산 86초). 2단계 벤치마크는 한 바퀴에 5.7초에서 12.8초였고, 분할 하나 × rotation 하나(30바퀴)가 3분에서 6.4분, 데이터셋 하나(분할 5 × rotation 5)가 1.2시간에서 2.7시간, 3단계와 overhead를 더하면 2시간에서 4시간이다. $\sigma$ 네 값 × 반복 20회면 100시간에서 210시간이라 Mac으로는 안 된다. 첫 DeltaAI 실행은 $\sigma$ 다섯 값 × 반복 5회로 줄이고, 고리가 완성된 뒤 실측해서 명령과 GPU-hour 예산을 함께 승인받는다.

## 13. 검토에서 거둔 것과 남긴 것

- 거둠: "정확한 predictor는 저차원을 강제한다", "인코딩 뒤 분할은 held-out 독립을 보장하지 못한다"(가정을 바꾼 증명), "SCM-4가 (R1)의 근거다", "reader 공분산의 바닥으로 $\delta^2$를 인증한다", "$z\cdot\mathrm{SE}$가 predictor 오차를 덮는다", "AIPW-(참 $L$)의 값이 이미지 기반 estimator의 천장이다".
- 남김(범위를 좁혀): Brier 분해(§3.1), 떼어 둔 쪽 고정(§3.3), 식별의 승계(§3.4), 선형 Gaussian 범위의 바닥과 편향 식(§3.6, phantom 평가 전용).
- 증명 필요: §3.6의 근사 오차와 선형 encoder 중 최소성, 사다리의 sieve 일치성, 교대학습의 수렴(보장 없음, 경험적으로만 확인).

## 14. 이 계획이 주는 것과 주지 않는 것

- 주는 것: $T$는 학습이 고르지 않는다. predictor 정확성을 재학습, 새 환자, 원본 픽셀 predictor로 점검한다. 이 절차 자체가 near-Bayes 보장은 아니다. 통과식이 표본 오차와 predictor 오차를 분리한다. 허용 예산은 격자 전체에 대해 보고한다. 편향 한계는 필요한 가정과 함께 보고한다.
- 주지 않는 것: $L_{\mathrm{keep}}$은 정보 소실을 줄이는 보조 목표이고 중증도 보존을 보장하지 않는다. 완비성은 가정이다. 처치 전용 신호를 $Z$가 이미 포함하면 additional-information audit는 그 신호를 별도로 탐지하지 못할 수 있다. 실제 자료에서는 matching $b_1$이 없으면 인증 통과를 주장하지 않는다.

## 15. 1차 구현의 기록 (2026-10-08)

코드 `code/single_image/`(테스트 22개 통과), 결과 `results/single_image/phase1/`. 기존 파일은 바꾸지 않았다.

### 15.1 수준 A와 비교군 ($n=24{,}000$, $\sigma=0.85$, seed 0, 당시 $X$ 포함 모드)
| 방법 | 추정값 ($\tau=1$) |
|---|---|
| PROBE 수준 A (reader 출력) | 추정 거부 (통과 split 0개) |
| PROBE, 정답 reader (참 $L_k$) | 추정 거부 |
| split별 값, 수준 A / 정답 reader | $1.44$에서 $1.45$ / $1.30$에서 $1.31$ |
| AIPW-(X, 참 $L$) | $1.25$ |
| AIPW-(X, reader) | $1.38$ |
| AIPW-X | $2.28$ |
| 단순 비교 | $2.32$ |
| oracle AIPW-(X, $U$) | $0.99$ |

추정 거부는 저장된 screen의 결과다. .003263 reference를 rank-2 하한으로 사용하여 거부의 필연성을 주장하지 않는다. n4000에서 값이 나온 원인도 별도 검증 없이 검정력 하나로 단정하지 않는다.

### 15.2 predictor 강도에 따른 판정 (image-only 모드, 고정 encoder, 영역 5 split, scoring 4,000명)
| $\widehat q_1$ 입력 | Brier 위험 | gap $\widehat g$ | 경험적 상한 | $\delta^2=0.01$ 판정 |
|---|---|---|---|---|
| 압축 reader | $0.1551$ | $0.0049$ | $0.0071$ | 통과 |
| 원본 이미지 + high-pass + reader | $0.1494$ | $0.0106$ | $0.0141$ | 탈락 |
| 참 $I$ positive control + reader | $0.1461$ | $0.0139$ | $0.0179$ | 탈락 |

$\widehat q_0$의 Brier 위험은 $0.1601$이고 예측의 $4.65\%$가 $[0.05,0.95]$ 밖이었다. 이 표는 같은 scoring 환자를 다시 쓴 development pilot이다.

### 15.3 당시 재지 않은 것과 현재 상태
1차 구현에서는 영역 1부터 4를 떼어 둔 split의 raw-image gap을 재지 않았다. 이번 local 실행에서 raw/corrected 다섯 split을 모두 측정했다(§17). 다섯 neural split의 교대학습, role 전체 회전, plurality, 반복 grid와 비선형 비교는 아직 수행하지 않았다.

### 15.4 정정
v1 작성 직후의 보고에서 "장치 split의 gap $0.0162$가 잡혔다"고 적었다. 그 값은 참 $I$를 넣은 positive control의 값이었다. 실제 reader $q_1$의 gap은 $0.0046$이었고, 장치 split은 원본 이미지 $q_1$으로 바꾼 뒤에야 탈락했다.

## 16. 2026-10-09 승인된 local 실행 계약

- Deliverable: image-only raw/corrected 5split audit, corrected S1 실제 교대학습, 독립 audit와 candidate effect, low-sigma control.
- Consumer/Form: 용한의 연구 판단을 위한 code와 `results/single_image/pipeline_v2_local.json`, `.log`.
- Acceptance: R6000/D-C-P-S-E각3600의 patient 분리, P 내부 validation→S scoring, B후criticrefit→C선택, E outcome 평가, 보호15개 hash 유지, tests 통과.
- 실행: n24000, seed20261009, 최대30round×20steps, primary .01 고정. S/E를 보기 전에 30round으로 확정했고 C 멈춤 규칙은 유지한다. Low-sigma만 n6000 별도seed. Stage1 reader30/MAE10, critics30/raw60 최대; 내부 validation의 epoch와 cap을 기록한다. Local keep loss는 80개 표준화 summary의 평균 제곱오차다.
- Finite bound: M12, alpha .05. S risk radius는 $\sqrt{2\log(2M/\alpha)/n_S}$, fresh pi-error m5000의 radius는 $\sqrt{\log(2M/\alpha)/(2m)}$. 같은 upper를 세 cutoff에 적용한다. 작은 cutoff 인증은 radius만으로 실패할 수 있고, 이를 method 실패로 해석하지 않는다.
- Evidence: actual command, source/protected SHA256, patient/role hash, loop history, validation/scoring risk, empirical normal upper와 finite upper, overlap tails, candidate AIPW와 sampling SE.
- Boundaries: main의 관련 project만 수정. 기존 phase1 JSON/PT, sealed7개와 v1 보존. External/DeltaAI, 새 dependency, Git stage/commit/push, unrelated/memory mutation 없음.
- PERR mapping: snapshot→regression tests→5split raw→paired brightness→S1 loop→fresh audit/E 비교군→low-sigma→hash/source QA. Cap은 성공이 아니며 suitable checkpoint와 diagnostic fallback을 구분한다. 인증/식별/overlap 미충족이면 returned causal estimate는 None이다.

## 17. 실제 local 결과 (2026-10-09)

명령과 source hash는 `results/single_image/pipeline_v2_local.json`의 provenance에 있다. 진행 기록은 같은 이름의 `.log`다. CPU 실행은 2,462.9초(41.05분) 걸렸다. 기존 phase1 JSON/PT, sealed7개와 v1의 보호15개 hash는 유지했다. Image-only $n=24{,}000$, seed20261009, R6000과 D/C/P/S/E각3600이다. $\sigma=.3$ control만 $n=6{,}000$ 별도 seed다.

### 17.1 보정 전후와 다섯 split

| Held region | Raw gap | Corrected gap | Corrected normal upper | .01 gap-only 판정 |
|---|---:|---:|---:|---|
| S1 | .000468 | .001083 | .004133 | 통과 |
| S2 | .003272 | .003822 | .006489 | 통과 |
| S3 | .003233 | .002716 | .005213 | 통과 |
| S4 | .004748 | .004433 | .007350 | 통과 |
| S5 | .012963 | .012230 | .016228 | 탈락 |

위 표는 $M=12$ Bonferroni normal approximation이다. 실제 $D^2$나 인과 식별 인증이 아니다. Raw/corrected 모두 $\delta^2=.005$에서 1개, $.01$에서 4개, $.02$에서 5개가 gap만 통과한다. $\widehat q_0$ tail은 raw3.64–8.44%, corrected7.67–8.81%여서 사전 1% 조건은 모두 실패한다. S5의 참 $I$ positive control gap은 raw.015023/corrected.014389이다. 실제 raw $\widehat q_1$ 값과 구분한다.

같은 P-fit/S-score의 reader-feature 선형 평가에서 $U$의 $R^2$는 .7921→.8221, $G$의 $R^2$는 .9975→−.0065였다. 보정은 이 평가에서 밝기 의존을 줄였다. 이 값은 neural $Z$의 중증도 보존 성능이 아니다. 음의 $R^2$는 해당 선형 평가가 평균 예측보다 못하다는 뜻이다. 원본 reference 좌표는 $Z$에 남긴다.

### 17.2 교대학습과 독립 검사

S1은 30round×20step=600 encoder step을 실제 수행했다. 모든 B 뒤 새 encoder에서 critic을 재적합했다. C keep loss는1.0467→.5310으로 줄었다. 이는 요약 복원 목적의 개선이다. $U$ 보존이나 인과 오차 감소를 보장하지 않는다. C에서 balance와 1% tail을 함께 만족한 checkpoint는 없었다. 30cap에 도달했고 round30을 diagnostic fallback으로 고정했다.

| 독립 S의 값 | Corrected fixed-reader S1 | Neural S1 |
|---|---:|---:|
| $\widehat q_0$ Brier | .161266 | .166202 |
| $\widehat q_1$ Brier | .160183 | .164165 |
| Gap | .001083 | .002037 |
| Normal upper | .004133 | .005329 |
| $\widehat q_0$ tail | 7.97% | 6.81% |
| Fresh $\widehat q_0$ $\pi$-MSE | .035413 | .041133 |
| Fresh $\widehat q_1$ $\pi$-MSE | .032032 | .036454 |

Neural $\widehat q_0$는 P 내부 validation에서 epoch29/최대30을 선택했다. Reader $\widehat q_1$은 epoch9를 선택했다. Raw residual은 epoch0을 선택했다. 따라서 최종 $\widehat q_1$은 reader prediction을 유지했다. Raw 입력은 있지만 추가 raw 학습효과를 선택하지 못했다. 두 predictor의 Bayes 근접은 미확인이다. $Z$가 바뀌므로 위 위험 차이를 정확한 approximation error 차이로 해석하지 않는다. 독립 prediction이 개선된 결과도 아니다.

선택된 같은 encoder의 Cgap은 .002617, 새 P-critic/Sgap은 .002037이다. 차이 −.000580은 표본과 refit을 함께 포함한다. Neural fresh $\pi$-error 평균의 SE는 $\widehat q_0$ .000940/$\widehat q_1$ .000891이다. 이 $\pi$-error는 latent uncertainty를 포함한 보수적 양이고 정확한 $\epsilon_b^2$가 아니다.

Neural finite upper는 .121904다. S risk radius .058565, $\pi$-error radius .024847과 error mean을 합친 값이다. 모든 12개 모델은 $\delta^2=.005/.01/.02$ finite gate를 실패했다. Sampling radius만으로도 primary를 초과한다. 이는 $D^2$가 크다는 증명이 아니다. Returned causal estimate는 None이다.

### 17.3 같은 E의 효과 후보와 대조

| 방법 | Candidate | Sampling SE |
|---|---:|---:|
| Fixed-reader S1 | 1.44095 | .04591 |
| Neural S1 | 1.47219 | .05083 |
| Full-image reader 보정 | 1.38642 | .05060 |
| 참 $L$ 보정, 평가 전용 | 1.27751 | .04517 |
| Known $U$ propensity/outcome, 평가 전용 | 1.00598 | .04632 |
| 같은 E의 단순 비교 | 2.38051 | .04337 |

참 효과는 $\tau=1$이다. Neural의 효과 오차 감소는 관측되지 않았다. 두 candidate 차이의 유의성이나 보편적 열등함을 주장하지 않는다. Fitted outcome nuisance는 P의 arm별 선형 모형이다. 남는 오차를 residual confounding 하나로 분해하지 않는다. 참 $L$ 보정값은 population target이나 모든 estimator의 정보 천장이 아니다. Known $U$는 P nuisance fitting 없이 E에서 평가했다. Sampling interval은 causal confidence interval이 아니다.

### 17.4 실패의 해석과 남은 것

- Overlap gate: known-$U$의 E tail은25.69%다. Population reference $e_U(u)=.7\Phi(1.5u)+.3\Phi(1.5u+1)$의 tail은23.835%다. 두 값을 learned-$Z$ tail과 구분한다. 좋은 severity 정보와 정확한 $\widehat q_0$ 자체도 이 DGP의 1% gate를 만족하지 않을 수 있다.
- Loss–gate 차이: 고정 $\mu=10$ hinge는 구간 밖의 거리를 벌한다. Tail 비율을 강제하지 않는다. 모든 $\widehat q_0$가 $.95+\varepsilon$이면 tail은100%이나 벌점은 $10\varepsilon^2\to0$이다. 이는 실패 원인 후보다. 실제 minibatch의 gradient 주원인을 입증한 결과는 아니다.
- Pretraining: corrected MAE hidden-pixel MSE=.2028은 mean-image=.1835보다 못했다. Low-sigma에서도 .5468은 .1589보다 못했다. 사전학습 성공으로 표시하지 않는다.
- Low-$\sigma$: reader→$U$ 선형 평가 $R^2=.91765$, 참 $L$ population reference=.98232다. 참 $L$의 유한 표본 평가는 .9842였다. Neural-$Z$의 값은 아니다. Low-$\sigma$ S1의 gap=.002672/normal upper=.006260이나 $\widehat q_0$ tail17.11%와 finite upper.166385로 qualification은 실패했다.
- 다음 대조는 고정된 neural $Z$에서 fresh predictor를 더 충분히 적합하는 검사와, 사전 지정한 bounded-treatment DGP에서 overlap 충돌을 분리하는 검사다. 이번 실행 뒤 architecture, threshold나 처치 배정은 바꾸지 않았다. ORC/stability, plurality, 전체 neural5split/rotation/반복 grid와 실제 CT 자료 검증은 남아 있다.

실행 후 known-oracle P-fit 표시만 정정했다. 수치4,633개는 그대로다. JSON은 실제 실행 source hash와 metadata 정정 후 current source hash를 따로 보존한다. 테스트는 본 실행 전29개 통과, metadata 수정 후 관련8개 통과, 최종 전체30개 통과(72.20초)했다.

## 18. Low-$\sigma$ 효과 후보 보완과 진단 정정 (2026-10-09)

### 18.1 보완한 범위

- Deliverable: 기존 low-$\sigma$ 고정 S1을 재현하고 누락된 효과 후보와 대조를 계산한다.
- Consumer/Form: 용한의 연구 판단을 위한 새 code, JSON/log와 이 진단이다.
- Acceptance/Evidence: 기존 $Z$, $\widehat q_0$, $\widehat q_1$, S 환자 fingerprint와 위험의 일치, P→E 분리, 후보·SE·보호 hash와 회귀검사다.
- Boundaries: 기존 image-only SCM, fold, seed, threshold, 학습 설정을 유지한다. Low-$\sigma$ neural loop, 새 DGP, $n=96{,}000$, 새로운 bound, MPS/외부 실행이나 원본 덮어쓰기는 포함하지 않는다.

원래 `run_local`의 low-$\sigma$ 분기에는 `effects` 호출이 없었다. Overlap gate가 효과 후보를 숨긴 것이 아니다.
새 `run_low_sigma_effects`는 후보 계산을 gate와 독립으로 수행한다. `effects`에 neural 모델이 없으면 고정 표현과 대조만 계산한다. 기존 main의 baseline/trained 순서와 수치 연산은 유지했다.

기존 $n=6{,}000$, seed20271010, $\sigma=.3$, $A=\mathbf1\{1.5U+I+\epsilon_A>0\}$를 유지했다. 이미지 밖의 $X$ 입력과 처치·결과 원인은 없다. R1500과 D/C/P/S/E각900이다. Stage1 seed는20271012, 고정 S1 seed는20271110, P predictor seed는20271210이다. Fresh phantom5000의 seed는20341010이다. S는 이전에 읽은 development sample을 재사용했다. E는 원래 low-$\sigma$ 실행에서 효과를 평가하지 않은 환자다. 어떤 E 정보도 모델 선택에 쓰지 않았다. $U,L$은 평가 전용 대조에서만 사용했다.

명령은 code 디렉토리에서 다음과 같다.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -B -m single_image.run_low_sigma_effects --n 6000 --seed 20271010 --reader-epochs 30 --mae-epochs 10 --critic-epochs 30 --raw-epochs 60 --mc 5000 --out-json ../results/single_image/pipeline_v2_low_sigma_effect_followup.json --log ../results/single_image/pipeline_v2_low_sigma_effect_followup.log
```

### 18.2 실제 결과

CPU 실행은109.69초(1.83분), peak RSS1079.13MiB였다. 기존 low-$\sigma$의 $Z$, 두 predictor와 S 환자 hash가 모두 일치했다. Gap, SE, predictor 차이 제곱평균과 두 Brier 위험의 차이는 모두0이었다. 기존 보호15개와 원 pipeline JSON/log의 총17개 hash도 같았다. 원 pipeline JSON SHA-256은 `7440581c974304008deb58d9fa770ef0eee503e1a258887f4d10e5f6e763cfb2`로 유지했다.

| 같은 E900의 방법 | Candidate | Sampling SE |
|---|---:|---:|
| 고정 표현 S1 | 1.23372 | .11053 |
| Full-image reader 보정 | 1.01477 | .10184 |
| 참 $L$ 보정, 평가 전용 | .95677 | .08944 |
| Known $U$ propensity/outcome, 평가 전용 | .90418 | .08969 |
| 단순 비교 | 2.29617 | .08535 |

참 효과는 $\tau=1$이다. 고정 S1의 sampling interval은 [1.01709, 1.45035]다. Known-$U$ interval은 [.72839, 1.07998]다. 이는 sampling 진단이고 causal confidence interval이 아니다. Full-reader point estimate가 더 가깝지만 paired 차이의 interval이 없어 통계적 우월함을 주장하지 않는다. $\sigma=.85$의 E3600과 이번 E900은 다른 표본이다. 두 실행의 point error 차이를 noise 감소의 인과 효과나 반복 실험 결과로 해석하지 않는다.

S의 $\widehat q_0$ tail17.11%, gap.002672/normal upper.006260/finite upper.166385는 그대로였다. E의 고정 S1 tail은15.44%다. Candidate가 존재해도 accepted return은 None이다. Low-$\sigma$ neural 결과는 없다. 결과 JSON/log는 `results/single_image/pipeline_v2_low_sigma_effect_followup.json`과 같은 이름의 `.log`다. 신규5개 테스트와 전체35개 테스트가 통과했다(각2.76초/65.16초). Python 실행 source hash, args, 역할 hash와 보호 hash를 새 JSON에 기록했다.

### 18.3 붙여 준 진단의 범위 정정

1. **Reference 편향은 모든 encoder의 하한이 아니다.** $\sigma=.85$에서 pure-probit true4-$L$ posterior reference는1.37372, $\sigma=.3$에서는1.06486이다. 이 범위의 bias 식은 exact expectation이며 코드의 수치는 Gaussian quadrature로 계산한다. 같은 절의 $D^2$ 식은 delta-method 근사다. 이번 $I$ 포함 처치 모형에서 $I$를 평균낸 true4-$L$ reference는 각각1.34536/1.05864로 다르다. Cor3.3의 고정 posterior 표현과 split 비교를 모든 neural $\phi$나 ATE estimator로 확대하지 않는다. Clipped finite AIPW가 이 population reference와 정확히 같아야 하는 것도 아니다. Reader $R^2=.8221$은 다섯 reader 전체의 선형 평가여서 true4-$L$과 직접 bias를 분해하지 않는다. Neural candidate 차이.03을 개별 SE.05와 비교해서 두 방법이 같다고 결론 내리지 않는다.
2. **Overlap 충돌은 모형 의존적이다.** 이번 phantom은 $\Phi(1.5U+I)$를 썼다. Sealed simulation의 $.1+.8\Phi$는 실제로 다른 link다. Bounded control은 별도 실험이고 기존1.37/1.06 reference를 그대로 기대값으로 쓰지 않는다. 모든 encoder가1% gate를 실패한다는 주장도 틀리다. 상수 $Z$의 처치확률은 약.5631로 tail0이다. 이는 좋은 causal representation이라는 뜻은 아니다.
3. **Latent-propensity 상한은 유효하지만 느슨하다.** 실제 frozen $\widehat q_1$과 $\pi=P(A=1\mid U,I)$의 MSE는 $\epsilon_1^2$의 보수적 상한이다. 잘못된 target으로 계산한 것이 아니다. 현재 renderer에서 어떤 image predictor도 $\mathbb E\operatorname{Var}(\pi\mid L_{1:5},I)$보다 작은 $\pi$-MSE를 가질 수 없다. 이 값은 $\sigma=.85$에서.017375, $\sigma=.3$에서.002544다. 따라서 현재 상한 방식으로 $\sigma=.85$의 $\delta^2=.01$ 인증을 얻으려는 문제는 표본 증가만으로 해결하지 못한다. 이것은 실제 $D^2$, predictor approximation error나 ATE 식별의 하한이 아니다. Fresh joint draw 백만 개도 raw-$T$/learned-$Z$의 $p(T,Z)$를 자동 제공하지 않는다. Auxiliary regression의 오차 통제가 별도로 필요하다. 이번 보완은 bound를 변경하지 않았다.
4. **예측 위험과 pretraining 결과만으로 원인을 확정하지 않는다.** Neural $\widehat q_0$의 높은 Brier 위험은 $U$ 정보가 줄었다는 증명이 아니다. Treatment-only 정보와 approximation error가 섞인다. Raw residual epoch0은 추가 학습이 validation에서 이득을 주지 않았다는 결과다. Reader가 충분한지, raw 학습이 부족한지 구분하지 못한다. 옛 MAE .086717/mean-image.695430은 실제 phase1 결과에 있지만 raw·40epoch·MPS·seed0이다. 현재 corrected·10epoch·CPU와 같은 protocol의 paired 비교가 아니다. 현재 mean-image 기준 미달은 인정하되 encoder 손상의 인과 원인으로 단정하지 않는다.
5. **데이터 생성 시간은 전체 훈련비가 아니다.** 현재 fold 비율에서 $n=96{,}000$이면 R24000과 D/C/P/S/E각14400이다. 각16000이 아니다. Float32 원본 이미지 한 배열만1.465GiB이고 corrected copy가 더 필요하다. 출력 $Z$가3차원이어도 입력 CNN은 $(B,2,64,64)$ tensor를 처리한다. Reader, MAE와 predictor의 epoch 비용도 sample 수와 함께 늘어난다. 고정600 encoder minibatch step을 유지하면 각 D 환자의 평균 노출은 줄어든다. 아직 $n=96{,}000$을 실행하거나 그 비용을 실측하지 않았다.
