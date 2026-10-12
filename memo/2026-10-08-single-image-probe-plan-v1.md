# 단일 이미지 PROBE 계획 v1

**대체됨.** 2026-10-08 CDT 저녁에 `2026-10-08-single-image-probe-plan-v2.md`가 이 문서를 대체했다. 이 파일은 기록으로만 남긴다.

작성: 2026-10-08 CDT. 작성자: `single-proxy-balancing` (수학 검토: `math-theory-agent`). 승인: 용한, 2026-10-08 ("계획을 상세히 업데이트 해. 승인.").

이 문서는 계획이다. 코드, PRD, 실험은 아직 없다. 다음 산출물은 phantom 생성기와 1단계의 PRD이고, 그 PRD는 별도 승인을 받는다.

이 계획의 대화 기록은 세 차례의 외부 검토(2026-10-07, 10-08)를 거쳤다. 거둔 주장과 남긴 주장은 §13에 적는다.

---

## 0. 한 문장 목표

환자마다 CT 한 장 $W$만 있을 때, $W$에서 남는 쪽 표현 $Z$와 떼어 둔 쪽 $T$를 만들어 PROBE를 적용한다. 두 predictor가 각자의 참 처치확률에 가깝고, 실제 불균형이 허용 예산 안에 있으며, 이미지 변환이 식별에 필요한 proxy 조건을 보존할 때, $Z$로 보정한 효과를 추정하고 남는 편향을 한계와 함께 보고한다.

## 1. 기호

- 관측: $O=(X,W,A,Y)$. $A\in\{0,1\}$ 처치, $Y\in\mathbb R$ 결과, $X$ 관측 공변량(없을 수 있음), $W\in\mathbb R^{p}$ CT 한 장의 픽셀 벡터.
- 숨은 변수: $U$ 중증도. $G$ 전역 요인(획득 조건, 체격). $I$ 처치 전용 원인(장치 표식 등).
- 잠재결과 $Y(a)$, 목표 $\tau=\mathbb E[Y(1)-Y(0)]$.
- 영역: 분석 전에 고정한 픽셀 분할 $W=(W_1,\dots,W_J)$, 기본 $J=5$(폐엽). 영역 $k$와 인접하지 않은 영역의 집합 $N(k)$.
- 분할 $S$: 떼어 둘 영역의 집합. 기본은 $|S|=1$인 다섯 개. 남는 쪽 픽셀 $V_S=W_{-S}$, 떼어 둔 쪽 $T_S=(X,W_S)$는 원 픽셀.
- encoder $\phi_S$: $Z_S=\phi_S(V_S)\in\mathbb R^{d_Z}$, 기본 $d_Z=2$.
- 참 처치확률: $e(z)=P(A=1\mid Z_S=z)$, $p(t,z)=P(A=1\mid T_S=t,Z_S=z)$.
- 실제 불균형: $D^2=\mathbb E[\{p(T_S,Z_S)-e(Z_S)\}^2]$.
- predictor: $\widehat q_0(Z_S)$는 $e$의 추정량, $\widehat q_1(T_S,Z_S)$는 $p$의 추정량. 오차 $\epsilon_0^2=\mathbb E[(\widehat q_0-e)^2]$, $\epsilon_1^2=\mathbb E[(\widehat q_1-p)^2]$.
- Brier 위험 $R_0=\mathbb E[(A-\widehat q_0)^2]$, $R_1=\mathbb E[(A-\widehat q_1)^2]$. 독립 환자에서 계산한 표본 차이 $\widehat g$.
- 허용 예산 $\delta^2$. 조정값 $\tau_Z=\mathbb E[\mathbb E(Y\mid A=1,Z)-\mathbb E(Y\mid A=0,Z)]$.
- 고정 요약 $\chi_l(W_l)\in\mathbb R^{16}$: 영역 $l$의 학습하지 않는 통계량.
- 영역 reader $\psi_k(W_k)\in\mathbb R^{d_\psi}$: 이미지만으로 학습하는 영역별 요약. 보조 부품이다(§6.4).

## 2. 가정

- (M1) 생성 구조. $W_k=f_k(U,G,\varepsilon_k)$, $A=a(U,G,X,I,\epsilon_A)$, $Y(a)=y(a,U,G,X,\epsilon_Y)$. 영역 잡음 $\varepsilon_1,\dots,\varepsilon_J$는 $(U,G,X)$가 주어지면 서로 독립이고 $(I,\epsilon_A,\epsilon_Y)$와 독립이다. 처치 전용 원인 $I$가 영상에 나타나는 경우 그 위치는 알려져 있지 않다.
- (M2) 중증도 가시성. 모든 $k$에 대해 $f_k$는 $u$에 대해 상수가 아니다. 미만성 질환을 뜻한다. 국소 질환은 범위 밖이다.
- (A3) outcome-relevant completeness. 논문 가정 3을 $T_S=(X,W_S)$에 대해 가정한다. 데이터로 검정하지 않는다.
- (A4) stability. 논문 가정 4의 상수 $\Gamma_S$가 유한하다고 가정한다. 편향 한계에만 쓴다.
- 가정하지 않는 것. 영역 reader나 encoder가 $U$를 복원한다고 가정하지 않는다. 선형 Gaussian 작업 모형의 결과를 신경망 encoder로 확대하지 않는다.

## 3. 수학적 뼈대

### 3.1 Brier 분해
$$
R_0-R_1=D^2+\epsilon_0^2-\epsilon_1^2 .
$$
근거: $A-e=(A-p)+(p-e)$의 직교성과 $R_b=\mathbb E(A-\text{참})^2+\epsilon_b^2$. 원고 부록 E(line 207 근처)의 분해와 같은 기반이다.

### 3.2 통과식
모집단 위험 차이의 상한을 $\widehat g+r$로 통제하고 $\epsilon_1^2\le b_1$의 근거가 있으면 $D^2\le\widehat g+r+b_1$이다. 보수적 통과 조건은
$$
\max(\widehat g,0)+r+b_1\le\delta^2 .
$$
$r$은 표본 오차의 상한이고 $z\cdot\mathrm{SE}$로 잡는다. $b_1$은 두 번째 predictor가 놓친 신호의 상한이다. $z\cdot\mathrm{SE}$는 $b_1$을 대신하지 않는다. $b_1$을 댈 수 없으면 결과를 **경험적 통과**로 보고하고, 댈 수 있으면 **인증 통과**로 보고한다.

### 3.3 떼어 둔 쪽은 고정한다
$Z$를 고정하고 $T'=h(T,Z)$가 임의의 가측 함수이면 $D^2(T')\le D^2(T)$다. 따라서 균형 목적함수로 $T$를 고르면 해는 상수 $T$다. $T_S$는 분석 전에 고정하고 어떤 학습 목적함수로도 바꾸지 않는다.

### 3.4 식별의 승계
(M1), (M2) 아래에서 모든 분할 $S$에 대해 논문 가정 2가 $U^*=(U,G)$로 성립한다. $A\perp W\mid(U,G,X)$인 모형에서는 임의의 결정적 변환에 대해 held-out 독립이 성립한다. $I$가 영상 안에 있는 모형에서는 $T_S$가 $I$와 독립인 영역 픽셀일 때 성립한다. (A3), 균형, overlap이 더해지면 논문 정리 1에 의해 $\tau=\tau_{Z_S}$다.

### 3.5 편향 한계
(A3), (A4), overlap $\eta\le e\le 1-\eta$, $D\le\delta$ 아래에서
$$
|\tau_{Z_S}-\tau|\le\frac{\Gamma_S\,\delta}{\eta(1-\eta)} .
$$
허용 예산을 넓히면 이 한계도 비례해 넓어진다. 한계는 추정값과 함께 보고한다.

### 3.6 작업 모형의 닫힌 식 (phantom 평가 전용)
선형 Gaussian 한 요인 모형($\psi_k=\lambda_kU+\sigma_k\xi_k$), probit 처치($A=\mathbf 1\{\gamma U+\epsilon_A>0\}$), 선형 결과($\beta U$), 남는 측정값의 선형 encoder라는 범위에서, 정밀도 $\Omega_K=\sum_{k\in K}\lambda_k^2/\sigma_k^2$, $v_K=1/(1+\Omega_K)$, $s_K=\sqrt{1+\gamma^2v_K}$에 대해
$$
\inf_{\phi\ \text{선형}}D^2_S\approx\frac{\gamma^2}{s_{[J]}^2}\cdot\frac{\Omega_S}{(1+\Omega_{[J]})(1+\Omega_{-S})}\cdot\mathbb E[\varphi(t_{-S})^2]>0,\qquad
\tau_{Z_S}-\tau=\beta\frac{\gamma v_{-S}}{s_{-S}}\mathbb E\!\left[\frac{\varphi(t_{-S})}{\Phi(t_{-S})\Phi(-t_{-S})}\right].
$$
이 식은 phantom에서 생성기의 참 모수로 계산해 실측과 대조하는 평가 도구다. 이미지 공분산만으로는 처치 신호의 불균형을 알 수 없으므로, 이 식으로 실제 자료의 $\delta^2$를 인증하지 않는다. 신경망 encoder의 하한으로도 쓰지 않는다. 근사 오차와 "선형 encoder 중 최소"의 증명은 남아 있다.

## 4. 허용 예산: 유연하게, 그러나 보고한다

용한의 지시: 허용 예산은 넓게, 유연하게 잡는다.

- 격자: $\delta^2\in\{0.01,\ 0.02,\ 0.04\}$, 즉 $\delta\in\{0.10,\ 0.14,\ 0.20\}$.
- 개발 기본값: $\delta^2=0.02$. 2단계의 벌점과 멈춤 조건은 이 값을 쓴다.
- 운영 규칙: 3단계에서 격자의 모든 값에 대해 통과 여부를 계산한다. 보고는 격자 전체에 대해 한다. 대표값은 "다섯 singleton 분할 중 셋 이상이 통과하는 가장 작은 $\delta^2$"로 두고, 그 값이 없으면 인증 추정값 없음으로 보고한다.
- 상한의 뜻: $\delta^2=0.04$는 처치 전용 신호의 불일치(§10.2의 예시 모형에서 $0.033$)보다 크다. 그 값에서는 장치 표식이 든 분할이 통과할 수 있다. 따라서 $\delta^2\ge0.03$에서의 통과는 "위반을 걸러내지 못할 수 있음"이라는 표시를 함께 붙인다. 이 수치는 해당 모형의 것이고, phantom에서 거짓 통과율로 확인한 뒤에 쓴다.
- 편향: $\delta$가 커지면 §3.5의 한계가 커진다. 격자의 각 값에 대해 한계를 함께 적는다.

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

P와 S를 분리하는 이유: 같은 환자로 predictor를 적합하고 평가하면 독립 검사가 아니다. S와 E를 분리하는 이유: 선별에 쓴 자료로 효과를 평가하면 선택 편향이 생긴다. D, C, P, S, E는 논문처럼 회전한다. R은 회전하지 않는다.

## 6. 1단계: 이미지만으로 준비한다 (집단 R)

### 6.1 영역과 인접 관계
해부학적 마스크로 $W_1,\dots,W_5$를 고정한다. 인접 관계를 고정한다. 데이터를 본 뒤 바꾸지 않는다.

### 6.2 MAE 사전학습
이미지 일부를 가리고 복원하는 Masked Autoencoder를 R의 모든 이미지로 학습한다. 가림 비율 0.75, patch 8×8(64×64 영상 기준). 결과 가중치는 encoder $\phi_S$와 CNN predictor의 초기값이다. 출처: He et al., CVPR 2022 `[Axiom, 미확인: Zotero에서 확인 필요]`.

### 6.3 고정 요약 $\chi_l$
영역마다 16개 숫자: 평균 강도, 분산, 분위수 8개(0.1에서 0.9), 임계값 이상 비율 4개, 고정 랜덤 사영 2개. 척도는 R에서 영역별로 맞추고 고정한다. 학습 대상이 아니다.

### 6.4 영역 reader $\psi_k$ (보조)
입력 $W_k$, 출력 $d_\psi=8$. 손실은 비인접 영역의 고정 요약 예측
$$
\mathcal L_k(\psi_k)=\min_\theta\frac1{|R|}\sum_{i\in R}\sum_{l\in N(k)}\|\chi_l(W_{l,i})-\theta_{kl}^\top\psi_k(W_{k,i})\|^2 .
$$
쓰임새는 둘이다. 수준 A encoder(§8.1)의 입력, 그리고 predictor 사다리 1·2단계의 $T$쪽 특징. 충분성 확인: $\psi_k$로 예측한 손실이 사전학습 특징 $g_k\in\mathbb R^{512}$로 예측한 손실의 1.05배 이하이면 $d_\psi=8$을 받아들이고, 아니면 16으로 올린다. 한 요인 구조 확인: 첫 주성분의 공분산에서 tetrad 잔차가 부트스트랩 표준오차의 2배 안인지 본다. 이 확인들은 진단이고 인증이 아니다.

## 7. 2단계: 분할마다 교대로 학습한다 (집단 D, 선택은 C)

다섯 singleton 분할 각각에 대해 encoder를 따로 학습한다. 초기 가중치는 §6.2의 MAE에서 복사하고, 이후 갱신은 분할마다 독립이다. 공유 backbone을 쓰지 않는 이유: backbone을 갱신하면 모든 분할의 표현이 바뀌어 두 predictor를 모든 분할에서 다시 적합해야 하고, 분할별 실패 원인을 구분하기 어렵다.

### 7.1 구성
- 떼어 둔 쪽 $T_S$: 영역 $S$의 원 픽셀(과 $X$). 고정. 해당 환자의 이 영역 픽셀은 encoder 입력에서 제외한다.
- encoder $\phi_S$: 작은 CNN. 입력은 $V_S$(떼어 둔 영역을 0으로 가린 전체 영상). 출력 $Z_S\in\mathbb R^{d_Z}$, $d_Z=2$ 기본, 4를 변형으로.
- decoder $d$: MLP. $Z_S$에서 다섯 영역의 고정 요약 전체($5\times16=80$개)를 복원한다. 떼어 둔 영역의 요약도 목표에 넣는다.
- predictor $\widehat q_0$: $Z_S$를 받는 bounded probit 또는 작은 MLP. $\widehat q_1$: $(T_S,Z_S)$를 받는 모형. 사다리(§9.2)의 단계 중 C에서 고른 단계를 쓴다. $\widehat q_1$의 함수군은 "$T_S$를 무시하고 $\widehat q_0$와 같은 출력"을 포함하도록 만든다($Z$ 경로를 공유하고 $T$ 경로를 더한다).

### 7.2 목적함수
- 정보 보존: $L_{\mathrm{keep}}(\phi,d)=\sum_{l=1}^{5}\mathbb E\|d_l(Z_S)-\chi_l(W_l)\|^2$. 원 픽셀 복원 $\mathbb E\|d(Z_S)-V_S\|^2$는 비교군으로 유지한다(§10.4).
- 균형 벌점: $\lambda\,(\widehat R_0-\widehat R_1-\delta^2)$, $\delta^2=0.02$(개발값).
- overlap 벌점: $\mu\cdot\mathbb E[\mathrm{softplus}(\widehat q_0-0.95)+\mathrm{softplus}(0.05-\widehat q_0)]$. 미분 가능하다. 범위 밖 비율은 검사 지표이지 학습 벌점이 아니다.
- B단계 손실: $L_{\mathrm{keep}}+\lambda(\widehat R_0-\widehat R_1-\delta^2)+\mu\cdot\text{overlap}$. predictor는 이 손실로 학습하지 않는다.

### 7.3 고리
1. (A) $\phi,d$ 고정. $\widehat q_0$, $\widehat q_1$을 각자의 Brier 위험으로 D에서 학습한다. C의 위험이 두 epoch 연속 개선되지 않으면 멈춘다. 더 큰 모형이나 다른 초기값이 C의 위험을 $0.002$ 넘게 낮추면 계속 학습한다. $\widehat q_1$의 C 위험이 세 번 연속 나빠지면 encoder 갱신을 보류하고 원인을 기록한다.
2. (B) $\widehat q_0,\widehat q_1$ 고정. $\phi,d$를 §7.2의 손실로 50 step 갱신한다. 학습률은 A단계의 1/10이다.
3. $\lambda$ 갱신: $\lambda\leftarrow\max\{0,\ \lambda+\rho_\lambda(\widehat R_0-\widehat R_1-\delta^2)\}$, 그리고 제약을 만족하면 $\lambda\leftarrow0.9\lambda$. 한 번 만족했다고 0으로 두지 않는다.
4. 멈춤: C에서 (i) $\widehat R_0-\widehat R_1\le\delta^2$, (ii) 범위 밖 비율 $\le1\%$, (iii) $L_{\mathrm{keep}}$이 세 바퀴 연속 $1\%$ 넘게 줄지 않음. 셋이 함께 성립하면 멈춘다. 최대 바퀴 수는 30이다.
5. 후보 선택: C에서 (i), (ii)를 만족하는 checkpoint 중 $L_{\mathrm{keep}}$이 가장 작은 것을 고른다. 선택 뒤 $\phi_S$를 고정한다.

### 7.4 기록
바퀴마다 $\widehat R_0$, $\widehat R_1$, $\widehat g$, $L_{\mathrm{keep}}$, $\lambda$, 범위 밖 비율, 사다리 단계를 D와 C 각각에 대해 기록한다.

## 8. encoder 두 수준

### 8.1 수준 A (기준선)
$Z_S=(X,\ B\,\psi_{-S}(W_{-S}))$, $B$는 rank 2 선형. 논문의 함수군이다. 학습은 논문과 같이 Brier 차이의 최소화다. §7의 교대학습 없이 바로 돈다.

### 8.2 수준 B (이 계획의 대상)
§7의 신경망 $\phi_S$. 수준 B가 수준 A보다 낫다는 것은 통과 조건이 아니다. 선형 encoder가 이미 충분할 수 있다. 신경망의 이점은 측정이 비선형인 별도 phantom(§10.3)에서 검사한다.

## 9. 3단계: 새 환자와 새 predictor로 검사한다 (집단 P, S)

### 9.1 절차
1. P에서 $\widehat q_0$, $\widehat q_1$을 새로 적합한다. $\phi_S$는 고정이다.
2. S에서 $\widehat g$를 계산한다. 표준오차 $\mathrm{SE}$를 계산한다. 다섯 분할에 대한 Bonferroni 정상근사 $z=\Phi^{-1}(1-0.05/5)=2.33$으로 $r=z\cdot\mathrm{SE}$를 둔다. $z$는 미리 고정한 다섯 검사에 대한 값이다. S에서 critic이나 checkpoint를 다시 고르면 그 선택을 $z$에 반영해야 한다. 기본 절차는 S에서 아무것도 다시 고르지 않는다.
3. $b_1$을 정한다. phantom에서는 오라클 predictor(§10.5)로 잰 $\epsilon_1^2$의 상위 95% 분위수를 $b_1$로 쓴다. 실제 자료에서는 $b_1$을 댈 근거가 없으므로 $b_1$ 없이 계산하고 "경험적 통과"로 보고한다. 사다리의 단계 간 $\widehat g$ 변화는 $b_1$의 대용으로 쓰지 않고 진단으로만 적는다.
4. 격자의 각 $\delta^2$에 대해 통과 여부 $\max(\widehat g,0)+r+b_1\le\delta^2$와 overlap(범위 밖 비율 $\le1\%$)을 계산한다.

### 9.2 predictor 사다리 (진단)
- 1단계: 선형 지수의 bounded probit. $T$쪽 입력은 reader 특징 $\psi_S(W_S)$.
- 2단계: 좌표별 spline(매듭 5개)의 additive probit. 입력은 1단계와 같다.
- 3단계: 원 픽셀 $T_S$를 읽는 CNN에 $Z_S$를 이어 붙인 모형.
- 세 단계는 자동으로 포함 관계가 아니다. 각 단계의 $R_0$, $R_1$, $\widehat g$를 따로 기록한다.
- 기준: 다음 단계에서 $\widehat g$가 $0.2\,\delta^2$ 넘게 뛰면(개발값에서 $0.004$) 그 분할을 "predictor 오차 미통제"로 표시한다. 뛰지 않아도 모든 단계가 같은 신호를 놓쳤을 가능성은 남는다.

## 10. phantom

### 10.1 생성기 (Gaussian 기본형)
- 영상 $64\times64$, 2D. 다섯 영역은 폐엽 모양의 고정 마스크.
- 중증도 $U\sim N(0,1)$. 영역 $k$의 병변 부담 $L_k=U+\sigma\varepsilon_k$, $\varepsilon_k\sim N(0,1)$ 독립. 병변은 부담에 비례하는 수와 밝기의 반투명 얼룩으로 그린다.
- 전역 밝기 $G\sim N(0,1)$. 모든 픽셀에 $0.3G$를 더한다.
- 장치 표식 $I\sim\mathrm{Bern}(0.3)$. 영역 5에만 선 모양으로 그린다.
- 관측 공변량 $X\sim N(0,1)$ 하나.
- 처치 $A=\mathbf 1\{1.5U+1.0I+0.2X+\epsilon_A>0\}$, $\epsilon_A\sim N(0,1)$.
- 결과 $Y=\tau A+1.0U+0.3X+\epsilon_Y$, $\tau=1$, $\epsilon_Y\sim N(0,1)$.
- $n=24{,}000$. $\sigma\in\{0.85,1.5,2,3\}$. 반복 20회.

### 10.2 참값 계산
생성기에서 $10^6$개를 따로 뽑아 오라클 predictor를 적합한다. 그것으로 참 $D^2$, 참 $e$, $p$, 그리고 우리 predictor의 $\epsilon_0^2,\epsilon_1^2$를 잰다. §3.6의 식은 참 모수 $\gamma=1.5$, $\beta=1$, $\lambda_k$, $\sigma_k$로 계산해 실측과 대조한다. 장치 표식의 불일치는 이 모형에서 약 $1.0^2\times\varphi(0)^2\times0.3\times0.7\approx0.033$이다.

### 10.3 비선형 변형
병변의 가시성이 포화하는 변형($L_k$ 대신 $\tanh(L_k)$로 그림)과, 결과가 $U$에 비선형인 변형($\beta U$ 대신 $U+0.5U^2$). 수준 B의 이점은 여기서 본다.

### 10.4 비교군
수준 A, 수준 B(요약 복원), 수준 B(원 픽셀 복원), 전체 픽셀 CNN 보정(균형 없음), $(X,W)$ 선형 보정, 역할을 준 proximal 2SLS(영역 두 묶음), 오라클 $(X,U)$, 단순 비교.

### 10.5 성공 기준 (Gaussian 기본형)
1. predictor 오차: 3단계 사다리의 $\epsilon_1^2$가 반복의 95%에서 $b_1$ 이하.
2. 거짓 통과: 영역 5를 떼어 둔 분할의 통과율이 $\delta^2\le0.02$에서 5% 이하. $\delta^2=0.04$에서의 통과율은 보고만 한다.
3. 재적합 안정성: P에서 새로 적합한 predictor의 $\widehat g$가 D의 마지막 $\widehat g$와 $0.002$ 안에서 일치.
4. 효과 추정: $|\widehat\tau-\tau|$가 §3.5의 한계 안에 있고, 수준 B의 MAE가 수준 A의 1.2배 이하.
5. 닫힌 식 대조: $\sigma$마다 실측 $D^2$가 §3.6의 식과 20% 안에서 일치. 편향도 같다.

## 11. 4단계: 추정, 집계, 보고 (집단 P, E)

- 통과한 분할마다 P에서 결과모형과 성향모형을 맞추고, 성향은 $[\eta,1-\eta]$, $\eta=0.05$로 자르며, E에서 AIPW 점수의 평균 $\widehat\theta_S$를 낸다.
- $|\widehat\theta_S-\widehat\theta_{S'}|\le2\rho$인 분할을 잇는다. 가장 큰 무리의 중앙값이 $\widehat\tau$다. 크기가 같으면 실패를 선언한다. $\rho$는 §3.6으로 예측한 분할 간 편향 차이의 최댓값에 $\widehat\theta_S$의 표준오차를 더한 값이다.
- 집계는 올바른 효과가 가장 큰 고유 무리를 이룰 때만 맞다. 여러 분할이 같은 방향으로 치우치면 집계가 그 치우침을 지우지 못한다. 그래서 §3.5의 한계를 함께 보고한다.
- 보고 항목: 격자의 $\delta^2$마다 통과 분할과 $\widehat\tau$, 각 분할의 $\widehat g$, $r$, $b_1$(있으면), 범위 밖 비율, 사다리 단계별 $R_0,R_1,\widehat g$, $L_{\mathrm{keep}}$, 편향 한계 $\Gamma\delta/\{\eta(1-\eta)\}$, 그리고 phantom에서는 참값 대조. 작업 모형의 $\beta$ 편향 곡선은 그 모형의 진단으로만 적는다.

## 12. 비용과 순서

1. phantom 생성기 (PRD 필요).
2. 1단계 코드: 영역 마스크, MAE, 고정 요약, reader.
3. 수준 A를 먼저 돌려 기준선을 얻는다.
4. 2단계와 3단계 코드. 분할 하나로 고리를 검증한 뒤 다섯으로 늘린다.
5. 10.5의 성공 기준을 통과하면 10.3의 비선형 변형.
6. 실제 CT로 가는 결정은 그 뒤다. 환자 수가 수천 명 아래이면 3단계의 $r$이 $\delta^2$보다 커질 수 있고, 그 경우 실제 자료는 진단 중심으로 보고한다.

계산 비용은 64×64 영상, $n=24{,}000$, 분할 다섯 개 기준으로 GPU에서 $\sigma$ 하나당 수 시간이다. Mac M4에서는 수준 A와 작은 반복은 가능하고, 수준 B 전체 격자는 DeltaAI를 쓰는 편이 낫다. 정확한 수치는 PRD에서 잰다.

## 13. 검토에서 거둔 것과 남긴 것

- 거둠: "정확한 predictor는 저차원을 강제한다", "인코딩 뒤 분할은 held-out 독립을 보장하지 못한다"(가정을 바꾼 증명), "SCM-4가 (R1)의 근거다", "reader 공분산의 바닥으로 $\delta^2$를 인증한다", "$z\cdot\mathrm{SE}$가 predictor 오차를 덮는다".
- 남김(범위를 좁혀): Brier 분해(§3.1), 떼어 둔 쪽 고정(§3.3), 식별의 승계(§3.4), 선형 Gaussian 범위의 바닥과 편향 식(§3.6, phantom 평가 전용), 충분한 reader가 완비성을 보존한다는 보조정리(reader를 $T$로 쓸 때만 해당, 현재 기본 설계에서는 $T$가 원 픽셀이라 쓰이지 않음).
- 증명 필요: §3.6의 근사 오차와 선형 encoder 중 최소성, 사다리의 sieve 일치성, 교대학습의 수렴(보장 없음, 경험적으로만 확인).

## 14. 이 계획이 주는 것과 주지 않는 것

- 주는 것: $T$는 학습이 고르지 않는다. predictor 정확성은 재학습, 새 환자, 사다리로 지킨다. 통과식이 표본 오차와 predictor 오차를 분리한다. 허용 예산은 격자 전체에 대해 보고한다. 편향은 한계와 함께 보고한다.
- 주지 않는 것: $L_{\mathrm{keep}}$은 정보 소실을 줄이는 보조 목표이고 중증도 보존을 보장하지 않는다. 완비성은 가정이다. 모든 영역에 공통인 처치 전용 신호는 어떤 분할도 걸러내지 못한다. 실제 자료에서는 $b_1$이 없어 인증 통과를 주장하지 않는다.
