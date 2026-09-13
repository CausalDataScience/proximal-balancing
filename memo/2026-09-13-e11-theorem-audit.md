# E11: Algorithm 1을 정리와 대조한 감사

작성일: 2026-09-13
판정: **PASS**, 열 조건 모두 충족
결과 파일: `results/probe_e11_theorem_audit_v1.json`
구현: `code/e11_theorem_audit.py`, `code/run_e11.py`, `code/e8_population.py`
대상 정리: `manuscript/main/4.tex`의 `thm:probe-search-error`와 `alg:probe-search`

---

## 0. 이 실험이 하는 일과 하지 않는 일

**하는 일.** 원고에 적힌 Algorithm 1을 그대로 구현하고, 정리가 요구하는 세 가정이 성립하는지 측정한
뒤, 정리가 약속한 확률 하한과 실제 성공률을 비교한다.

**하지 않는 일.** 유한 시뮬레이션은 정리의 보편적 참을 증명하지 못한다. 정리 자체는 증명이 담당한다.
이 실험이 검증하는 것은 구현이 페이지 위의 알고리즘과 같은지, 가정이 사전등록된 positive control에서
실제로 성립하는지, 그리고 관측된 성공률이 명시된 하한을 지키는지다.

성공사건은 정리가 정의한 그대로다.

$$\mathcal E=\left\{\text{단일값을 반환하고 }\lvert\widehat\tau-\tau\rvert\le\rho\right\}$$

후보집합 반환, 동점, 무반환은 전부 실패로 센다.

---

## 1. 결과

$T=30$, $m=20$, $\delta=0.05$, 역할당 $n=20{,}000$, outer state 5개, 각 5,000회 반복이다.

| seed | $N$ | $n(\tau)$ | $L$ | $\Delta$ | $\rho$ | $g$ | 정리 하한 $B$ | 관측 성공률 | 그 하한 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7720000 | 8 | 7 | 1 | 0.7500 | 0.2613 | 1.4432 | 0.7145 | 0.9988 | 0.9976 |
| 7720100 | 8 | 7 | 1 | 0.7500 | 0.2591 | 1.4432 | 0.7145 | 0.9970 | 0.9954 |
| 7720200 | 8 | 7 | 1 | 0.7500 | 0.2588 | 1.4432 | 0.7145 | 0.9980 | 0.9966 |
| 7720300 | 8 | 7 | 1 | 0.7500 | 0.2594 | 1.4432 | 0.7145 | 0.9988 | 0.9976 |
| 7720400 | 8 | 7 | 1 | 0.7500 | 0.2667 | 1.4432 | 0.7145 | 0.9976 | 0.9961 |

다섯 state 모두에서 **관측 성공률의 단측 하한이 정리 하한을 크게 웃돈다.** 균일반지름 사건은
$5{,}000$회 전부 성립해 하한이 $0.9993$이므로 요구치 $0.95$를 넘는다.

---

## 2. 열 가지 통과 조건

| 번호 | 조건 | 결과 |
| ---: | --- | --- |
| 1 | 다섯 state 모두 $\Delta>0$ | 통과, 전부 $0.75$ |
| 2 | 다섯 state 모두 $g>4\rho$ | 통과, $1.4432>1.067$ |
| 3 | 다섯 state 모두 $B\ge0.50$ | 통과, 전부 $0.7145$ |
| 4 | 균일반지름 사건의 동시 단측 하한 $\ge0.95$ | 통과, $0.9993$ |
| 5 | 성공률 하한 $\ge$ 각 state의 $B$ | 통과, $0.995$ 이상 대 $0.7145$ |
| 6 | $R$의 생성이 초기하 법칙과 일치 | 통과, $\chi^2$가 $2.4$에서 $6.8$, 자유도 $6$ |
| 7 | 후보집합, 동점, 무반환이 실패로 기록 | 통과, 동점 6에서 15회가 실패로 계산됨 |
| 8 | 학습표본과 평가표본의 행 중복 | 통과, 역할마다 독립 추출이고 해시가 다름 |
| 9 | source, protocol, seed, 결과의 해시 검증 | 통과 |
| 10 | 확인 실행 후 어떤 값도 재조정하지 않음 | 통과 |

$R$의 평균이 $5.32$에서 $5.36$인데 초기하 기대값 $Nm/T=5.333$과 일치한다.

---

## 3. 정리의 각 부분이 어디서 왔는가

### 3.1 $\theta_S$는 정확하다

$\theta_S$는 학습된 표현으로 조정했을 때 도달하는 모집단 값이다. 후보 사상이 전부 아홉 개 독립
표준정규의 선형결합이고 처치 링크가 probit이므로

$$\mathbb E[\Phi(V)\mid W]=\Phi\!\left(\frac{m}{\sqrt{1+s^2}}\right)$$

가 정확히 성립하고, Stein 항등식으로 $\theta_S$가 닫힌 형태로 나온다. 추정이 아니다.

**$\theta_S=\tau$의 인증은 수치 clustering이 아니라 설계의 analytic identity로 한다.** 오염 블록
$W_4$를 held-out으로 빼지 않고 균형 뿌리를 고른 split은 잔여 discrepancy가 $0$이므로 근사편향
정리에 의해 $\theta_S=\tau$가 정확히 성립한다. 고정밀 수치값은 그 identity를 **검사하는 데만**
쓰며, 어긋나면 오류로 멈춘다. 허용오차를 넓혀 흡수하지 않는다.

### 3.2 $\rho$는 원고의 공식 그대로다

$$\rho=\max_{S\in\widehat{\mathcal B}}\left[\frac{\sigma_S}{\sqrt{n_{\mathrm E}\,\delta/N}}
+\frac{q_{e,S}(q_{0,S}+q_{1,S})}{\eta}\right]$$

이는 원고 식 `eq:aipw-error-radius`에서 $\delta_{\mathrm E}$를 $\delta/N$으로 바꾼 것이고, 조건 (b)에
대한 정리의 설명이 지시하는 바 그대로다. $\sigma_S$와 nuisance 오차는 SCM의 **정확한** nuisance를
기준으로 독립 evaluator 추출 $200{,}000$행에서 측정한다. **평가표본은 $\rho$를 정하는 데 쓰지
않는다.**

최대값을 내는 split은 다섯 state 모두 오염 singleton $(4,)$이고, 그 성분은 분산항 $0.2519$와
편향항 $0.0094$다. 분산항이 지배하는데, 원고의 반지름이 가우스 분위수가 아니라
$1/\sqrt{n\delta}$ 형태의 보수적 경계를 쓰기 때문이다.

### 3.3 $B$는 정확한 유한합이다

$$B=1-\delta-P(R=0)-L\,\mathbb E\!\left[e^{-R\Delta^2/2}\right],
\qquad R\sim\operatorname{Hypergeometric}(T,N,m)$$

$N=8$, $L=1$, $\Delta=0.75$, $m=20$, $\delta=0.05$에서 $P(R=0)=7.69\times10^{-6}$,
$\mathbb E[e^{-R\Delta^2/2}]=0.2355$, 따라서 $B=0.7145$다. 하한이 자명하지 않다.

---

## 4. 표현 학습을 제한한 이유

정리는 여러 target split이 **정확히 같은** $\tau$를 가질 것을 요구한다. 연속적인 학습 계수는
일반적으로 $\theta_S\approx\tau$만 만들므로 literal theorem에 쓸 수 없다.

그래서 split마다 유한한 encoder class를 실행 전에 고정했다. 각 class는 그 split의 균형 계수 중
production 탐색 구간 $[-1,1]$ 안에 있는 것과, 모든 split에 공통인 미끼 계수 $(-0.95,\,+0.95)$로
이루어진다. **runtime은 $\tau$도 유효성 표지도 보지 않고 $\mathcal I_{\mathrm D}$에서 같은 경험적
Brier 목적만 최소화한다.**

미끼 위치는 파일럿에서 정했고 그 이유를 적어 둔다. 처음에는 미끼를 $(-0.8,-0.4,0,0.4,0.8)$로 두었는데
$+0.4$가 균형 뿌리 $+0.454$와 너무 가까워 유한표본 목적이 둘을 구분하지 못했다. 그때 선택된 사상의
$\theta_S$가 $0.904$가 되어 분리조건이 $0.096$으로 무너졌다. **정리와 무관한 이유로 실패한 것이므로**
미끼를 모든 뿌리에서 멀리 떨어뜨렸다. 파일럿 seed는 확인 seed와 겹치지 않는다.

표본 크기도 같은 방식으로 정했다. $n=2{,}000$에서는 $\rho=0.86$이라 $4\rho$가 $g=1.443$을 넘어
조건 (c)가 성립하지 않았다. $n=20{,}000$에서 $\rho\approx0.26$이 되어 $4\rho=1.07<1.443$이 된다.

이것은 표현학습 성능 실험이 아니다. Algorithm 1의 sampling, screen, estimation, aggregation 네 단계를
점검하는 positive control이다.

---

## 5. 자료 분할

각 outer replicate에서 $\mathcal I_{\mathrm D}$, $\mathcal I_{\mathrm N}$, $\mathcal I_{\mathrm E}$를
서로 독립으로 만든다. 역할은 원고 그대로다. $\mathcal I_{\mathrm D}$는 표현 최소화와 screen,
$\mathcal I_{\mathrm N}$은 nuisance 적합, $\mathcal I_{\mathrm E}$는 retained split 전부의 honest
AIPW 평가다.

별도 screen fold, rotate4, 회전 평균, 평가표본 bootstrap으로 $\rho$ 선택, split마다 다른 평가표본은
전부 쓰지 않았다. **모든 후보가 같은 $\mathcal I_{\mathrm E}$를 쓴다.**

nuisance 함수는 $\mathcal I_{\mathrm N}$에서 한 번 적합해 고정한다. inner 반복에서 다시 적합하면
정리가 조건으로 삼는 대상이 바뀌기 때문이다.

---

## 6. 원고에 넣을 문장

> 원고의 Algorithm 1을 one-shot 형태로 문자 그대로 구현했고, finite-sample theorem의 조건과 조건부
> 확률 하한을 사전등록한 positive-control SCM에서 경험적으로 검증했다.

---

## 7. 남는 경계

- 이것은 SCM-1 하나에서의 positive control이다. 다른 SCM, 신경망 encoder, 이미지 proxy로 넘어가면
  같은 감사를 다시 해야 한다.
- 유한한 encoder class를 썼다. 연속 학습 계수에서는 target split들이 $\tau$를 근사만 하므로,
  그 경우 정리는 $\rho$ 대신 $\rho+b$로 읽어야 한다. 원고 본문이 이미 그 확장을 적어 두었다.
- $\rho$의 $\sigma_S$와 nuisance 오차는 $200{,}000$행 evaluator 추출에서 측정한 Monte Carlo 값이고,
  그 자체의 오차를 결과 파일에 함께 저장했다.
