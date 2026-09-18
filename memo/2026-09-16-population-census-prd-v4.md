# Positive-control audit PRD v4

Created: 2026-09-16 CDT
Supersedes: v3
Status: **제안. 검토 대기. 구현하지 않았고 seed를 쓰지 않았다.**

## 0. 통제 문장

하나의 역할 비인지적이고 정확히 풀리는 SCM을 population에서 사전 검증한 뒤 동결하고, fresh data에서 원고의 Algorithm 1을 문자 그대로 실행하여
실제 learned screened set에 대한 finite-sample theorem의 조건과 성공확률 하한을 검증한다.

이 실험의 이름은 **Algorithm 1과 정리의 positive-control audit**이다. 다음은 이 실험의 질문과 무관하므로 뺀다.
parameter grid, approximate arm, neural network와 이미지, root-recovery 기준, interval 병합, 치환 실험, robustness 확장, fitted nuisance 성능 비교.

## 1. SCM 하나

관측 자료는 $O=(X,W,A,Y)$뿐이다. $U=(U_{\mathrm c},U_{\mathrm t})$는 항상 미관측이며 시뮬레이터와 oracle evaluator만 쓴다.
학습기는 $U$, loading, 아래 어떤 계수도 받지 않는다.

$$X_1=U_{\mathrm c}+\kappa\eta_1,\quad X_2=\eta_2,\quad W_j=b_{jc}U_{\mathrm c}+b_{jt}U_{\mathrm t}+\sigma_j\varepsilon_j,$$
$$P(A=1\mid U,X)=0.1+0.8\,\Phi(\alpha_{\mathrm c}U_{\mathrm c}+\alpha_{\mathrm t}U_{\mathrm t}+0.2X_1+0.3X_2),\qquad
Y=A-c\,U_{\mathrm c}+0.2X_1-0.5X_2+s\,\varepsilon_Y,\qquad\tau=1.$$

$h=0$으로 고정한다. overlap은 구성상 $0.1\le e(z)\le0.9$이므로 $\eta=0.1$이다.

| 계수 | 값 |
|---|---|
| $J$ | 7 |
| $b_{\cdot c}$ | $(1,1,1,1,1,0,0)$ |
| $b_{\cdot t}$ | $(0,0,0,-0.6,\ 0.4,\ b_{\mathrm t}^{\circ},b_{\mathrm t}^{\circ})$ |
| $\sigma$ | $0.85\cdot\mathbf 1_7$ |
| $\kappa,\ \alpha_{\mathrm c},\ c,\ s$ | $2,\ 1,\ 2.5,\ 0.3$ |
| $\alpha_{\mathrm t}$ | 3절에서 푼다 (target family의 정확한 균형) |
| $b_{\mathrm t}^{\circ}$ | 3절에서 푼다 (wrong family의 정확한 균형) |

계수는 과도하게 크지 않게 둔다. 풀린 $\alpha_{\mathrm t}$가 $\alpha_{\mathrm c}$의 몇 배를 넘으면 $\sigma$나 $\kappa$를 조정하고 이유를 두 줄로 적는다. 격자는 없다.

## 2. 표현 class와 선택 규칙

$d=|S^{c}|$에 대해 $\mathcal B_d=\{v/\lVert v\rVert_2: v\in\{-1,0,1\}^d\setminus\{0\}\}/\{\beta\sim-\beta\}$,
$\Phi_S=\{(X_1,X_2,\beta^\top W_{-S}):\beta\in\mathcal B_d\}$. library는 치환과 부호 반전에 닫혀 있고 역할 정보가 없다.
선택은 library 위에서 균형 목적함수의 최소화이고, 동점은 집합으로 보존한다.

## 3. Population preflight: 126개 split을 한 번만 조사한다

### 3.1 정확한 균형의 판정

$C=\beta^\top W_{-S}$, $p=\beta^\top b_{\cdot c}$, $q=\beta^\top b_{\cdot t}$, $v=\sum_j\beta_j^2\sigma_j^2$, $G=\alpha_{\mathrm c}U_{\mathrm c}+\alpha_{\mathrm t}U_{\mathrm t}$,
$\Sigma=\mathrm{Var}(X_1,C)=\begin{pmatrix}1+\kappa^2&p\\p&p^2+q^2+v\end{pmatrix}$,
$$\gamma(\beta)=\bigl(\mathrm{Cov}(G,U_{\mathrm c}\mid X_1,C),\ \mathrm{Cov}(G,U_{\mathrm t}\mid X_1,C)\bigr).$$
결합 정규성과 probit의 단조성 때문에 $D_S^2(\beta)=0\iff B_S\gamma(\beta)=0$이 정확히 성립한다.
판정은 이 대수식으로 한다. 수치 적분을 쓰지 않는다. 기준은 $\lVert B_S\gamma(\beta)\rVert_\infty\le10^{-10}$이고, 설계된 균형점은 기계 정밀도 수준이어야 한다.

### 3.2 계수 풀이

- target family $S\subseteq\{W_1,W_2,W_3\}$: 지정 원소 $\beta^{*}=W_4+W_5$. $\gamma_{\mathrm c}(\beta^{*})=0$은 $\alpha_{\mathrm t}$에 선형이므로 닫힌 식으로 푼다. $(p,q,v)$가 held-out 순수 블록에 무관하므로 일곱 split이 한 번에 균형을 이룬다.
- wrong family $S=\{W_6\},\{W_7\}$: 지정 원소 $W_4+W_7$, $W_4+W_6$. $\gamma_{\mathrm t}=0$을 $b_{\mathrm t}^{\circ}$에 대해 bracket 근 찾기로 푼다. 이 계수는 $\beta^{*}$의 $(p,q,v)$에 들어가지 않는다. $\beta^{**}\neq\beta^{*}$인 이유는 같은 channel이면 $\theta$가 같아지기 때문이다.

### 3.3 ORC 라벨과 class

split $S$는 $e_{\mathrm c}=(1,0)\in\operatorname{rowspan}(B_S)$일 때 ORC-valid다. 평가 전용 라벨이며 학습기에 주지 않는다.

- population candidate: library 안에 정확한 균형점이 있는 split.
- target class $\mathcal G$: ORC-valid인 candidate. Theorem 1에 의해 $\theta_S=\tau$다. 적분이 필요 없다.
- wrong class: ORC-invalid인 candidate. $\theta_S$는 4절의 식을 Gauss-Hermite(차수 48, 96, 144 일치 확인)로 계산한다. 같은 값(차수 간 차이 이내)끼리 한 class다.

### 3.4 확인할 세 사실

1. **plurality**: $\Delta=p_\tau-p_\star>0$. $p_\tau=|\mathcal G|/N$, $p_\star$는 wrong class 최대 비율, $N$은 candidate 수.
2. **non-target 존재**: wrong class가 하나 이상이다. $L$은 서로 다른 wrong 값의 수.
3. **분리 가능한 표본 크기 존재**: $g_{\min}=\min|\theta-\theta'|$(서로 다른 candidate value 쌍 전부)에 대해
   $4\rho_{\mathrm{thm}}(n_{\mathrm E})<g_{\min}$인 $n_{\mathrm E}$가 존재한다. $\rho_{\mathrm{thm}}$은 4절의 식이다. 그 최소 $n_{\mathrm E}$를 적는다.

세 사실이 성립하면 **즉시 DGP를 동결**한다. 성립하지 않으면 조정한 계수와 이유를 두 줄로 적고 다시 preflight한다.

### 3.5 preflight가 남기는 것

`results/positive_control/preflight_v1.json`: 계수, 풀린 $\alpha_{\mathrm t}$와 $b_{\mathrm t}^{\circ}$, 126행(split, ORC, 균형점 원소들, class, $\theta$),
$N$, $|\mathcal G|$, $L$, $\Delta$, $g_{\min}$, 최소 $n_{\mathrm E}$, 그리고 모든 $(S,\beta)$ 쌍의 $\sigma_{S,\beta}$ 표.

$\sigma_{S,\beta}$는 정확한 nuisance 아래 AIPW score의 표준편차다. 표기한 고정 seed의 $10^6$ 표본 Monte Carlo로 계산하고 MC 오차를 적는다.
이 seed는 확증 seed 구간과 겹치지 않는다. 확증에서는 이 표를 조회만 한다.

## 4. Literal Algorithm 1 confirmation

정확한 nuisance $e(z)=P(A=1\mid Z=z)$, $m_a(z)=\mathbb E(Y\mid A=a,Z=z)$는 정규 조건부 사영과 probit 닫힌 식으로 준다.
반경은 원고 Section 4의 유한표본 식이다.
$$\rho=\max_{S\in\widehat{\mathcal B}}\sigma_{S,\widehat\beta_S}\sqrt{\frac{N}{n_{\mathrm E}\,\delta}},\qquad N=|\widehat{\mathcal B}|,\ \delta=0.05.$$
정확한 nuisance이므로 nuisance 오차항은 0이다.

fresh seed(미사용 구간에서 새로 정함)마다 원고 알고리즘을 그대로 돈다.

1. $\mathfrak S$에서 $m$개 split을 균일 비복원 추출. **주 설정은 $m=T=126$**이다. 후보가 아홉 개 안팎이면 $m=30$에서는 $P(R=0)\approx0.11$이라 하한이 무의미해진다.
2. $\mathcal I_{\mathrm D}$에서 library 위 표현 선택과 balance screen.
3. $\mathcal I_{\mathrm N}$에서 nuisance 준비. 정확한 nuisance를 쓰므로 이 단계는 함수 평가다.
4. $\mathcal I_{\mathrm E}$에서 honest AIPW $\widehat\theta_S$.
5. 거리 $2\rho$로 연결.
6. 가장 큰 connected component의 중앙값 반환. 동률이면 candidate set.

실행마다 실제 retained set $\widehat{\mathcal B}$로 다시 계산한다: $N$, $\Delta$(ORC 라벨로 $\widehat{\mathcal B}$ 안의 target 비율과 wrong 최대 비율), $g_{\min}$(population $\theta$ 값 사이), $\rho$.
그리고 원고의 세 조건만 판정한다.

| 조건 | 판정 |
|---|---|
| plurality | $\Delta>0$ |
| uniform estimation radius | 모든 $S\in\widehat{\mathcal B}$에서 $|\widehat\theta_S-\theta_S|\le\rho$ |
| separation | $g_{\min}>4\rho$ |

마지막으로 관측 성공률 $\widehat P(|\widehat\tau-\tau|\le\rho)$을 원고 하한
$$1-\delta-P(R=0)-L\,\mathbb E\bigl[e^{-R\Delta^2/2}\bigr]$$
과 비교한다. $m=T$이면 $R=N_{\mathrm{pop}}$이 확정이라 하한은 $1-\delta-L\,e^{-N_{\mathrm{pop}}\Delta_{\mathrm{pop}}^2/2}$다.

## 5. 동결 항목

확증 규약에 적는 것: preflight 파일 해시, 계수 전부, library 정의, screen 규칙과 문턱, $n_{\mathrm D},n_{\mathrm N},n_{\mathrm E}$ (3.4의 최소 $n_{\mathrm E}$ 이상), $m$, $\delta$, 반복 수, seed_start(미사용 구간), 소스 해시.
확증 결과가 실패해도 동결한 항목은 바꾸지 않는다.

## 6. 코드에 두는 검사 (실험이 아님)

- 블록 인덱스 치환 하나(고정 seed)에 대해 preflight 출력이 그대로 치환되는지 확인하는 단위 테스트. 역할 무지를 강제하는 장치다.
- 학습기와 선택 코드에 블록 역할 기호가 없음을 검색으로 확인하는 테스트.

## 7. 범위

이 실험은 Sections 2–4의 Algorithm 1과 finite-sample theorem이 역할을 모르는 유한 library 위에서 작동하는지를 보는 positive-control audit이다.
학습된 nuisance, 신경망 표현, 이미지 proxy는 다루지 않는다. $J=7$ 결과와 진단 그림은 잘못 지정된 $X/W$ 분할에 대한 negative-control stress test로 보존한다.
