# Population census PRD v1

Created: 2026-09-16 CDT
Status: **제안. 검토 전. seed를 쓰지 않는다.**

## 0. 철회

직전 보고의 "모집단으로 확정했다"는 주장을 철회한다. 네 가지가 성립하지 않았다.

1. 식과 설명이 모순됐다. $W_j=U+\sigma_j\varepsilon_j+a_jI$라고 써 놓고 일부 블록은 $U$ 정보가 없다고 설명했다. $U$ loading이 식에 없었다.
2. 126개 split 전수 census artifact가 없다. 조합 계산 $126=95+28+3$만 있었다.
3. 평평한 목적함수에서 $\beta$ tie-break가 정의되지 않았다. 보고한 $-0.23$, $-0.10$, $0.74$는 Algorithm 1의 candidate value가 아니라 임의 minimizer 셋이다.
4. separation을 $2\rho$와 비교했다. 기준은 $g_{\min}>4\rho$이고, $0.264 < 4\rho\approx0.34$이므로 통과하지 못한다.

여기에 더해, 정보가 전혀 없는 블록을 두면 그 split의 minimizer 집합이 구면 전체가 되어
$\{\theta_S(\beta):D_S^2(\beta)=0\}$가 $\tau$를 포함할 수 있다. 그러면 $g_{\min}=0$이다. 설계를 바꾼다.

## 1. 변수와 DGP

잠재변수는 벡터 하나다. 별도의 $I$ 표기를 쓰지 않는다.

$$U=(U_{\mathrm c},U_{\mathrm t})$$

- $U_{\mathrm c}$: $A$와 $Y$에 모두 영향을 주는 성분.
- $U_{\mathrm t}$: $A$에만 영향을 주고 $Y$에는 직접 영향을 주지 않는 성분.

독립 표준정규 $U_{\mathrm c},U_{\mathrm t},\eta_1,\eta_2,\varepsilon_1,\dots,\varepsilon_J,\varepsilon_Y$에 대해

$$X_1=U_{\mathrm c}+\kappa\eta_1,\qquad X_2=\eta_2$$
$$W_j=b_{jc}U_{\mathrm c}+b_{jt}U_{\mathrm t}+\sigma_j\varepsilon_j,\qquad j=1,\dots,J$$
$$P(A=1\mid U,X)=0.1+0.8\,\Phi\!\left(\alpha_{\mathrm c}U_{\mathrm c}+\alpha_{\mathrm t}U_{\mathrm t}+0.2X_1+0.3X_2\right)$$
$$Y=A\,(1+hU_{\mathrm c})-c\,U_{\mathrm c}+0.2X_1-0.5X_2+s\varepsilon_Y,\qquad \tau=\mathbb E\{Y(1)-Y(0)\}=1$$

관측 자료는 $O=(X,W,A,Y)$뿐이다. $U_{\mathrm c}$, $U_{\mathrm t}$, loading $b_{jc},b_{jt}$, $\sigma_j$는 시뮬레이터만 안다.
oracle 비교군만 $U$를 쓴다. $X_1$은 $U_{\mathrm c}$의 잡음 섞인 측정이므로 $U_{\mathrm c}$가 관측된 것이 아니다.

### 블록 유형 (학습기에 알리지 않는다)

| 유형 | loading | 역할 |
|---|---|---|
| 결과 관련 proxy | $b_{jc}\neq0$ | $U_{\mathrm c}$를 잰다. held-out이면 outcome-relevant completeness가 성립할 후보다 |
| 처치 전용 proxy | $b_{jc}=0$, $b_{jt}\neq0$ | $A$에 대해서는 정보가 있으나 $Y$ 평균을 가르지 못한다. completeness 실패 후보다 |
| 혼합 proxy | 둘 다 $\neq0$ | 균형을 위해 조합이 필요하다 |

정보가 전혀 없는 블록($b_{jc}=b_{jt}=0$)은 **넣지 않는다.** 목적함수를 평평하게 만들어 $\theta_S$를 정의 불가능하게 하기 때문이다.
"균형은 되지만 표적이 틀린" 사례는 처치 전용 proxy가 담당한다. 이것이 원고가 단순 completeness가 아니라
outcome-relevant completeness를 쓰는 이유와 정확히 맞는다. 이 주장은 가설이며 census가 검증한다.

## 2. 표현 class와 결정적 선택 규칙

$$\Phi_S=\left\{\phi_{S,\beta}(X,W_{-S})=\left(X_1,X_2,\beta^\top W_{-S}\right)\ :\ \beta\in\mathbb R^{|S^{c}|},\ \lVert\beta\rVert_2=1\right\}$$

class 안에 블록 역할이 없다. 선택은 자료에 의존하지 않는 tie-break로 한 점으로 확정한다.

1. 균형 목적함수 $D_S^2(\beta)$를 최소화한다.
2. 최소값에서 허용오차 $\epsilon_{\mathrm{tie}}$ 안에 있는 $\beta$ 가운데, 미리 고정한 기준 벡터 $\beta_{\mathrm{ref}}=\mathbf 1/\sqrt{|S^{c}|}$에 유클리드 거리가 가장 가까운 것을 고른다.
3. 부호는 $\beta^\top\beta_{\mathrm{ref}}\ge0$이 되도록 뒤집는다. 정확히 0이면 첫 비영 좌표를 양수로 만든다.

$\beta_{\mathrm{ref}}$는 블록 번호에 대칭이므로 역할 정보를 담지 않는다.

## 3. Census 산출물 (전수 $2^J-2$행)

각 split $S$마다 다음을 기록한 파일 하나를 만든다.

| 항목 | 정의 |
|---|---|
| `split`, `size` | held-out 블록 집합과 크기 |
| `category` | 평가 전용 라벨. 학습기와 선택에 쓰지 않는다 |
| `min_discrepancy` | $\min_\beta D_S^2(\beta)$ |
| `beta_selected` | 2절 규칙이 고른 $\beta_S$ |
| `balanceable` | $\min_\beta D_S^2(\beta)\le\epsilon_{\mathrm{bal}}$ |
| `theta` | $\theta_S=\tau_{Z_{\beta_S}}$ |
| `theta_range` | $\{\theta_S(\beta):D_S^2(\beta)\le\epsilon_{\mathrm{bal}}\}$의 최소와 최대 |
| `targets_tau` | $\lvert\theta_S-\tau\rvert\le\epsilon_\tau$ |

출력은 `results/population_census_<design>_v1.json`, 생성 코드는 `code/population_census.py`다.
계산은 모집단 값이므로 seed를 쓰지 않는다. Gauss-Hermite 차수 두 개로 같은 값이 나오는지 검증한다.

## 4. seed를 쓰기 전에 통과해야 하는 기준

| 코드 | 기준 | 판정 방법 |
|---|---|---|
| A1 | plurality $\Delta=p_\tau-p_\star>0$ | 균형 가능한 split의 $\theta_S$를 $\epsilon_\tau$로 군집화한다. $p_\tau$는 $\tau$ 군집의 비율, $p_\star$는 그 외 최대 군집의 비율 |
| A2 | separation $g_{\min}>4\rho_{\max}$ | $g_{\min}$은 균형 가능한 split의 서로 다른 $\theta$ 값 사이 최소 거리. $\rho_{\max}$는 계획한 $n$에서 나오는 최대 연결 반경으로, 개발 seed에서 측정한다 |
| A3 | 비퇴화 | 어떤 균형 가능한 split도 `theta_range`가 $\tau$를 포함하지 않는다. 단 $\tau$를 겨냥하는 split은 예외 |
| A4 | 역할 무지 | 블록 번호를 치환하면 결과가 그대로 치환되는지 검사한다. 학습기 코드에 블록 역할 기호가 없음을 검색으로 확인한다 |

A2의 $\rho$는 평가 표본 크기에 따라 줄어든다. 직전 $J=7$ 실행에서 $n=6{,}000$일 때 $\rho\approx0.085$, 즉 $4\rho\approx0.34$였다.
따라서 설계는 $g_{\min}>0.34$를 만들거나, $n$을 키워 $4\rho_{\max}$를 $g_{\min}$ 아래로 내려야 한다. 둘 중 무엇을 택할지는 census 결과가 정한다.

## 5. 파라미터 탐색

$b$, $\sigma$, $\kappa$, $\alpha$, $c$, $h$, $J$를 작은 격자에서 돌려 A1부터 A3까지를 만족하는 설계를 찾는다.
이 단계는 개발이며 seed를 쓰지 않는다. 선택한 설계와 탈락한 설계를 모두 기록한다.

## 6. 그다음

A1부터 A4까지 통과한 뒤에만 fresh-seed 확증을 봉인한다. 봉인 문서에는 $n$, 반복 수, 예산 $m$, 관문, 미사용 구간에서 뽑은 seed_start,
그리고 census 파일의 해시를 적는다.

## 7. 이번 PRD가 바꾸지 않는 것

$J=7$ 결과와 진단 그림은 그대로 둔다. 지위는 잘못 지정된 $X/W$ 분할에 대한 negative-control stress test이고 appendix failure analysis 자리다.
