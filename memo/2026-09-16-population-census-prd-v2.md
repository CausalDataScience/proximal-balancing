# Population census PRD v2

Created: 2026-09-16 CDT
Supersedes: `2026-09-16-population-census-prd-v1.md` (판정 FIX REQUIRED)
Status: **제안. 검토 대기. 구현하지 않았고 seed를 쓰지 않았다.**

v1에서 남은 다섯 문제와 열두 승인 조건을 아래 절에 대응시켰다. 절 제목 뒤 괄호가 조건 번호다.

## 1. 변수와 DGP (조건 2)

잠재변수는 벡터 하나 $U=(U_{\mathrm c},U_{\mathrm t})$다. 관측 자료는 $O=(X,W,A,Y)$뿐이다.
학습기는 $U$의 두 성분도, loading도, 아래 어떤 계수도 받지 않는다. oracle 비교군만 시뮬레이터의 $U$를 쓴다.

독립 표준정규 $U_{\mathrm c},U_{\mathrm t},\eta_1,\eta_2,\varepsilon_1,\dots,\varepsilon_J,\varepsilon_Y$에 대해

$$X_1=U_{\mathrm c}+\kappa\eta_1,\qquad X_2=\eta_2,$$
$$W_j=b_{jc}U_{\mathrm c}+b_{jt}U_{\mathrm t}+\sigma_j\varepsilon_j,\qquad j=1,\dots,J,$$
$$P(A=1\mid U,X)=0.1+0.8\,\Phi(L),\qquad L\triangleq\alpha_{\mathrm c}U_{\mathrm c}+\alpha_{\mathrm t}U_{\mathrm t}+0.2X_1+0.3X_2,$$
$$Y=A\,(1+hU_{\mathrm c})-c\,U_{\mathrm c}+0.2X_1-0.5X_2+s\,\varepsilon_Y,\qquad \tau=\mathbb E\{Y(1)-Y(0)\}=1.$$

### 기준 cell

| 계수 | 값 |
|---|---|
| $J$ | 7 |
| $b_{\cdot c}$ | $(1,1,1,1,1,0,0)$ |
| $b_{\cdot t}$ | $(0,0,0,-0.6,\,0.4,\,1,1)$ |
| $\sigma$ | $0.85\cdot\mathbf 1_7$ |
| $\kappa$ | 2 |
| $\alpha_{\mathrm c},\alpha_{\mathrm t}$ | $1,\ 2.5$ |
| $c,\ h,\ s$ | $2.5,\ 0,\ 0.3$ |

블록 유형은 평가 전용 라벨이고 학습기에 주지 않는다. $W_1,W_2,W_3$은 $U_{\mathrm c}$만 잰다. $W_4,W_5$는 둘 다 잰다. $W_6,W_7$은 $U_{\mathrm t}$만 잰다.
정보가 전혀 없는 블록($b_{jc}=b_{jt}=0$)은 넣지 않는다.

### 파라미터 격자 (조건 2)

기준 cell에서 네 손잡이를 각각 두 값으로 바꾼 $2^4=16$ cell을 모두 돈다.

| 손잡이 | 값 |
|---|---|
| $b_{5t}$ | $0.4,\ 0.6$ |
| $\alpha_{\mathrm t}$ | $1.5,\ 2.5$ |
| $h$ | $0,\ 0.3$ |
| $\sigma$ | $0.85\cdot\mathbf 1_7$, $(0.8,0.9,1.0,0.85,0.95,0.9,0.9)$ |

$s$는 모집단 $\theta$에 영향이 없으므로 격자에 넣지 않는다. 격자 밖의 cell을 시도하면 4절 ledger에 이유와 함께 적는다.

### 가설 (문제 1)

held-out 집합이 $U_{\mathrm t}$만 재는 블록으로만 이루어지면, $Z$가 $U_{\mathrm t}$ 관련 처치 정보를 흡수하여 균형은 이루되 $U_{\mathrm c}$의 잔여 교란이 남아 $\theta_S\neq\tau$가 될 수 있다.
이것은 **가설**이다. 현재 계수에서 성립하는지는 census가 판정한다.

## 2. 표현 class: 유한하고 치환에 닫힌 library (조건 4)

연속 구면은 쓰지 않는다. held-out 집합 $S$의 여집합 크기를 $d=|S^{c}|$라 하고

$$\mathcal B_d=\Bigl\{\tfrac{v}{\lVert v\rVert_2}:\ v\in\{-1,0,1\}^{d}\setminus\{0\}\Bigr\}\big/\{\beta\sim-\beta\},\qquad |\mathcal B_d|=\tfrac{3^{d}-1}{2},$$
$$\Phi_S=\bigl\{\phi_{S,\beta}(X,W_{-S})=(X_1,X_2,\beta^\top W_{-S}):\ \beta\in\mathcal B_d\bigr\}.$$

$\beta\to-\beta$를 동일시하는 근거: $\beta^\top W_{-S}$와 $-\beta^\top W_{-S}$는 같은 $\sigma$-대수를 생성하므로 아래 $D_S^2$와 $\theta_S$가 같다.
library는 좌표 치환과 부호 반전에 닫혀 있고 역할 정보가 없다.

$J=7$에서 전수 크기는 $\sum_{k=1}^{6}\binom{7}{k}\,\frac{3^{7-k}-1}{2}=7{,}035$개의 $(S,\beta)$ 쌍이다. 쌍마다 $D_S^2$와 $\theta_S$를 한 번씩 계산한다.

Fallback: 어떤 cell에서도 ORC-valid split의 $\min D_S^2$가 3절의 gap 검사를 통과하지 못하면 alphabet을 $\{-2,-1,0,1,2\}$로 넓힌 $\mathcal B_d'$를 같은 규칙으로 쓴다. 어느 alphabet을 썼는지 결과 파일에 적는다.

이 library는 census뿐 아니라 이후 유한표본 Algorithm 1의 실제 후보 class로도 쓴다. 그래야 census가 알고리즘의 후보 집합을 정확히 서술한다.

## 3. $D_S^2$와 $\theta_S$의 정의와 계산 (조건 1, 3)

$Z\triangleq Z_{S,\beta}=(X_1,X_2,\beta^\top W_{-S})$로 쓴다. $Z$가 $X$ 전체를 좌표로 갖고 있으므로 조건부 집합 $(X,W_S,Z)$는 $(W_S,Z)$와 같은 $\sigma$-대수다.

$$D_S^2(\beta)\triangleq\mathbb E\Bigl[\bigl\{P(A=1\mid W_S,Z)-P(A=1\mid Z)\bigr\}^2\Bigr],$$
$$\theta_S(\beta)\triangleq\mathbb E\bigl[\mathbb E(Y\mid A=1,Z)-\mathbb E(Y\mid A=0,Z)\bigr].$$

계산은 다음 세 사실로 닫힌 형태가 된다.

1. 모든 변수가 독립 표준정규의 고정 선형결합이므로, 임의의 선형 조건부 집합 $G$에 대해 $L\mid G$는 정규분포이고 평균 $m_G$와 분산 $s_G^2$는 정규직교 기저를 통한 사영으로 정확히 구한다.
2. probit link이므로 $P(A=1\mid G)=0.1+0.8\,\Phi\!\bigl(m_G/\sqrt{1+s_G^2}\bigr)$가 정확히 성립한다.
3. $\mathbb E(Y\mid A=a,Z)$는 Stein 항등식으로 $\mathbb E(U_{\mathrm c}\mid A=a,Z)$의 닫힌 식이 된다. $(U_{\mathrm c},L)\mid Z$가 이변량 정규이므로
   $\mathbb E[U_{\mathrm c}\,P(A=1\mid U,X)\mid Z]=0.1\mu_{\mathrm c}+0.8\{\mu_{\mathrm c}\Phi(q)+\mathrm{Cov}(U_{\mathrm c},L\mid Z)\,\varphi(q)/\sqrt{1+s^2}\}$, $q=\mu_L/\sqrt{1+s^2}$이고, $A=0$ 쪽은 $\mathbb E(U_{\mathrm c}\mid Z)=p\,\mathbb E(U_{\mathrm c}\mid A=1,Z)+(1-p)\,\mathbb E(U_{\mathrm c}\mid A=0,Z)$로 얻는다.

남는 것은 두 정규 선형형에 대한 바깥 기대값이고, 2차원 Gauss-Hermite로 계산한다.

이것은 원고의 조건부 확률 discrepancy 그 자체다. 유한표본 알고리즘이 쓰는 Brier critic은 이 양의 경험적 대용물이며, 그 관계는 확증 규약에서 따로 적는다.

### 수치 인증 (조건 3, 11)

- Gauss-Hermite 차수 48, 96, 144 세 개로 각 값을 계산한다. 값은 차수 144의 것을 쓰고, 잔차 $r=\max(|v_{96}-v_{48}|,\ |v_{144}-v_{96}|)$를 기록한다.
- 인증 구간은 $[v_{144}-2r,\ v_{144}+2r]$로 둔다. 이것은 적분 정확도의 잔차이지 최적화 인증이 아니다. 최적화 인증은 2절의 유한 전수로 대신한다.
- $\epsilon_{\mathrm{bal}}=10^{-5}$를 잠정값으로 둔다. 확정 규칙: 한 cell에서 $\min_\beta D_S^2$의 분포를 정렬했을 때 $\epsilon_{\mathrm{bal}}$ 바로 아래 값과 바로 위 값의 비가 10 이상이어야 한다(gap 검사). 통과하지 못하면 그 cell은 ledger에 `no_balance_gap`으로 적는다.
- 동점 허용치 $\epsilon_{\mathrm{tie}}=10^{-9}$ (절대값, $D_S^2$ 기준).
- 균형 표적 인증: ORC-valid split에서 library 최소점의 $D_S$가 0이 아닐 수 있으므로 $\theta_S=\tau$는 항등식으로 인증할 수 없다.
  대신 Theorem 3의 상계 $|\theta_S-\tau|\le\Gamma\,D_S/\{\eta(1-\eta)\}$를 쓴다. $\Gamma$는 이 선형-정규 설계에서 닫힌 식으로 유도해 적는다. 유도가 안 되면 $|\theta_S-\tau|$와 인증 구간만 보고하고 표적 판정은 검토자에게 맡긴다.
  $D_S^2\le r$(적분 잔차 이하)이면 `exact_balance_hit`로 따로 표시한다.

## 4. 동점 처리 (조건 5, 문제 4)

임의의 tie-break를 만들지 않는다.

1. split마다 $\min_{\beta\in\mathcal B_d}D_S^2(\beta)$를 구하고, 그 값에서 $\epsilon_{\mathrm{tie}}$ 안에 있는 모든 $\beta$를 **동점 집합** $\mathcal T_S$로 보존한다.
2. $\mathcal T_S$의 모든 원소가 같은 인증 구간의 $\theta$를 주면 그 값이 split $S$의 candidate value다.
3. 서로 다른 $\theta$를 주면 그 cell을 census 단계에서 **FAIL**로 판정하고 ledger에 `tie_set_ambiguous`로 적는다.

좌표 순서에 의존하는 규칙은 없다. 이 규칙은 블록 치환에 정확히 동변이다.

## 5. Census 산출물 (조건 11)

파일: `results/population_census/<cell_id>.json`. 생성 코드: `code/population_census.py`. 검증: `code/test_population_census.py`.

```
{
  "complete": true,
  "prd": "memo/2026-09-16-population-census-prd-v2.md", "prd_sha256": "...",
  "source_sha256": {"population_census.py": "...", "test_population_census.py": "..."},
  "cell_id": "...", "coefficients": {...}, "alphabet": [-1,0,1],
  "quadrature_orders": [48,96,144], "eps_bal": 1e-5, "eps_tie": 1e-9,
  "rows": [ {
      "split": [..], "size": k, "category": "...",      # category는 평가 전용
      "orc_valid_analytic": true,                        # 6절의 해석적 라벨
      "min_discrepancy": ..., "discrepancy_residual": ...,
      "tie_set": [[beta], ...], "tie_set_size": ...,
      "balanceable": ..., "exact_balance_hit": ...,
      "theta": ..., "theta_interval": [lo, hi], "theta_residual": ...,
      "theta_bound_thm3": ...,                           # 3절, 유도되면
      "tie_set_theta_range": [lo, hi]
  }, ... ],
  "candidate_values": [ {"value_interval": [lo,hi], "splits": [...], "count": n, "merged_from": [...]} ],
  "plurality": {"p_top": ..., "p_second": ..., "delta": ..., "top_is_tau_certified": ...},
  "separation": {"g_min": ..., "pairs_checked": n, "closest_pair": [...]},
  "balance_gap_check": {"below": ..., "above": ..., "ratio": ..., "pass": ...},
  "permutation_test": {"permutation": [...], "max_abs_diff": ..., "pass": ...},
  "verdict": "PASS" | "FAIL", "fail_reasons": [...]
}
```

`complete`가 `true`이려면 $2^J-2$개 split 모두가 `rows`에 있어야 한다.

## 6. 해석적 ORC 라벨 (조건 7의 전제)

split $S$를 **ORC-valid**로 표시하는 조건은 $\max_{j\in S}|b_{jc}|>0$이다.
근거: held-out 블록이 $U_{\mathrm c}$에 0이 아닌 선형 loading과 가법 정규 잡음을 가지면 $u_{\mathrm c}\mapsto\mathbb E[W_S\mid u]$가 $u_{\mathrm c}$에 단사이고, 이것은 Proposition 1의 (a) 경우다.
반대로 $S$의 모든 블록이 $b_{jc}=0$이면 $W_S\mid(Z,U)$의 법칙이 $U_{\mathrm c}$에 의존하지 않아 outcome-relevant completeness가 실패한다.
이 라벨은 math-theory-agent의 확인이 필요하다. census는 그 귀결(ORC-valid split의 candidate가 $\tau$ 근방인지)을 독립적으로 검사한다.

## 7. 후보값 병합과 plurality (조건 6, 7)

tolerance clustering은 쓰지 않는다.

1. 각 split의 candidate value는 3절의 인증 구간이다.
2. 두 구간이 겹치면 같은 값으로 병합할 수 있다. 병합은 **$p_\star$를 가장 크게 만드는 방향**으로만 한다. 즉 최빈값 후보의 구간과 겹치는 것은 최빈값에 넣지 않고 경쟁 후보 쪽에 넣으며, 경쟁 후보끼리 겹치면 하나로 합친다.
3. `overlap` 항목에 겹친 쌍을 모두 적는다.
4. $p_\tau$는 최빈 병합값의 비율, $p_\star$는 그다음 비율이다. 분모는 균형 가능한 split 수다.

## 8. $\rho$ 규칙 (조건 8)

직전 $J=7$의 $\rho\approx0.085$는 계획용 참고치이고 acceptance quantity가 아니다. 두 arm을 분리한다.

**Theorem arm.** 정확한 nuisance $e(z)=P(A=1\mid Z=z)$, $m_a(z)=\mathbb E(Y\mid A=a,Z=z)$를 3절의 닫힌 식으로 쓴다.
균형 가능한 split 집합 $\mathcal R$에 대해 AIPW score의 모집단 공분산 $\Sigma$를 정확한 nuisance로 계산하고
(닫힌 식이 없으면 표기한 seed 구간의 대규모 Monte Carlo로 추정하며 MC 오차를 기록한다),
평가 표본 $n_{\mathrm E}$에서
$$\rho_{\mathrm{thm}}(n_{\mathrm E},\delta)\triangleq q_{1-\delta}\Bigl(\max_{S\in\mathcal R}|G_S|\Bigr),\qquad G\sim N(0,\Sigma/n_{\mathrm E}),\ \delta=0.05$$
로 둔다. 이것이 Section 4 조건에 들어가는 반경이다.

**Operational arm.** 개발 seed 구간(확증에 절대 재사용하지 않는 구간, 예: `seed_start=4100000`)에서 코드의 bootstrap 반경 규칙을 그대로 돌려 경험적 $\rho$의 분포를 기록한다. 이것은 운영 참고치이며 theorem certificate라고 부르지 않는다.

확증의 $n_{\mathrm E}$는 $4\rho_{\mathrm{thm}}(n_{\mathrm E},0.05)\le g_{\min}/1.25$가 되도록 정한다.

## 9. 승인 기준 (조건 7, 9)

| 코드 | 기준 |
|---|---|
| A1 | $\Delta=p_\tau-p_\star>0$ (7절의 보수적 병합 기준) |
| A1' | 최빈 병합값의 구간이 3절의 Theorem 3 상계 안에서 $\tau$를 포함한다. 유도가 안 되면 $|\theta-\tau|$를 보고하고 판정 보류 |
| A2 | 서로 다른 모든 병합값 쌍이 $|\theta-\theta'|>4\rho_{\mathrm{thm}}$을 만족한다. 표적 대 비표적뿐 아니라 **비표적 대 비표적**도 포함 |
| A3 | ORC-valid split의 동점 집합이 모두 같은 $\theta$를 주고, ORC-invalid split의 동점 집합도 split 안에서 같은 $\theta$를 준다. 어느 쪽이든 다르면 FAIL |
| A4 | 12절의 치환 검사 통과, 학습기와 census 코드에 블록 역할 기호가 없음을 검색으로 확인 |
| A5 | 3절의 balance gap 검사 통과 |

## 10. Ledger (조건 10)

`results/population_census/LEDGER.md`에 시도한 모든 cell을 적는다.

| 열 | 내용 |
|---|---|
| cell_id, 계수 | 1절 격자 좌표 |
| alphabet | ternary 또는 five-level |
| 균형 가능 수 / ORC-valid 균형 가능 수 / ORC-invalid 균형 가능 수 | 개수 |
| $p_\tau$, $p_\star$, $\Delta$ | 7절 |
| $g_{\min}$, 가장 가까운 쌍 | 9절 |
| 판정과 탈락 이유 | `tie_set_ambiguous`, `no_balance_gap`, `plurality_fail`, `separation_fail`, `orc_consequence_fail`, `permutation_fail` 중 해당 항목 |

탈락한 cell도 지우지 않는다.

## 11. 치환 검사 (조건 12)

블록 인덱스의 무작위 치환 $\pi$(고정 seed 한 개, 기록)를 시뮬레이터의 loading과 $\sigma$에 적용한 뒤 census 전체를 다시 돈다.
치환 전 결과의 split $S$와 치환 후 결과의 split $\pi(S)$에서 `min_discrepancy`, `tie_set`(치환 후 정렬), `theta`가 인증 잔차 이내로 같아야 한다.
이후 유한표본 알고리즘에도 같은 검사를 적용한다(전체 파이프라인).

## 12. 승인 뒤 순서

1. `code/population_census.py`와 test 구현. 구현 안에 블록 역할 기호가 없어야 한다.
2. 16 cell 전수 census와 ledger 작성.
3. 9절 기준을 모두 통과한 cell 목록과 각 cell의 $g_{\min}$, $\rho_{\mathrm{thm}}$ 곡선 보고.
4. 검토 뒤에만 fresh-seed 확증 규약 봉인. 봉인 문서에 census 파일 해시와 cell_id를 적는다.

## 13. 바꾸지 않는 것

$J=7$ 결과와 진단 그림은 잘못 지정된 $X/W$ 분할에 대한 negative-control stress test로 보존한다.
