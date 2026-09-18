# Population census PRD v3

Created: 2026-09-16 CDT
Supersedes: v2 (판정 FIX REQUIRED)
Status: **제안. 검토 대기. 구현하지 않았고 seed를 쓰지 않았다.**

v2에 대한 여덟 가지 최소 수정을 어디에 반영했는지 먼저 적는다.

| 수정 | 반영 절 |
|---|---|
| 1 ORC 라벨을 row-space 조건으로 교체 | 2절 |
| 2 exact arm과 approximate arm 중 하나 선택 | 5절. **exact arm을 주 arm으로 택한다.** |
| 3 approximate arm을 남긴다면 $R=\rho+b$로 수정 | 10절. 운영 부록으로만 남기고 $R=\rho+b$ 규칙을 적는다 |
| 4 $p_\tau$를 ORC로 인증된 target class로 정의 | 7절 |
| 5 Gaussian 반경을 원고 Section 4 유한표본 반경으로 교체 | 8절 |
| 6 "certified interval" 표현 수정 | 4절. `numerical_stability_interval`로 낮추고 판정은 항등식과 여유로 한다 |
| 7 split 내부 selection margin $m_S$ gate 추가 | 6절 |
| 8 alphabet fallback 제거 | 3절. ternary 하나로 고정한다 |

## 1. 변수와 DGP

잠재변수는 벡터 하나 $U=(U_{\mathrm c},U_{\mathrm t})$다. 관측 자료는 $O=(X,W,A,Y)$뿐이다.
학습기는 $U$의 두 성분도, loading도, 아래 어떤 계수도 받지 않는다. oracle 비교군만 시뮬레이터의 $U$를 쓴다.

독립 표준정규 $U_{\mathrm c},U_{\mathrm t},\eta_1,\eta_2,\varepsilon_1,\dots,\varepsilon_J,\varepsilon_Y$에 대해

$$X_1=U_{\mathrm c}+\kappa\eta_1,\qquad X_2=\eta_2,$$
$$W_j=b_{jc}U_{\mathrm c}+b_{jt}U_{\mathrm t}+\sigma_j\varepsilon_j,\qquad j=1,\dots,J,$$
$$P(A=1\mid U,X)=0.1+0.8\,\Phi(L),\qquad L\triangleq\alpha_{\mathrm c}U_{\mathrm c}+\alpha_{\mathrm t}U_{\mathrm t}+0.2X_1+0.3X_2,$$
$$Y=A\,(1+hU_{\mathrm c})-c\,U_{\mathrm c}+0.2X_1-0.5X_2+s\,\varepsilon_Y,\qquad \tau=1.$$

블록 $j$의 loading 행을 $\mathbf b_j\triangleq(b_{jc},b_{jt})$로 쓴다. held-out 집합 $S$의 loading 행렬은 $B_S$ ($|S|\times2$)다.
결과는 $U_{\mathrm c}$에만 의존하므로 결과 관련 방향은 $e_{\mathrm c}\triangleq(1,0)$이다.

## 2. ORC 라벨 (수정 1)

split $S$는 다음일 때 **ORC-valid**다.

$$e_{\mathrm c}\in\operatorname{rowspan}(B_S).$$

| held-out 구성 | 판정 | 이유 |
|---|---|---|
| 순수 $U_{\mathrm c}$ 블록 $(b_c,0)$ 하나 이상 | valid | $e_{\mathrm c}$가 행 자체다 |
| 서로 비례하지 않는 mixed 블록 둘 | valid | 두 행이 $\mathbb R^2$를 생성한다 |
| 단일 mixed 블록 $(b_c,b_t)$, 둘 다 $\neq0$ | **invalid** | $U_{\mathrm c}$가 혼합값으로만 관측되어 $e_{\mathrm c}$가 행공간 밖이다 |
| $U_{\mathrm t}$만 재는 블록들 | invalid | 행공간이 $(0,1)$ 방향뿐이다 |

$\operatorname{rank}(B_S)=2$는 충분조건이지만 순수 $U_{\mathrm c}$ 블록 하나를 배제하므로 쓰지 않는다.
이 라벨은 해석적 판정이고 학습기에 주지 않는다. census는 그 귀결(5절)을 독립적으로 검사한다.

## 3. 표현 class: ternary library 하나로 고정 (수정 8)

$d=|S^{c}|$에 대해

$$\mathcal B_d=\Bigl\{\tfrac{v}{\lVert v\rVert_2}:\ v\in\{-1,0,1\}^{d}\setminus\{0\}\Bigr\}\big/\{\beta\sim-\beta\},\qquad
\Phi_S=\{(X_1,X_2,\beta^\top W_{-S}):\ \beta\in\mathcal B_d\}.$$

$\beta\sim-\beta$ 동일시의 근거는 같은 $\sigma$-대수다. library는 치환과 부호 반전에 닫혀 있고 역할 정보가 없다.
alphabet은 이것 하나다. 넓은 alphabet은 쓰지 않으며, 정확한 균형점은 alphabet이 아니라 5절의 DGP 계수 설계로 library 안에 넣는다.
$J=7$에서 $(S,\beta)$ 쌍은 7,035개다. 이 library를 이후 유한표본 Algorithm 1의 실제 후보 class로도 쓴다.

## 4. $D_S^2$, $\theta_S$, 수치 잔차 (수정 6)

$Z\triangleq(X_1,X_2,\beta^\top W_{-S})$. $Z$가 $X$ 전체를 포함하므로 $(X,W_S,Z)$와 $(W_S,Z)$는 같은 $\sigma$-대수다.

$$D_S^2(\beta)\triangleq\mathbb E\bigl[\{P(A=1\mid W_S,Z)-P(A=1\mid Z)\}^2\bigr],\qquad
\theta_S(\beta)\triangleq\mathbb E\bigl[\mathbb E(Y\mid A=1,Z)-\mathbb E(Y\mid A=0,Z)\bigr].$$

계산은 정규 조건부 사영, probit 닫힌 식 $P(A=1\mid G)=0.1+0.8\Phi(m_G/\sqrt{1+s_G^2})$, Stein 항등식, 2차원 Gauss-Hermite로 한다(v2 3절과 같다).

수치 잔차: 차수 48, 96, 144로 계산해 $r=\max(|v_{96}-v_{48}|,|v_{144}-v_{96}|)$를 기록하고
`numerical_stability_interval` $=[v_{144}-2r,\ v_{144}+2r]$로 부른다. **이것은 엄밀한 오차 한계가 아니라 수렴 진단이다.**
판정은 구간이 아니라 5절의 항등식과 여유(margin)로 한다. 결과 분포를 보고 정하는 문턱값은 두지 않는다.

## 5. Exact arm: 정확한 균형점을 library 안에 설계한다 (수정 2)

### 5.1 균형의 정확한 특성

$C\triangleq\beta^\top W_{-S}$, $p\triangleq\beta^\top b_{\cdot c}$, $q\triangleq\beta^\top b_{\cdot t}$, $v\triangleq\sum_j\beta_j^2\sigma_j^2$, $G\triangleq\alpha_{\mathrm c}U_{\mathrm c}+\alpha_{\mathrm t}U_{\mathrm t}$로 두면
$C=pU_{\mathrm c}+qU_{\mathrm t}+\nu$, $\mathrm{Var}(\nu)=v$이다. $X_2$는 모든 것과 독립이므로 조건부 집합은 사실상 $(X_1,C)$다.

$$\gamma(\beta)\triangleq\bigl(\gamma_{\mathrm c},\gamma_{\mathrm t}\bigr),\qquad
\gamma_{\mathrm c}\triangleq\mathrm{Cov}(G,U_{\mathrm c}\mid X_1,C),\quad
\gamma_{\mathrm t}\triangleq\mathrm{Cov}(G,U_{\mathrm t}\mid X_1,C).$$

$\Sigma\triangleq\mathrm{Var}(X_1,C)=\begin{pmatrix}1+\kappa^2&p\\p&p^2+q^2+v\end{pmatrix}$에 대해 닫힌 식은

$$\gamma_{\mathrm c}=\alpha_{\mathrm c}-\bigl(\alpha_{\mathrm c},\ \alpha_{\mathrm c}p+\alpha_{\mathrm t}q\bigr)\Sigma^{-1}\begin{pmatrix}1\\p\end{pmatrix},\qquad
\gamma_{\mathrm t}=\alpha_{\mathrm t}-\bigl(\alpha_{\mathrm c},\ \alpha_{\mathrm c}p+\alpha_{\mathrm t}q\bigr)\Sigma^{-1}\begin{pmatrix}0\\q\end{pmatrix}.$$

결합 정규성과 probit의 단조성 때문에 다음이 **정확히** 성립한다.

$$D_S^2(\beta)=0\iff B_S\,\gamma(\beta)=0.$$

즉 held-out 블록 $j$마다 $b_{jc}\gamma_{\mathrm c}+b_{jt}\gamma_{\mathrm t}=0$이다.
순수 $U_{\mathrm c}$ 블록만 held-out이면 조건은 $\gamma_{\mathrm c}(\beta)=0$ 하나이고, $U_{\mathrm t}$만 재는 블록만 held-out이면 $\gamma_{\mathrm t}(\beta)=0$ 하나다.
$\operatorname{rank}(B_S)=2$이면 $\gamma(\beta)=0$ 두 조건이 모두 필요하다.

### 5.2 exact-hit cell 구성

두 family에 각각 library 원소 하나를 지정하고, DGP 계수를 풀어 그 원소가 정확한 균형점이 되게 한다.

**Target family** ($S\subseteq\{W_1,W_2,W_3\}$, 순수 $U_{\mathrm c}$ 블록만 held-out, ORC-valid):
지정 $\beta^{*}$는 $W_4+W_5$ (나머지 좌표 0). 그 $(p,q,v)=(2,\ b_{4t}+b_{5t},\ \sigma_4^2+\sigma_5^2)$는 어느 순수 블록을 held-out으로 두든 같다.
$\gamma_{\mathrm c}(\beta^{*})=0$은 $\alpha_{\mathrm t}$에 대해 **선형**이므로 닫힌 식으로 푼다($q\neq0$ 필요).
이 한 번의 선택으로 target family의 **일곱 split 전부**가 같은 $\beta^{*}$에서 정확히 균형을 이룬다.

**Wrong family** ($S=\{W_6\}$, $\{W_7\}$, $U_{\mathrm t}$만 재는 단일 블록 held-out, ORC-invalid):
지정 $\beta^{**}$는 $S=\{W_6\}$에 대해 $W_4+W_7$, $S=\{W_7\}$에 대해 $W_4+W_6$. 대칭을 위해 $b_{6t}=b_{7t}$, $\sigma_6=\sigma_7$로 둔다.
$\gamma_{\mathrm t}(\beta^{**})=0$을 $b_{7t}(=b_{6t})$에 대해 1차원 근 찾기로 푼다. 이 계수는 $\beta^{*}$의 $(p,q,v)$에 들어가지 않으므로 target family의 균형을 흔들지 않는다.
근은 부호 변화 bracket으로 잡고, 이후 census의 $D_S^2$ 잔차로 재확인한다.

$\beta^{**}$를 $\beta^{*}$와 다르게 두는 이유: 같은 channel이면 $Z$가 같아 $\theta$도 같아지므로 wrong family가 $\tau$를 겨냥해 버린다.

지정하지 않은 split(단일 mixed 블록 held-out, rank-2 held-out, 혼합 held-out 등)은 설계 대상이 아니다. census가 찾은 대로 보고한다.

### 5.3 exact arm의 정의

- **exact hit**: $D_S^2(\beta)\le r_S$ (수치 잔차 이하로 0과 구별되지 않음).
- **population candidate**: exact hit가 존재하는 split. 그 외 split은 candidate가 아니며 $\min_\beta D_S^2$만 보고한다.
- **target class** $\mathcal G$: ORC-valid이면서 exact hit인 split. Theorem 1에 의해 $\theta_S=\tau$이고, census는 $|\theta_S-1|\le 2r_S$를 확인한다. 이것이 항등식에 의한 인증이다.
- **wrong class**: ORC-invalid이면서 exact hit인 split. $\theta_S$는 계산값이며, 서로 다른 split의 $\theta$가 `numerical_stability_interval`에서 겹치면 7절 규칙으로 병합한다.

### 5.4 계수 (exact-hit 기준 cell)

| 계수 | 값 |
|---|---|
| $J$ | 7 |
| $b_{\cdot c}$ | $(1,1,1,1,1,0,0)$ |
| $b_{\cdot t}$ | $(0,0,0,-0.6,\ 0.4,\ b_{6t},b_{7t})$, $b_{6t}=b_{7t}$는 5.2에서 푼다 |
| $\sigma$ | $0.85\cdot\mathbf 1_7$ |
| $\kappa$ | 2 |
| $\alpha_{\mathrm c}$ | 1 |
| $\alpha_{\mathrm t}$ | 5.2에서 푼다 |
| $c,\ h,\ s$ | $2.5,\ 0,\ 0.3$ |

격자: $b_{5t}\in\{0.4,0.6\}$, $h\in\{0,0.3\}$, $\sigma$ 균일 또는 $(0.8,0.9,1.0,0.85,0.95,0.9,0.9)$의 $2^3=8$ cell.
각 cell마다 $\alpha_{\mathrm t}$와 $b_{6t}=b_{7t}$를 5.2로 다시 푼다. 풀린 값과 잔차를 ledger에 적는다.

## 6. 동점 집합과 split 내부 selection margin (수정 7)

- 동점 집합 $\mathcal T_S\triangleq\{\beta\in\mathcal B_d: D_S^2(\beta)\le\min_{\beta'}D_S^2(\beta')+\epsilon_{\mathrm{tie}}\}$, $\epsilon_{\mathrm{tie}}=10^{-9}$.
- $\mathcal T_S$의 모든 원소가 같은 $\theta$(구간 겹침)를 주면 그 값이 candidate value다. 다르면 그 cell은 FAIL(`tie_set_ambiguous`).
- **selection margin** $m_S\triangleq\min_{\beta\notin\mathcal T_S}D_S^2(\beta)-\min_{\beta\in\mathcal T_S}D_S^2(\beta)$를 모든 split에 기록한다.
- gate: 모든 population candidate에서 $m_S\ge m_{\min}$, $m_{\min}=10^{-4}$. 이 값은 지금 고정하고 결과를 보고 바꾸지 않는다. 미달이면 `selection_margin_fail`.

$m_S$는 유한표본 empirical objective가 다른 library 원소를 고를 여지를 재는 양이다. split 사이의 gap과 다른 양이다.

## 7. Plurality (수정 4)

$p_\tau$는 최빈 class가 아니다.

$$p_\tau\triangleq\frac{|\mathcal G|}{N_{\mathrm{cand}}},\qquad
p_\star\triangleq\max_{c}\frac{n(c)}{N_{\mathrm{cand}}},$$

$\mathcal G$는 5.3의 ORC 인증 target class, $n(c)$는 wrong class $c$의 크기, $N_{\mathrm{cand}}$는 population candidate 수다.
wrong class의 병합은 구간이 겹칠 때만, 그리고 $p_\star$를 키우는 방향으로만 한다. 겹친 쌍은 `overlap`에 모두 적는다.

A1: $\Delta\triangleq p_\tau-p_\star>0$.
검증 항목: 최빈 class가 $\mathcal G$와 일치하는가를 따로 기록한다(`modal_equals_target`). 이것이 census가 확인하는 내용이지 정의가 아니다.

## 8. 반경 $\rho$: 원고 Section 4 유한표본 공식 (수정 5)

candidate $S$의 AIPW 오차 반경은

$$\varepsilon_{\mathrm{AIPW},S}=\frac{\sigma_S}{\sqrt{n_{\mathrm E}\,\delta_{\mathrm E}}}+\frac{q_{e,S}\,(q_{0,S}+q_{1,S})}{\eta},\qquad \delta_{\mathrm E}=\frac{\delta}{N},$$

$N$은 유지된 candidate 수이고, 따라서

$$\rho\triangleq\max_{S\in\widehat{\mathcal B}}\Bigl[\sigma_S\sqrt{\tfrac{N}{n_{\mathrm E}\,\delta}}+\tfrac{q_{e,S}(q_{0,S}+q_{1,S})}{\eta}\Bigr].$$

**Theorem arm (exact nuisance)**: 4절의 닫힌 식 $e(z)$, $m_a(z)$를 쓰므로 둘째 항이 0이다. $\sigma_S^2$는 정확한 nuisance 아래 AIPW score의 모집단 분산이며, 닫힌 식이 없으면 표기한 seed 구간의 대규모 Monte Carlo로 추정하고 MC 오차를 적는다.
$$\rho_{\mathrm{thm}}(n_{\mathrm E})=\max_{S}\sigma_S\sqrt{N/(n_{\mathrm E}\delta)},\qquad \delta=0.05.$$
$N$은 census의 population candidate 수를 상한으로 쓴다.

**참고치**: Gaussian 최대 분위수 반경은 `rho_asymptotic`으로만 기록하고 유한표본 정리 검증에 쓰지 않는다.

이 공식은 Chebyshev형이라 보수적이다. census는 $g_{\min}$이 주어졌을 때 $4\rho_{\mathrm{thm}}(n_{\mathrm E})\le g_{\min}/1.25$가 되는 최소 $n_{\mathrm E}$를 표로 낸다.
그 $n_{\mathrm E}$가 크면 그것이 사실이고, 확증 $n$은 그 표를 보고 정한다.

## 9. 분리

A2: 서로 다른 모든 candidate value 쌍(target 대 wrong, wrong 대 wrong 모두)이 $|\theta-\theta'|>4\rho_{\mathrm{thm}}(n_{\mathrm E})$를 만족한다.
$g_{\min}$은 그 최소 거리이고, 가장 가까운 쌍을 기록한다.

## 10. Approximate arm은 운영 부록으로만 (수정 3)

exact arm이 census의 판정 arm이다. approximate bound를 쓰는 arm을 부록에 둔다면 다음을 **모두** 바꿔 적는다.

- $b_S\triangleq\Gamma_SD_S/\{\eta(1-\eta)\}$, $b\triangleq\max_{S\in\mathcal G}b_S$
- 유효 반경 $R\triangleq\rho+b$, 그래프 간선 $|\widehat\theta_S-\widehat\theta_{S'}|\le2R$, 분리 $g_{\min}>4R$, 결론 $|\widehat\tau-\tau|\le R$
- plurality는 exact 값 빈도가 아니라 ORC와 bias bound로 먼저 인증한 target band class와 각 wrong class의 수 비교

이 arm은 census 판정에 쓰지 않는다.

## 11. Census 산출물

파일 `results/population_census/<cell_id>.json`, 코드 `code/population_census.py`, 검증 `code/test_population_census.py`.

```
{
  "complete": true, "prd": "...v3.md", "prd_sha256": "...", "source_sha256": {...},
  "cell_id": "...", "coefficients": {...}, "solved": {"alpha_t": ..., "b_6t": ..., "bracket": [...], "residual": ...},
  "alphabet": [-1,0,1], "quadrature_orders": [48,96,144], "eps_tie": 1e-9, "m_min": 1e-4,
  "rows": [ { "split": [...], "size": k, "category": "...",
      "orc_rowspace_valid": bool,
      "min_discrepancy": ..., "discrepancy_residual": ..., "exact_hit": bool,
      "tie_set": [[...]], "tie_set_size": n, "selection_margin": ...,
      "theta": ..., "theta_residual": ..., "numerical_stability_interval": [lo, hi],
      "tie_set_theta_range": [lo, hi],
      "class": "target" | "wrong:<id>" | "none" } ],
  "target_class": {"splits": [...], "count": n, "max_abs_theta_minus_tau": ...},
  "wrong_classes": [ {"id": ..., "theta_interval": [...], "splits": [...], "count": n, "merged_from": [...]} ],
  "overlap": [...],
  "plurality": {"p_tau": ..., "p_star": ..., "delta": ..., "modal_equals_target": bool},
  "sigma_S": {...}, "rho_thm_table": [{"n_E": ..., "rho": ..., "four_rho": ...}], "rho_asymptotic_reference": ...,
  "separation": {"g_min": ..., "closest_pair": [...], "min_n_E_for_4rho": ...},
  "permutation_test": {"permutation": [...], "max_abs_diff": ..., "pass": bool},
  "verdict": "PASS" | "FAIL", "fail_reasons": [...]
}
```

`complete`는 $2^J-2$개 split이 모두 `rows`에 있을 때만 `true`다.

## 12. Ledger

`results/population_census/LEDGER.md`에 시도한 모든 cell을 적는다. 열: cell_id, 격자 좌표, 풀린 $\alpha_{\mathrm t}$와 $b_{6t}$, candidate 수, $|\mathcal G|$, wrong class 크기들, $p_\tau$, $p_\star$, $\Delta$, $g_{\min}$, $\min m_S$, $4\rho_{\mathrm{thm}}$에 필요한 최소 $n_{\mathrm E}$, 판정, 탈락 이유
(`tie_set_ambiguous`, `selection_margin_fail`, `plurality_fail`, `separation_fail`, `orc_consequence_fail`, `permutation_fail`, `no_exact_hit`). 탈락 cell도 지우지 않는다.

## 13. 치환 검사

고정 seed 하나로 블록 인덱스 치환 $\pi$를 만들어 loading과 $\sigma$에 적용한 뒤 census 전체를 다시 돈다.
치환 전 split $S$와 치환 후 split $\pi(S)$의 `min_discrepancy`, 정렬한 `tie_set`, `theta`, `selection_margin`이 잔차 이내로 같아야 한다.
이후 유한표본 알고리즘에도 같은 검사를 적용한다.

## 14. 승인 기준 요약

| 코드 | 기준 |
|---|---|
| A1 | $\Delta=p_\tau-p_\star>0$, $p_\tau$는 ORC 인증 target class 비율 |
| A2 | 모든 candidate value 쌍이 $4\rho_{\mathrm{thm}}(n_{\mathrm E})$보다 멀리 있음. $n_{\mathrm E}$ 표 첨부 |
| A3 | $\mathcal G$의 모든 split에서 $|\theta_S-1|\le2r_S$. ORC-invalid exact hit의 동점 집합이 split 안에서 한 값 |
| A4 | 치환 검사 통과, 학습기와 census 코드에 블록 역할 기호 없음 |
| A5 | 모든 candidate에서 $m_S\ge m_{\min}$ |
| A6 | target family 일곱 split과 wrong family 두 split이 모두 exact hit (설계가 실제로 성립함) |

## 15. 승인 뒤 순서

1. `code/population_census.py`와 test 구현. 5.2의 계수 풀이 함수 포함. 코드에 블록 역할 기호가 없어야 한다.
2. 8 cell 전수 census, 치환 검사, ledger.
3. 통과 cell의 $g_{\min}$, $\min m_S$, $\rho_{\mathrm{thm}}$ 대 $n_{\mathrm E}$ 표 보고.
4. 검토 뒤에만 fresh-seed 확증 규약 봉인. 봉인 문서에 census 파일 해시와 cell_id를 적는다.

## 16. 바꾸지 않는 것

$J=7$ 결과와 진단 그림은 잘못 지정된 $X/W$ 분할에 대한 negative-control stress test로 보존한다.
