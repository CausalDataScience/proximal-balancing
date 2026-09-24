# IHDP와 ACIC 사전 등록 v1

작성: 2026-09-22 CDT. 확증 replication을 열기 전에 고정한다.

## 0. 공통 recipe

두 벤치마크 모두 공변량이 전부 관측되므로 배포된 상태로는 proxy 문제가 아니다. Twins와 **같은 recipe**로 만든다. 코드는 `code/hidden_proxy_recipe.py` 하나를 세 자료가 공유한다.

결과면을 미는 실제 공변량 하나를 표준화해 $U$로 삼고 $X$에서 뺀 뒤, SCM-1의 기제를 그대로 얹는다.
$$M_j=U+0.85\epsilon_j\ (j{=}0..3),\quad M_4=U-3.0I+0.85\epsilon_4,\quad W_5=Q_5[M_0,I]$$
$$P(A{=}1)=0.1+0.8\Phi(1.5U+2.0I+w^\top\tilde X),\qquad w\sim N(0,0.1^2)\ \text{한 번 뽑아 고정}$$
$$Y=A\,Y(1)+(1-A)\,Y(0),\qquad \tau=\tfrac1n\textstyle\sum_i\{Y_i(1)-Y_i(0)\}\ \text{정확}$$

실제인 것은 공변량, 숨은 교란, 두 잠재결과다. 생성하는 것은 $I$, $A$, proxy 다섯 블록이다. **proxy는 생성물**이며 논문에 그대로 적는다.

## 1. 두 자료

| | IHDP | ACIC 2016 |
|---|---|---|
| 출처 | Hill (2011), Johansson et al. 배포 npz. `materials/real_world_data/ihdp/`, manifest 기록 | Collaborative Perinatal Project. `materials/real_world_data/acic2016/` |
| 크기 | $n=747$ (train 672 + test 75 합침), 공변량 25개 중 24개 사용 | $n=4{,}802$, one-hot 뒤 79개 중 78개 사용 |
| 숨긴 $U$ | 열 5번, replication 0에서 $\mu_0$와 상관 $0.858$ | `x_44`, realization 1에서 $\mu_0$와 상관 $0.689$ |
| 잠재결과 | `yf`와 `ycf`에서 복원 | `y0`, `y1` 직접 |
| seed | `98,400,000 + 1000r`, $w$는 `98,400,777` | `98,500,000 + 1000r`, $w$는 `98,500,777` |
| 겹당 행 | 186 | 1,200 |

## 2. 개발 replication에서 본 것 (확증에서 제외)

**replication/realization 1을 설계 점검용으로 이미 열었다.** 본 것은 아래가 전부다.

| | $\tau$ | naive | 편향 | corr$(A,U)$ | 블록별 $R^2$ (깨끗한 넷) |
|---|---:|---:|---:|---:|---|
| IHDP | $+4.029$ | $+4.187$ | $+0.158$ | $0.317$ | 0.57–0.59 |
| ACIC | $+2.151$ | $+0.499$ | $-1.652$ | $0.330$ | 0.58 |

**IHDP는 교란이 약하다.** 편향이 참효과의 4%다. 원인은 숨긴 열 5번이 실현된 $Y(0)$과 상관 $0.268$뿐이고 $Y(1)$과는 $0.071$이라 대비에서 상당 부분 상쇄되기 때문이다. 따라서 IHDP는 "고칠 교란이 큰 경우"가 아니라 **"교란이 작은 경우"** 로 보고한다. 묻는 것은 PROBE가 이미 거의 맞는 답을 과잉 교정하지 않는가다. ACIC는 편향이 참효과의 77%로 교정할 것이 크다.

결과를 본 뒤 $U$의 선택, 계수 $1.5$, 블록 구성, seed를 바꾸지 않는다. 바꾸려면 v2를 새로 적는다.

## 3. 확증

- IHDP: replication **2–11** (10개). 100개 중 앞에서 순서대로.
- ACIC: realization **2–10** (9개). 배포된 10개 중 1번을 개발로 썼으므로 남은 전부.
- fold seed는 자료 seed $+7$. 봉인된 `run_dataset`를 고치지 않는다. `Level(d_blocks=(1,1,1,1,2), k=2)`.
- 문턱은 SCM-1의 $1.5\times10^{-3}$, 봉인 규칙과 pooled 규칙을 모두, 격자 $\{0.5,1,2,4\}\times10^{-3}$로 민감도.
- 비교군은 Twins와 같다.
- CPU만, worker 2개, BLAS 스레드 2개.

## 4. 보고

replication별 $\tau$ 대비 오차(PROBE 두 규칙, 비교군 전부), 반환 종류와 보류 집합, screen TPR/FPR(census는 SCM-1과 같다: 정답 7, $W_4$ 1, $W_5$ 포함 22), 평균 절대오차와 짝지은 한쪽 상한. 통과 판정은 하지 않는다.
