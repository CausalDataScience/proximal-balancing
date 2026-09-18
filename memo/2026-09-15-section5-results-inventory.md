# Section 5 실험 결과 목록 (ICLR 원고용, 2026-09-15)

Section 5 설계를 시작하기 위해 `results/`의 결과 파일 171개를 정리했다. 각 실험을 원고의 정리와 짝짓고,
본문, 부록, 원고에서 제외 중 어디에 둘지 추천한다. 실행 로그는 `results/logs/section5_experiments_2026-09-15/INDEX.md`에 있다.

## 0. 고르는 기준

- **본문과 부록의 증거가 될 수 있는 것.** 세 가지다.
  - 결과를 보기 전에 seed, 표본 크기, 학습기, 관문을 봉인한 확증.
  - 봉인된 적합을 그대로 다시 평가하고, 재현 차이가 0임을 확인한 분석.
  - 적합 없이 구조식에서 계산한 모집단 값.
- **증거로 쓰지 않는 것.** 개발 실행, 파일럿, 봉인 전 실행, 중간 버전은 과정 기록으로만 둔다.
- **실패한 확증도 보고한다.** 결과를 본 뒤 설계를 바꾼 경우에는 무엇을 왜 바꿨는지 부록에 적는다.
- **그림 규칙.** 본문 그림은 3–5칸이고, 한 칸짜리는 wrapfigure다.

## 1. 본문 후보: 정리별

### 5.1 Theorem 1, 미리 정한 held-out 블록

| 실험 | 설계 | 결과 | 파일 | 추천 |
|---|---|---|---|---|
| T1 블록 고정, SCM family | SCM-1–5, $n=6{,}000$, SCM마다 자료 20개, 봉인 family 확증의 적합, 재현 차이 0 | held-out $W_1$ 평균 $0.996$–$1.000$, 평균 절대오차 $0.048$–$0.053$; 나머지 깨끗한 선택 $0.027$–$0.052$; held-out $W_4$ 평균 $-0.47$ ~ $-0.44$; $X$ 조정 $0.03$–$0.05$; $(X,W)$ 조정 $0.80$–$0.82$ | `results/t1_fixed_block_scm_family_v1.json` | **본문**, 그림 1×5 완성 |
| T1 블록 고정, SCM-1 큰 표본 | $n=12{,}000$, 자료 20개, 봉인 E12 적합 | $W_1$ $1.009$ (오차 $0.038$), $W_4$ $-0.446$ | `results/t1_fixed_block_scm1_v1.json` | 부록 |
| 모집단 검산, SCM-1 | 구조식 quadrature | $W_1$의 두 균형 표현 모두 $1.0000$; $W_4$는 $-0.4432$와 $0.9530$ | 스토리라인 메모 6절 | 본문 한 문장, 부록 표 |

### 5.2 Theorem 2와 Corollary 1, 모르는 유효 블록

| 실험 | 설계 | 결과 | 파일 | 추천 |
|---|---|---|---|---|
| SCM family 확증 (`restricted-brier-rotate4-g-confirm-v2`) | SCM-1–5, $n=6{,}000$, SCM마다 fresh seed 20개, 네 rotation, Algorithm 1 | **PASS 5/5**. PROBE 평균 절대오차 $0.028$–$0.030$, 90번째 백분위 $0.050$–$0.060$, 답한 비율 $1.00$. 짝지은 상한: $X$ 대비 $-0.906$ ~ $-0.889$, $(X,W)$ 대비 $-0.149$ ~ $-0.134$, oracle 초과분 $0.031$–$0.036$ | `results/probe_structured_scm_brier_g_confirm_summary_v1.json`, `..._v2_g1.json`–`g5.json` | **본문** |
| 같은 확증의 균형 검사 기록 | 위와 같음 | 오염된 $W_4$가 네 rotation 모두 통과한 자료: 20개 중 16, 16, 16, 18, 17. 반환된 무리에 $W_4$가 들어간 자료: 모두 0. 자료당 통과한 깨끗한 선택 4–7개(평균 5.9–6.2) | 위 raw 파일 | 본문 그림 1×5 (평균, 중앙값, PROBE 다수결), 집계 스크립트 필요 |

주장 범위: 선형 probit critic을 쓴 제한된 학습기이며 신경망 인코더 확증이 아니다.

### 5.3 Theorem 3, 근사 균형

| 실험 | 설계 | 결과 | 파일 | 추천 |
|---|---|---|---|---|
| 모집단 균형 근 곡선 | SCM-1 상수, oriented split 30개, $r$ 격자, 적합 없음 | 균형 가능한 8개 중 7개가 $\tau$를 겨냥하고, $W_4$ 하나는 불균형 0에서 bias $1.44$. $W_5$를 held-out으로 두는 선택과 $W_4$를 포함한 합집합은 균형 근이 없다 | `results/population_certificate_curve_v1.json` | 본문 1×3 또는 wrapfigure. $\Gamma_j(\phi)$ 계산이 남음 |

### 5.4 Theorem 4와 5, 고차원에서 학습한 표현

| 실험 | 설계 | 판정 | 핵심 수치 | 파일 | 추천 |
|---|---|---|---|---|---|
| SCM-1, 7개 숫자 | E12, $n=12{,}000$, 자료 20개 | PASS | PROBE 평균 절대오차 $0.0235$ | `results/e12_scm1_ncurve_confirm_v2.json` | **본문** 칸 |
| SCM-6-v3, 1,282차원 벡터 | $n=12{,}000$, fresh seed 30개 | 관문 13개 중 12개 | 오차 중앙값 $0.030$, 90번째 백분위 $0.086$; 짝지은 상한 $X$ $-0.884$, raw $-0.164$, 최선 요약 $-0.106$; **oracle 초과분 $0.0532$가 관문 $0.05$ 초과** | `results/scm6_v3/scores_confirmation_v1.json` | **본문** 칸, 미달을 적는다 |
| SCM-6-v3 큰 표본 | $n=24{,}000$, 자료 10개 | PASS 13/13 | 90번째 백분위 $0.055$, oracle 초과분 $0.0446$ | `results/scm6_v3/scores_larger_n_v1.json` | 본문 한 문장 또는 부록 |
| SCM-7-v3, 이미지 | $n=12{,}000$, fresh seed 20개 | **FAIL 8/13** | 답한 비율 $0.80$, oracle 초과분 $0.066$, screen TPR $0.694$, learned target 90번째 백분위 $0.167$ | `results/scm7_v3/scores_confirmation_v1.json` | 부록, 반드시 보고 |
| SCM-7-v4, 이미지 | 이미지 전용 문턱 $4.2\times10^{-3}$(v3 자료로 보정), fresh seed 15개 | **PASS 13/13** | 평균 절대오차 $0.030$, 90번째 백분위 $0.064$, oracle 초과분 $0.039$; 짝지은 상한 $X$ $-0.886$, raw CNN $-0.647$, 최선 요약 $-0.342$; learned target $0.033$ / $0.098$ | `results/scm7_v4/scores_confirmation_v1.json`, `tolerance_calibration.json` | **본문** 칸 |

본문 그림은 1×3(숫자, 벡터, 이미지)이다. 지금의 그림 1이 이 모양이다.

### 5.5 Theorem 6과 Corollary 3, 4, 탐색 예산

| 실험 | 설계 | 결과 | 파일 | 추천 |
|---|---|---|---|---|
| E11-v2 정확한 유한상태 확증 v2 | 5개 상태, 정리의 모든 조건을 계산으로 확인 | PASS (점검 항목 모두 참) | `results/e11_v2_exact_confirm_v2.json` | 부록 |
| E11-v2 $m$-curve | $m=1,\dots,14$, 2,000 반복, $\rho\approx0.20$ | $m=1$에서 보장 $0.158$, 관측 $0.862$; $m=14$에서 보장 $0.940$, 관측 $1.000$. 모든 $m$에서 관측이 보장 이상 | `results/e11_v2_mcurve_v1.json` | **본문** wrapfigure |
| E12 SCM-1 표본 크기 곡선 | 중첩 표본, 자료 20개 | PASS. $n=1,500$: 20/20 답, 평균 절대오차 $0.0556$; $n=3,000$: 20/20 답, 평균 절대오차 $0.0512$; $n=6,000$: 20/20 답, 평균 절대오차 $0.0264$; $n=12,000$: 20/20 답, 평균 절대오차 $0.0235$ | `results/e12_scm1_ncurve_confirm_v2.json` | 부록 또는 5.4 한 문장 |

## 2. 부록에 과정으로 적을 것

| 기록 | 내용 | 파일 |
|---|---|---|
| SCM family 첫 확증 실패와 설계 변경 | 이전 family(outcome 잡음이 큰 f 계열)의 확증에서 5개 중 4개가 90번째 백분위 관문($0.10$)만 $0.105$–$0.109$로 넘었다. 그 뒤 outcome 잡음을 낮춘 g 계열(지금의 SCM-1–5)로 바꾸고 fresh seed로 다시 확증했다 | `results/probe_structured_scm_final_summary_v1.json` (`confirm_v1_failures`) |
| structured-moment positive control | 같은 g 계열, 자료 40개씩, 5/5 PASS, 평균 절대오차 $0.026$–$0.029$ | 같은 파일 (`g_confirm`) |
| SCM-6 v1 실패 | 봉인 전 규약, 평균 절대오차 $0.65$, v2 개발을 거쳐 v3로 대체 | `results/scm6_highdim_confirm_v1.json`, 감사 메모 3절과 A.5 |
| SCM-7 v3 실패와 v4 문턱 재보정 | v3 자료를 개발 자료로 돌려 문턱을 정한 규칙과 검증 절반의 결과 | `results/scm7_v4/tolerance_calibration.json`, 감사 메모 C.4–C.6 |
| 결함으로 대체된 확증 | E11-v2 v1(반지름 반올림), E12 v1(중첩 표본 아님). 둘 다 고친 뒤 다시 돌려 PASS | `*_superseded*.json`, 감사 메모 A.1, A.3 |
| 계산 자원 | GPU 553.8분, 자료당 시간 | 로그 INDEX |

## 3. 원고에 쓰지 않는 것

- **개발과 진단 실행:** `dgp_*`, `probe_e1_*`, `probe_e4_mnist_*`, `probe_e8_*`–`probe_e10_*`, `probe_structured_scm_*dev*`, `*diagnostic*`, `*candidate_ledger*`, `scm6_v2_development_*`, `scm7_image_development_*`, `scm6_objective_curves_v1.json`, `scm6_highdim_dev_v1.json`
- **이전 원고의 8월 실험:** 근사 SCM 1–2(`honest_canonical_representation_*`), Track A/B(`exact_injectivity_*`), WSC 벤치마크(`wsc_*`). 이전 이론과 cross-mask 방법에 기반하며, 지금의 PROBE(블록 제외, 균형 검사, 다수결)와 다르다.
- **이미 제외가 결정된 역사 자료:** `benchmark_suite_*`, `external_*`, `rerun_*`, KMU, Zika, finite-state bridge와 certificate 자료. `ARTIFACTS.md`의 Historical Boundary 절을 따른다.

## 4. 정해야 할 것

1. **실제 자료:** WSC 벤치마크는 이전 방법으로 돌렸고 overlap 관문을 10번 중 0번 통과했다. ICLR 원고에서 뺄지, 아니면 새 규약으로 PROBE를 실제 자료에 다시 돌릴지 정해야 한다.
2. **5.2 그림:** 저장된 적합에서 평균, 중앙값, 다수결을 다시 계산하는 스크립트를 돌린다(CPU만 사용).
3. **5.3:** $\Gamma_j(\phi)$를 계산해 bound 선을 그릴지, 곡선만 보일지 정한다.
4. **5.4:** SCM-1 칸은 E12($n=12{,}000$)로 해서 다른 두 렌더링과 표본 크기를 맞추는 것을 추천한다.
5. **5.5:** E12 표본 크기 곡선을 본문에 둘지 부록에 둘지 정한다.
6. **`ARTIFACTS.md` 갱신:** 이 목록이 확정되면 9월 9일 기준 목록을 새 증거 목록으로 바꾼다.
