# DGP A--E 구현 보고서 (모니터링 에이전트용)

작성일: 2026-09-11
대상 계획: `memo/2026-09-11-dgp-a-e-experiment-plan.md` (이하 "계획")
구현 범위: 계획 2--9절의 자료생성과 주 방법, 11절의 요약, 16절의 검증
지위: 구현과 단위검증 완료, 주 실험(pilot/confirmatory) 미실행

## 0. 이 보고서의 사용법

모니터링 에이전트는 아래 순서로 확인한다.

1. 2절의 구현 현황표에서 각 계획 조항의 상태를 본다.
2. 3절의 검증 증거를 재실행해 동일한 출력이 나오는지 확인한다.
3. 4절의 구현 판단(계획이 지정하지 않은 부분)이 타당한지 검토한다. 이 부분이 감시의 핵심이다.
4. 5절의 미구현 항목이 현재 주장의 범위를 어떻게 제한하는지 확인한다.
5. 6절의 계산 예산을 보고 실행 규모 결정이 필요한지 판단한다.
6. 8절의 금지 주장 목록을 기준으로 이후 보고가 과장되지 않는지 감시한다.

이 보고서는 결과 보고서가 아니다. 성능 수치는 아직 없다.

## 1. 산출물

| 파일 | 줄 수 | 내용 |
|---|---|---|
| `code/probe_dgp_ae.py` | 303 | 공통 SCM, DGP A--E, seed namespace, 후보 집합, oracle/raw/X 조정집합 |
| `code/probe_brier_neural.py` | 551 | 표현, nested Brier critic, 학습, audit, screen, nuisance, AIPW, linking, aggregation, 영상 raw baseline |
| `code/run_dgp_ae.py` | 234 | replicate 실행, 세 rotation, 30개 후보, 비교군, 결과 직렬화 |
| `code/summarize_dgp_ae.py` | 150 | 반환 분포, 조건부 오차, paired Bonferroni-$t$, screen TPR/FPR, set-valued 평가 |
| `code/test_probe_dgp_ae.py` | 107 | 계획 16.1 생성 단위검증 |
| `code/test_probe_brier_neural.py` | 101 | 계획 16.2--16.3 split/leakage와 알고리즘 검증 |

기존 파일은 수정하지 않았다. `code/probe_e1_pipeline.py`, `code/probe_scms.py`, `EXPERIMENTS.md`, 원고는 그대로다.

## 2. 계획 조항별 구현 현황

| 계획 조항 | 상태 | 위치 |
|---|---|---|
| 2.1 고정 계수와 `b_good` 8개 값 | 구현·검증 | `probe_dgp_ae.coefficients` |
| 2.2 공통 외생변수, $C=0.5U+\sqrt{0.75}\epsilon_C$ | 구현·검증 | `probe_dgp_ae.generate` |
| 2.3 A/C/D/E 처치와 결과 | 구현·검증 | 같음 |
| 2.4 $W_0=(I_1,I_2,C)$ | 구현·검증 | 같음 |
| 2.5 oracle $(X,U,C)$, raw, $X$, $(X,C)$ | 구현 | `oracle_matrix`, `raw_matrix`, `xc_matrix` |
| 3 DGP A | 구현·검증 | `generate("A", ...)` |
| 4 DGP B 다항식과 B 전용 처치·결과 | 구현·검증 | `generate("B", ...)` |
| 5 DGP C Poisson과 log1p 표준화 | 구현·검증 | `generate("C", ...)`, `build_standardizers` |
| 6 DGP D 직교 template과 렌더링 | 구현·검증 | `d_templates`, `generate("D", ...)` |
| 7 DGP E MNIST pool, corruption, numeric 채널 | 구현·검증 | `mnist_pool`, `generate("E", ...)` |
| 8.1 30개 후보 전수, pruning 없음 | 구현·검증 | `all_subsets`, runner |
| 8.2 unit split과 세 rotation | 구현·검증 | `run_dgp_ae.folds` |
| 8.3 D 내부 50/25/25 독립 audit | 구현·검증 | `run_dgp_ae.d_subsplit` |
| 8.4 block encoder와 fusion, $Z\in\mathbb R^4$ | 구현 | `Representation` |
| 8.5 nested Brier critic, residual zero 동치 | 구현·검증 | `AugmentedCritic` |
| 8.6 alternating AdamW, 고정 hyperparameter, restart 3 | 구현 | `train_representation` |
| 8.7 discrepancy screen $t_n=0.02+2\hat s$ | 구현 | `audit_gap` |
| 8.8 overlap screen (N에서 98%) | 구현 | `overlap_pass` |
| 8.9 nuisance head와 AIPW score | 구현 | `fit_nuisances`, `aipw_scores` |
| 8.10 multiplier bootstrap, linking, median, tie 유지 | 구현·검증 | `linking_radius`, `aggregate` |
| 9.1 비교군 6종 | 부분 구현 | PROBE-RFF 제외 5종 |
| 9.2 영상 raw의 end-to-end CNN 동등 예산 | 구현 | `fit_nuisances_blocks` |
| 9.3 PROBE-RFF reference arm | 미구현 | 5절 참조 |
| 9.4 privileged diagnostics 4종 | 미구현 | 5절 참조 |
| 10.1--10.2 pilot/confirmatory seed 분리 | 구현 | `PHASE_ID`, `--phase` |
| 10.4 $m$ ablation | 미구현 | 5절 참조 |
| 11.1--11.3 paired Bonferroni, 무조건 반환율, set-valued | 구현 | `summarize_dgp_ae` |
| 11.4 selection 포함 coverage | 미구현 | 5절 참조 |
| 12.1--12.3 failure control과 aggregation boundary | 미구현 | 5절 참조 |
| 15.1 seed namespace와 tuple 구성 | 구현·검증 | `run_key` |
| 15.2 metadata 저장 | 부분 구현 | peak memory 미기록 |
| 16.1--16.3 단위검증 | 구현·통과 | 두 test 파일 |
| 16.4 reporting test | 부분 구현 | `--expect-replicates` gate 있음, label 분리는 보고 시 적용 |

## 3. 검증 증거

재실행 명령과 기대 출력이다.

```bash
cd code && python3 test_probe_dgp_ae.py      # 36개 검사, 마지막 줄 "all generation tests passed"
cd code && python3 test_probe_brier_neural.py # 12개 검사, 마지막 줄 "all algorithm tests passed"
```

생성 검증에서 확인한 주요 수치다.

- `b_good`가 계획 2.1의 여덟 값과 $10^{-12}$ 이내로 일치한다.
- 다섯 DGP 모두에서 $\operatorname{Var}(U)$, $\operatorname{Var}(C)$가 1에 가깝고 $\operatorname{Corr}(U,C)=0.50$이다.
- 다섯 DGP 모두 $Y(1)-Y(0)=1$이 정확히 성립하고 structural propensity가 $[0.05,0.95]$ 안에 있다.
- naive 대비가 다섯 DGP 모두 음수다: A $-2.24$, B $-1.81$, C $-2.25$, D $-2.15$, E $-2.37$. 계획 17절의 1번 중단 규칙에 해당하는 calibration failure는 없다.
- B의 첫 conditional mean 미분 하한 최소값이 $0.4489>0$이다.
- C의 log-rate loading matrix rank가 2이고 특이값이 $(0.863, 0.863)$이다.
- D의 block별 세 template 직교성 오차가 $8.9\times10^{-16}$이고 관측 proxy 차원이 4099다.
- E의 pool이 class당 1000장이고 class 간 exact pixel 중복이 0건이다. class 빈도 최대 편차 0.0018, block별 corruption rate $(0.250,0.248,0.248,0.250)$, corrupted-label channel 최소 특이값 0.7222, 조건수 1.38이다.

알고리즘 검증에서 확인한 항목이다.

- augmented critic의 residual branch를 0으로 두면 base critic과 출력이 일치한다.
- 세 fold가 서로소이고 전체를 덮으며, 모든 unit이 정확히 한 번 $E$가 된다.
- $D$가 50/25/25로 나뉘고 세 부분의 합집합이 $D$와 같다.
- standardizer 통계가 다른 fold 사용 후에도 바뀌지 않는다.
- 후보 집합이 30개이고 pruning이 없다.
- survivor 0이면 no-return, 1이면 single_survivor, 최대 component가 복수면 tie set을 반환하고 point 값을 내지 않는다.
- 짝수 component의 median이 중앙 두 값의 평균이다.
- linking radius가 seed에서 재현되고 최대 표준오차 규모에 있다.

## 4. 구현 판단 (계획이 지정하지 않은 부분)

감시 대상이다. 각 항목은 계획 문구가 비어 있는 지점이며, 바꾸려면 protocol amendment가 필요하다.

**D1. 영상 held-out block에 대한 augmented critic의 특징 추출.**
계획 8.5는 critic을 width $(128,64)$ MLP로 고정하지만, D와 E에서 $T_S=(X,W_S)$는 image를 포함한다.
구현은 augmented critic에 계획 6.3과 동일한 CNN 사본과 numeric MLP를 부여했다. 근거는 critic이 표현과
같은 architecture로 image를 보아야 nested 포함 관계와 compute parity가 유지된다는 것이다.
대안은 pixel flatten인데 critic의 표현력이 급감해 discrepancy가 과소평가된다.

**D2. restart 선택 시 overlap 위반의 계산 위치.**
계획 8.6은 "overlap violation 2% 이하"인 restart를 고르라고 하지만 overlap screen은 8.8에서 $N$으로
정의된다. restart 선택 시점에 $N$을 보면 8.2의 분리를 깬다. 구현은 base critic $q_0$를 $D_{\mathrm{val}}$에서
평가한 fitted propensity로 위반 비율을 계산한다. 이 값은 결과 record에 `val_overlap_violation`으로 저장된다.

**D3. count 전처리 통계의 출처.**
계획 5.2의 "training split"을 $D_{\mathrm{fit}}$로 읽었다. standardizer 전체가 같은 split에서 한 번만
적합되고 $N$과 $E$에서 재적합되지 않는다.

**D4. naive 비교군의 계산 표본.**
rotation별 $E$에서 계산하고 세 값을 평균했다. 다른 비교군과 같은 unit을 쓰기 위한 선택이다.

**D5. multiplier bootstrap의 $n$.**
세 rotation을 합치면 각 unit이 정확히 한 번 $E$에 들어가므로 pooled score 벡터의 길이가 전체 표본 크기와
같다. 계획 8.10의 $n^{-1}\sum_{i=1}^n$을 이 전체 표본으로 읽었다.

**D6. retained 후보의 정의.**
계획 8.9의 intersection rule대로 세 rotation 모두에서 두 screen을 통과한 후보만 retained로 둔다. 통과하지
못한 rotation이 있는 후보는 pooled score를 만들지 않는다.

## 5. 미구현 항목과 그 영향

| 항목 | 계획 조항 | 영향 | 권장 순서 |
|---|---|---|---|
| PROBE-RFF reference arm | 9.3 | 과거 s9 recipe와의 연결을 보일 수 없다. 주 판정에는 불필요하다. | 3 |
| privileged diagnostics 4종 | 9.4 | 실패 시 원인 분해가 약해진다. D/E 실패가 encoder 때문인지 정보 때문인지 가르는 데 필요하다. | 2 |
| failure control 11종 | 12.1--12.2 | 계획 17절의 중단 규칙 5, 6을 판정할 수 없다. | 2 |
| aggregation boundary 계산 | 12.3 | target plurality가 성립하는지 사전 확인할 수 없다. | 1 |
| $m$ ablation | 10.4 | 주 결과에 영향 없다. | 4 |
| selection 포함 coverage | 11.4 | coverage 주장을 할 수 없다. | 3 |
| peak memory 기록 | 15.2 | 예산 산출의 정확도만 낮춘다. | 4 |

권장 순서 1번을 먼저 하는 이유는 계획 12.3이 지적한 대로 30개 후보 중 15:15 구조가 target plurality를
보장하지 않기 때문이다. 이것이 실패하면 pilot을 돌리기 전에 설계를 고쳐야 한다.

## 6. 계산 예산

측정값은 DGP A, $n=6000$, 후보 30개, rotation 3회, 병렬 작업자 5개 기준 replicate당 91--163초다.
CPU 10코어 장비이며 GPU를 쓰지 않는다. 이 값에서 선형으로 늘리면 A의 confirmatory 한 cell(50 replicate)은
약 1.5--2.5시간이고, $n$ 네 개를 합치면 $n$에 대략 비례하여 약 12--20시간이다. B와 C는 같은 규모로 보고,
D와 E는 CNN 때문에 훨씬 크므로 별도 측정이 필요하다. 계획 10.3이 요구한 대로 전체 실행시간을 미리 약속하지
않는다.

작업 구조는 다음과 같다.

- 한 replicate는 후보 30개 $\times$ rotation 3회 $\times$ restart 3회 = 270회의 표현 학습을 요구한다.
  각 학습은 최대 300 epoch이며 critic update는 그 5배다.
- 여기에 후보마다 nuisance head 3개(propensity, 처치군 outcome, 대조군 outcome)를 restart 3회로 적합하므로
  replicate당 nuisance 적합은 $30\times3\times3\times3=810$회다.
- 비교군 4종도 rotation마다 같은 head를 적합한다.
- 계획 10.3의 전체 confirmatory는 5 DGP $\times$ 4 $n$ $\times$ 50 replicate이며, 위 단위작업이 그만큼 반복된다.

D와 E는 CNN을 쓰므로 tabular보다 훨씬 비싸다. 현재 장비는 CPU 10코어다. 측정이 끝나면 전체 예산을
`results/dgp_ae_pilot.json`의 `seconds` 필드로 산출한다. 계획 10.3이 요구한 대로 실행시간을 미리 약속하지
않는다.

## 7. 실행 기록과 구현 중 발견한 결함

### 7.1 실행 기록

| 실행 | 결과 |
|---|---|
| 생성 단위검증 | 전부 통과 |
| 알고리즘 단위검증 | 전부 통과 (encoder 갱신 검사 포함) |
| smoke run (A, $n=1200$, 후보 4개, rotation 1회) | end-to-end 동작 확인. 규모가 작아 수치는 해석하지 않는다. |
| 타이밍 측정 (A, $n=6000$, 후보 30개 $\times$ rotation 3회, 병렬 5) | replicate당 91--163초 |
| pilot 9 replicate (A $n=6000$ 2, A $n=12000$ 3, B $n=6000$ 2, C $n=6000$ 2) | 7.3의 진단 결과. replicate당 91--365초 |

pilot 산출물은 `results/dgp_ae_pilot.json`, `results/dgp_ae_pilot_summary.json`,
`results/dgp_ae_pilot_runs.csv`다. replicate id는 10000번대이며 계획 10.2에 따라 confirmatory와 합치지
않는다. D와 E는 실행하지 않았다. 현재 screen 구성에서는 같은 선택 실패가 먼저 나오므로 amendment 결정 전에
CNN 계산을 쓰지 않는 편이 낫다고 판단했다.

### 7.2 구현 중 고친 결함 두 가지

**F1. 병렬 작업자가 fork 직후 죽었다.** macOS에서 torch가 부모 프로세스의 Objective-C graph runtime을
초기화한 뒤 `fork`로 자식을 만들면 자식이 즉시 crash한다. 로그에 `MPSGraphObject initialize ... Crashing
instead`가 남고 pool은 진행 없이 멈춘다. 24분 동안 한 replicate도 끝나지 않았다. 수정은 pool의 start
method를 `spawn`으로 바꾸고 표본과 standardizer를 worker당 한 번만 pickle로 전달하는 것이다. 이를 위해
count 전용 standardizer를 lambda에서 클래스로 바꿨다. 수정 후 같은 cell이 91초에 끝났다.

**F2. encoder가 한 번도 갱신되지 않았다.** 계획 8.6의 "critic update 5회당 encoder update 1회"를 epoch
안의 batch 번호로 구현했는데, $n=6000$이면 $D_{\mathrm{fit}}$이 1000 unit이고 batch 256 기준 epoch당
batch가 4개뿐이라 조건이 영원히 성립하지 않았다. 표현이 무작위 초기값에 머물렀다. 수정은 critic 갱신
횟수를 epoch를 가로질러 누적하는 것이다. 회귀 검사를 `test_probe_brier_neural.py`에 추가했다.

이 두 결함은 계획의 문제가 아니라 구현의 문제였다. 모니터링 에이전트는 위 두 회귀 검사가 계속 통과하는지
확인한다.

### 7.3 첫 pilot 진단: 두 screen이 구조적으로 올바른 후보를 배제한다

DGP A, $n=6000$, replicate 10000의 rotation 0에서 후보 30개를 모두 기록한 결과다. 여기서 "구조적으로
유효"는 계획 2.4대로 block 0을 held-out set에 넣지 않은 후보, 즉 block 0이 표현 입력으로 들어가는 후보다.

| 지표 | 구조적으로 유효한 15개 | 구조적으로 무효한 15개 |
|---|---|---|
| discrepancy screen 통과 | 15/15 | 15/15 |
| overlap screen 통과 | 0/15 | 15/15 |
| $\widehat D^2$ 원값 평균 | $-0.098$ | $-0.000$ |
| overlap 비율 평균 | 0.936 (최소 0.888) | 1.000 |
| 후보 추정치 $\widehat\theta_S$ | $+0.618\pm0.414$ | $-1.790\pm0.223$ |

세 가지가 동시에 관측된다.

1. **discrepancy screen에 검정력이 없다.** 구조적으로 유효한 후보의 $\widehat D^2$가 모두 음수라
   $\max(0,\cdot)$가 0이 되어 무조건 통과한다. 음수의 크기 $-0.098$은 표본 요동으로 보기 어렵다.
   augmented critic의 residual branch가 $D_{\mathrm{fit}}$ 1000 unit에서 과적합하여 독립 audit split에서
   base critic보다 나빠진 결과로 읽힌다. 계획 8.5는 음수를 표본 요동으로 보고 screen에서만 절단하라고
   하지만, 이 규모에서는 절단이 screen 전체를 무력화한다.
2. **overlap screen이 정확히 반대로 작동한다.** 구조적으로 유효한 후보는 block 0이 표현 입력에 있으므로
   $Z$가 도구변수 $I_1,I_2$를 담을 수 있고 적합된 propensity가 0과 1 쪽으로 퍼져 overlap 비율이
   0.89--0.96에 머문다. 요구치는 0.98이다. 반대로 block 0을 held-out으로 보내면 $Z$에 도구변수가 없어
   propensity가 압축되고 overlap이 1.000이 된다.
3. **따라서 retained 집합이 정확히 틀린 15개다.** `summarize_dgp_ae.py`가 계산한 cell별 screen 성능은
   다음과 같다. TPR은 rotation 평균이다.

   | cell | replicate | 단일반환율 | retained (유효) | screen TPR | screen FPR | MAE PROBE | MAE raw | MAE oracle |
   |---|---|---|---|---|---|---|---|---|
   | A $n=6000$ | 2 | 1.00 | 15.0 (0.0) | 0.06 | 1.00 | 2.853 | 0.088 | 0.060 |
   | A $n=12000$ | 3 | 1.00 | 15.0 (0.0) | 0.01 | 1.00 | 2.848 | 0.180 | 0.027 |
   | B $n=6000$ | 2 | 1.00 | 15.0 (0.0) | 0.07 | 1.00 | 2.547 | 0.363 | 0.050 |
   | C $n=6000$ | 2 | 1.00 | 15.0 (0.0) | 0.04 | 1.00 | 2.958 | 0.342 | 0.035 |

   계획 17절 5번 중단 규칙은 TPR 80% 미만 또는 FPR 20% 초과다. 네 cell 모두 두 조건을 동시에 위반한다.

결과적으로 Algorithm 1은 $-1.92$와 $-1.78$을 반환했고 raw는 $+0.93$과 $+0.90$, oracle은 $+0.93$과
$+0.95$였다. 개별 후보 중에는 참값에 가까운 것이 있다. 같은 rotation에서 $S=\{3,4\}$는 $+1.026$,
$S=\{4\}$는 $+0.917$을 냈다. 문제는 추정이 아니라 선택이다.

같은 양상이 표본 크기와 DGP를 바꾸어도 유지된다.

| cell | retained | 그중 구조적으로 유효 | Algorithm 1 반환 | raw | oracle | $X$ only |
|---|---|---|---|---|---|---|
| A, $n=6000$, rep 10000 | 15/30 | 0 | $-1.921$ | $+0.930$ | $+0.932$ | $-0.775$ |
| A, $n=6000$, rep 10001 | 15/30 | 0 | $-1.785$ | $+0.895$ | $+0.949$ | $-0.762$ |
| A, $n=12000$, rep 10000 | 15/30 | 0 | $-1.865$ | $+0.820$ | $+0.983$ | $-0.786$ |
| B, $n=6000$, rep 10000 | 15/30 | 0 | $-1.563$ | $+0.686$ | $+1.005$ | $-0.562$ |
| B, $n=6000$, rep 10001 | 15/30 | 0 | $-1.531$ | $+0.588$ | $+0.904$ | $-0.588$ |
| C, $n=6000$, rep 10000 | 15/30 | 0 | $-2.046$ | $+0.599$ | $+0.991$ | $-0.697$ |

여섯 cell 모두에서 retained가 15/30이고 그중 구조적으로 유효한 후보가 0개다. 표본 크기를 두 배로 해도,
Gaussian에서 polynomial과 Poisson으로 채널을 바꾸어도 구성이 그대로다. 원인이 표본이나 채널이 아니라
screen의 구조라는 뜻이다. $X$-only가 네 cell 모두에서 부호를 뒤집으므로 계획 17절의 1번 중단 규칙(calibration failure)은
해당하지 않고, oracle이 $0.93$에서 $1.01$ 사이이므로 2번(nuisance failure)도 해당하지 않는다. 남는 분류는
5번 search/aggregation failure다.

이것은 계획이 예상한 실패 유형이다. 계획 12.3은 15:15 구조가 target plurality를 보장하지 않는다고 적었고,
17절은 이를 search/aggregation failure로 분류한다. 계획 0.5는 결과를 본 뒤 threshold나 screen을 바꾸어
주 결과로 재사용하는 것을 금지하므로, 구현자는 이 지점에서 멈추고 protocol amendment를 요청한다.

### 7.4 외부 검토와 그에 따른 정정 (2026-09-11 2차)

모니터링 검토가 여섯 항목을 지적했다. 여섯 항목 모두 저장된 데이터로 사실 확인했고, 전부 실제 결함이었다.
아래는 검증 수치와 수정 내용이다. 7.3의 결론 두 개는 **철회**한다.

**검증 결과**

| 지적 | 확인 방법 | 결과 |
|---|---|---|
| 학습되지 않은 checkpoint 채택 | 저장된 `best_epoch` 집계 | 구조적으로 무효한 후보 405개 중 401개가 epoch 0 선택. 유효한 후보는 405개 중 4개 |
| restart 탈락 규칙 누락 | `val_overlap_violation > 0.02`인 후보의 `screen_pass` | 위반 후보 380개 중 6개가 통과 |
| "추정은 정상, 선택만 문제" | $S=\{3,4\}$의 rotation별 값 | 1.026, 0.021, $-0.105$이며 세 rotation 평균 0.314 |
| overlap 해석 | 생성식 | 처치확률이 구조적으로 $[0.05,0.95]$에 묶이므로 $P(A=1\mid Z)=\mathbb E[e\mid Z]$도 그 안에 있다 |
| headline 준비 표시 | pilot summary | 9 replicate인데 `headline_ready: true`로 저장되어 있었다 |
| 비교군 seed | 코드 | Python 문자열 해시는 프로세스마다 달라진다 |

**철회하는 결론 두 가지**

1. 7.3에서 "overlap screen이 구조적으로 작동한다"고 적었다. 틀렸다. 참 처치확률은 어떤 $Z$에 대해서도
   $[0.05,0.95]$ 안에 있으므로 이론의 overlap 가정은 깨지지 않는다. 관측된 위반은 제약 없는 신경망
   처치확률 모형의 적합 오차이며, 도구변수가 $Z$에 들어갈 때 그 오차가 커진다. 수정 대상은 가정이 아니라
   모형이다.
2. 7.3에서 "문제는 추정이 아니라 선택이다"라고 적었다. 틀렸다. 올바른 후보를 골라도 세 rotation 평균이
   0.314다. 추정 단계에도 문제가 있다.

**수정한 결함 여섯 가지 (F3--F8)**

- **F3. 학습되지 않은 표현을 checkpoint로 채택하지 않는다.** encoder 갱신 횟수를 세고, 한 번도 갱신되지
  않은 epoch은 early stopping 후보에서 제외한다. `encoder_updates`와 `critic_updates`를 결과에 저장한다.
- **F4. 표현을 고정한 뒤 두 critic을 다시 학습하는 audit을 추가했다.** `refit_critics`가 encoder를 얼려
  두고 base critic과 augmented critic을 처음부터 다시 학습한 뒤 fit, validation, audit 세 split에서 두
  Brier risk와 그 차이를 각각 기록한다. 계획 8.5의 audit은 주 값으로 유지하고 이 값은 진단으로 병기한다.
- **F5. 계획 8.6의 강제 탈락 규칙을 구현했다.** 선택된 restart가 validation overlap 허용치를 위반하면
  후보가 screen을 통과하지 못한다.
- **F6. 요약 gate를 고쳤다.** `--expect-replicates`가 없거나, 원본 파일이 미완이거나, phase가 섞이거나
  confirmatory가 아니면 `headline_ready`가 거짓이고 그 이유를 함께 저장한다. acceptable-return rate의
  분모를 전체 replicate로 바꿨다.
- **F7. 비교군 seed를 고정 정수 대응표로 바꿨다.**
- **F8. linking radius를 실제로 평가된 unit만으로 계산한다.** 평가되지 않은 열을 0으로 채우지 않는다.
  회귀 검사에서 동일한 점수만 있을 때 반경이 0이 되는 것을 확인한다.

회귀 검사도 고쳤다. 이전 검사는 표본 200개, batch 256, 3 epoch로 encoder 갱신이 0회인데도 통과했다. 지금은
갱신 횟수가 1 이상인지와 선택된 parameter가 새로 초기화한 표현과 다른지를 직접 확인한다.

**수정 후 pilot 재실행 (A, $n=6000$, replicate 10000--10001)**

| 후보군 | epoch 0 선택 | encoder 갱신 중앙값 | 학습 critic의 $\widehat D^2$ | 재학습 critic의 audit gap | restart 적격 | screen 통과 |
|---|---|---|---|---|---|---|
| 구조적으로 유효한 15개 | 0 | 77 | $-0.0976$ | $+0.0005$ | 1/15 | 0/15 |
| 구조적으로 무효한 15개 | 0 | 17 | $+0.0032$ | $+0.1076$ | 15/15 | 15/15 |

**이 표가 진단을 바꾼다.** 표현을 고정하고 critic을 제대로 학습하면 discrepancy는 완벽하게 분리한다.
구조적으로 유효한 후보는 $+0.0005$, 무효한 후보는 $+0.1076$이다. 학습 중 critic으로 계산한 값은 각각
$-0.0976$과 $+0.0032$로 부호가 뒤집혀 있었다. 즉 계획 8.5의 audit이 학습 중 critic을 쓰는 한 신호가
반전된다.

사후 진단으로 같은 실행에 재학습 audit만 적용하면 retained가 정확히 구조적으로 유효한 15개가 되어 TPR 1,
FPR 0이 된다. 그러나 Algorithm 1의 반환값은 0.612와 0.184로 참값 1에서 멀다. 여기에 overlap gate를 함께
걸면 남는 후보가 0개가 되어 no-return이다.

따라서 현재 확인된 실패는 세 층이다.

1. **선택.** 학습 중 critic으로 계산한 discrepancy가 반전되어 있다. 재학습하면 해결된다.
2. **overlap gate.** 구조적으로 유효한 후보를 전부 탈락시킨다. 원인은 제약 없는 처치확률 모형의 적합
   오차이며 이론의 가정 위반이 아니다.
3. **추정.** 선택이 완벽해도 후보별 추정치가 $-1.26$에서 $+0.29$까지 퍼져 있어 최종값이 참값에 닿지
   않는다.

7.9의 amendment 후보는 이 세 층에 맞추어 적는다.

### 7.5 2차 검토 반영: 설계 결함 확인과 설계 F 도입 (2026-09-11 3차)

두 번째 검토가 네 가지를 더 지적했다. 전부 확인했고 전부 맞았다.

**확인 1. 기존 A에는 정확한 balance가 불가능한 후보가 있다.** $S=\{1,2,3,4\}$를 감사용으로 남기면 표현
입력에 $(X,I_1,I_2,C)$만 남아 잠재 $U$의 측정값이 하나도 없다. 감사 대상은 $U$를 측정하므로 어떤 함수로도
정확한 balance를 만들 수 없다. 따라서 "block 0을 감사용으로 쓰지 않는 15개"와 "정확한 balance가 가능한
14개"는 다르다. `balance_feasible`을 추가해 이 구분을 코드에 넣었고, screen의 TPR/FPR은 이제 후자를 기준으로
계산한다.

**확인 2. 학습 문제는 아직 고쳐지지 않았다.** 재학습 audit은 기록만 되고 선택에는 쓰이지 않았으며, encoder
갱신 1회짜리 checkpoint도 적격이었고, 보고한 "갱신 중앙값 77"은 학습 종료 시점의 누적값이라 선택된
checkpoint의 학습량이 아니었다. 세 가지 모두 수정했다.

**확인 3. "선택만 고치면 된다"는 아직 성립하지 않는다.** Brier gap이 작다는 것과 인과 편향이 작다는 것은
다른 조건이며, 현재 수치는 두 항으로 분해되지 않았다.

**확인 4. 보고 수치에 오해의 소지가 있었다.** 7.4에서 "후보 추정치 $[-1.26,0.29]$, 출력 0.612"라고 적었는데,
$[-1.26,0.29]$는 정렬한 15개 중 **작은 쪽 여섯 개**였다. 중앙값 0.612와 모순이 아니다. 표현을 고쳤고,
수정 후 실행의 원시 결과를 `results/dgp_ae_pilot_postfix.json`으로 저장했다.

**추가 수정 F9--F14**

- **F9.** 선택 기준을 재학습 audit으로 바꿨다. 계획 8.5와 8.7의 학습 중 critic audit은 `d2_raw`,
  `discrepancy_pass`, `screen_pass_plan`으로 기록에 남는다. 이것은 protocol amendment다.
- **F10.** checkpoint 적격 조건에 encoder 갱신 하한 20회를 넣었다. 하한에 못 미치면 `epoch = -1`로 표시한다.
- **F11.** 갱신 횟수를 checkpoint 시점 값으로 기록하고, 학습 종료 시점 값은 `encoder_updates_at_end`로 따로
  둔다.
- **F12.** 처치확률 출력을 생성식의 구조적 범위 $[0.05,0.95]$로 사상한다. 참 처치확률이 모든 $Z$에서 그
  범위 안에 있으므로 이탈은 추정 오차다. 이 변경 뒤 overlap 통과는 후보가 옳다는 증거가 아니라 모형이
  범위를 지킨다는 진술일 뿐이며, 보고에서 그렇게 표기한다.
- **F13.** 수정 후 실행의 원시 결과를 프로젝트에 저장한다.
- **F14.** screen 성능을 `balance_feasible` 기준으로 채점한다.

### 7.6 설계 F: 정확한 표현이 존재함을 먼저 확인한 생성식

두 번째 검토가 제안한 설계를 구현했다. $U$는 잠재 교란, $I$는 관측된 처치 원인, $C$는 관측된 공통 원인이며
셋과 모든 잡음이 독립 표준정규다.

$$
X=U+2\epsilon_X,\qquad M_j=U+\epsilon_j\ (j=0,\ldots,4),
$$
$$
W_0=(I,C,M_0),\qquad W_j=(M_j,\ C+0.35\zeta_j,\ N_j)\ (j=1,\ldots,4),
$$
$$
A\sim\mathrm{Bernoulli}\{0.05+0.9\,\mathrm{expit}(U+3I+0.2X+0.3C)\},\qquad
Y(a)=a-4U+0.2X-0.5C+\epsilon_Y.
$$

**정확한 표현의 존재를 직접 확인했다.** 표현 입력에 남은 측정값이 $k$개일 때
$Z_r=(X,C,\overline M_k+rI)$의 balance 조건은 Gaussian 공분산 조건이고, 이를 풀면 $r^2-3r+1/k=0$이다.
$k=1,\ldots,5$에서 두 근 모두 잔여 조건부 공분산을 $10^{-16}$ 수준으로 0으로 만든다는 것을 수치로
확인했다. 검증용 계수는 생성식 점검에만 쓰고 학습기에는 주지 않는다.

**단계별 검증의 두 번째 행을 실행했다.** $n=24000$, replicate 3회, 학습한 처치확률과 결과회귀다.

| 조정 | 평균값 | 평균 절대오차 |
|---|---:|---:|
| 무조정 | $-0.774$ | 1.774 |
| $X$ | $-0.332$ | 1.332 |
| $(X,C)$ | $-0.268$ | 1.268 |
| raw $(X,W)$ | $+0.681$ | 0.319 |
| 검증용 정확한 표현, $S=\{1\}$ | $+1.031$ | 0.031 |
| 검증용 정확한 표현, $S=\{2,3\}$ | $+1.002$ | 0.026 |
| oracle $(X,U,C)$ | $+0.996$ | 0.022 |

즉 이 생성식은 Simpson reversal, $X$ 조정 실패, 강한 학습기로도 남는 raw 편향 0.32, 그리고 oracle 수준에
도달하는 정확한 표현을 동시에 갖는다. 추정 구현 자체는 정상이다. 남은 질문은 학습된 표현이 이 표현을
찾아내는가 하나다. 결과는 `results/dgp_f_validation.json`에 있다.

### 7.7 3차 검토 반영: 확률 학습 불일치와 판정 경로 수정 (2026-09-11 4차)

세 번째 검토가 다섯 가지를 지적했다. 전부 코드에서 확인했고 전부 맞았다.

| 지적 | 확인 내용 |
|---|---|
| F12의 처치확률 변환 불일치 | 학습과 검증은 제약 없는 로짓의 이진 손실을 쓰고 예측만 $0.05+0.9p$로 사상했다. 참값 0.20을 정확히 학습해도 0.23을 쓰게 된다 |
| F의 경계 사례 오판정 | `balance_feasible`이 `dgp` 인자를 받고도 쓰지 않아 F의 $S=\{1,2,3,4\}$를 불가능으로 판정했다. F는 $W_0$에 $M_0$가 있어 $k=1$에서도 정확한 표현이 존재하므로 15개가 맞다 |
| 검증 스크립트의 조용한 건너뜀 | 같은 오판정 때문에 $S=\{1,2,3,4\}$를 목록에 넣고도 건너뛰어 `exact_S1234`가 저장되지 않았다 |
| 최소 갱신 규칙 미전달 | `screen_pass`가 `epoch`이나 갱신 횟수를 보지 않았다 |
| 저장 결과의 지위 | `postfix` 파일에 `balance_feasible`, `refit_pass`, `refit_threshold`, `screen_pass_plan`이 없다. 또한 새 기준으로 다시 채점하면 불가능한 후보 96개 중 6개가 통과하며 전부 $S=\{1,2,3,4\}$다 |

**추가 수정 F15--F19**

- **F15.** 처치확률의 구조적 범위 사상을 모형 안으로 옮겼다. `bounded_prob`이 학습 손실, 검증 손실, 예측에
  동일하게 쓰이고, 이미지 비교군의 end-to-end 모형도 같은 함수를 쓴다. 회귀 검사에서 적합된 확률이 구간
  안에 있고 참값 대비 치우침이 $-0.0036$임을 확인한다.
- **F16.** `balance_feasible`을 설계별로 분리했다. A--E는 14개, F는 15개다.
- **F17.** 두 critic도 같은 bounded map을 쓴다. 두 critic이 추정하는 양이 처치확률이고 그 참값이 구간
  안에 있기 때문이다. 그 결과 restart 적격 판정은 이 설계에서 항상 참이 되어 규칙이 공허해지고, restart
  선택은 계획 8.6의 본래 기준인 최소 validation gap으로 돌아간다. 구조적 범위를 끄면 원래 의미가 복원된다.
- **F18.** `screen_pass`가 `epoch >= 0`과 갱신 횟수 하한을 함께 요구한다. `training_ok`를 별도 필드로도
  기록한다. overlap screen은 기록만 하고 판정에서 뺐다. 범위를 강제한 뒤에는 자동으로 통과하기 때문이며,
  `overlap_screen_binding` 필드에 그 사실을 남긴다.
- **F19.** 검증 스크립트가 필수 사례를 건너뛰면 예외로 중단한다.

**수정 후 설계 F 재검증 ($n=24000$, 3 replicate, 학습 nuisance)**

| 조정 | 평균값 | 평균 절대오차 |
|---|---:|---:|
| 무조정 | $-0.774$ | 1.774 |
| $X$ | $-0.331$ | 1.331 |
| $(X,C)$ | $-0.269$ | 1.269 |
| raw $(X,W)$ | $+0.700$ | 0.300 |
| 정확한 표현 $S=\{1\}$ | $+1.031$ | 0.031 |
| 정확한 표현 $S=\{2,3\}$ | $+1.004$ | 0.025 |
| 정확한 표현 $S=\{1,2,3,4\}$ | $+1.001$ | 0.051 |
| oracle | $+0.995$ | 0.021 |

빠졌던 경계 사례가 포함되었고 통과한다.

**수정 후 A cell 재실행 (`results/dgp_ae_pilot_v3.json`, $n=6000$, replicate 10000--10001)**

| 후보군 | 개수 | 재학습 audit gap | audit 통과 | 학습 하한 통과 | screen 통과 | 후보 추정치 평균 |
|---|---|---|---|---|---|---|
| balance-feasible | 14 | $-0.0015$ | 14 | 14 | 14 | $+0.564$ |
| infeasible | 16 | $+0.0796$ | 1 | 16 | 1 | $-0.185$ |

선택이 바로잡혔다. TPR 1.00, FPR 0.06이며 통과한 유일한 infeasible 후보는 $S=\{1,2,3,4\}$다. 두 replicate의
Algorithm 1 반환값은 0.385와 0.314이고 raw는 0.756과 0.784, oracle은 0.910과 0.928이다.

**남은 문제는 표현 학습이다.** feasible 후보의 재학습 audit gap이 $-0.0015$로 사실상 0인데 그 후보들의 추정치
평균은 $+0.564$다. 즉 경험적 balance가 작다는 사실이 인과 편향이 작다는 것을 뜻하지 않는다. 원고 Section 4가
discrepancy를 인과 오차로 옮길 때 안정성 계수와 학습 오차를 함께 요구하는 이유가 여기서 수치로 드러난다.
설계 F는 정확한 표현이 존재하고 그것을 주면 오차가 0.03이라는 것이 확인되어 있으므로, 다음 단계는 학습된
표현이 그 표현에 얼마나 가까워지는지를 재는 것이다.

**실행별 코드 상태**

| 결과 파일 | 코드 상태 | 지위 |
|---|---|---|
| `results/dgp_ae_pilot.json` | F1--F2까지 | 최초 pilot 9 replicate. 선택 절차가 반전되어 있던 상태의 기록 |
| `results/dgp_ae_pilot_f9f14_partial.json` | F3--F8까지 | 재학습 audit 값을 담고 있으나 새 필드가 없어 현재 선택 절차의 증거가 아니다 |
| `results/dgp_ae_pilot_v3.json` | F1--F19 | 독립 사본 trunk를 쓰던 판의 증거 |
| `results/dgp_ae_pilot_v4.json` | F1--F22 전체 | 현재 선택 절차와 nested 포함관계의 증거 |
| `results/dgp_f_validation.json` | F1--F19 전체 | 설계 F 검증, 필수 경계 사례 포함 |

### 7.8 4차 검토 반영: 집계 정정과 nested 포함관계 강제 (2026-09-11 5차)

네 번째 검토가 다섯 가지를 지적했다. 전부 확인했고 전부 맞았다.

| 지적 | 확인 |
|---|---|
| A 표가 한 rotation만 보여준다 | 7.7의 수치는 replicate 10000의 rotation 0이었다. 두 replicate와 세 rotation 전체는 feasible 84건에 audit gap $-0.001338$, 추정치 $+0.359046$, infeasible 96건에 $+0.082199$, $-0.178415$다 |
| encoder가 여전히 잘못된 신호를 최소화한다 | feasible 후보의 학습 중 validation gap이 $-0.0898$과 $-0.1033$인데 같은 checkpoint를 고정해 재학습하면 $-0.00175$와 $-0.00092$다 |
| 원인을 하나로 확정할 수 없다 | 관측 gap은 참 discrepancy와 두 critic의 초과오차 차이의 합이라 분해되지 않았다 |
| 범위 제한을 꺼도 overlap이 복원되지 않는다 | `screen_pass`가 overlap을 무조건 제외했다 |
| F15 검사가 판별력이 없다 | 검사 자료가 버그 경로와 같은 형태여서, 잘못된 사후 변환이 오차 0을 냈다 |

**추가 수정 F20--F22와 보조 수정**

- **F20.** augmented critic이 base critic의 trunk를 **공유**한다. 이전에는 같은 형태의 사본을 따로 두어 두
  critic이 갈라질 수 있었다. 공유하면 residual을 0으로 두었을 때 두 critic이 문자 그대로 같은 함수가 되고,
  augmented class가 base class를 포함한다는 계획 8.5의 요구가 구조로 보장된다. 재실행에서 **학습 split의
  gap이 180개 후보 기록 전부에서 음수가 아니다**(음수 비율 0.000).
- **F21.** 학습 split의 gap을 checkpoint 시점 값으로 함께 기록한다. 학습 gap과 재학습 gap의 불일치가
  결과 파일에서 바로 보인다.
- **F22.** 통과 판정을 `screen_decision` 함수 하나로 모았다. 범위 제한을 끄면 overlap 조건이 다시 필수가
  되고, 켜져 있으면 발동하지 않는다. 회귀 검사가 소스 문자열이 아니라 이 함수의 동작을 직접 확인한다.
- **요약기.** 후보 수준 집계를 replicate와 rotation 전체에 대해 자동 생성하고 집계 범위를 함께 출력한다.
  한 rotation만으로 표를 만드는 일이 다시 일어날 수 없다.
- **검사.** 처치확률 검사를 좁은 구조적 범위 $[0.3,0.7]$에서 수행한다. $[0.05,0.95]$에서는 잘못된 사후
  변환이 확률을 최대 0.05만 옮겨 적합 오차에 묻힌다. 좁은 범위에서는 최대 0.18을 옮기므로 판별된다.
  현재 검사는 상수 예측기(오차 0.094)와 이중 변환(0.064)을 올바른 경로(0.023)와 구분한다.

**F20 이후 A cell 재실행 (`results/dgp_ae_pilot_v4.json`, 2 replicate $\times$ 3 rotation)**

| 후보군 | 기록 수 | 학습 split gap | validation gap | 재학습 audit gap | 추정치 평균 | screen 통과율 |
|---|---:|---:|---:|---:|---:|---:|
| balance-feasible | 84 | $+0.109$ | $-0.096$ | $-0.00129$ | $+0.611$ | 1.000 |
| infeasible | 96 | $+0.157$ | $+0.026$ | $+0.0802$ | $+0.006$ | 0.062 |

Algorithm 1 반환값은 0.546과 0.621이다. 직전 판(`v3`)의 0.385와 0.314에서 올라갔고, 같은 실행의 raw는
0.756과 0.784, oracle은 0.910과 0.928이다. TPR 1.00, FPR 0.062는 그대로다.

**남은 불일치를 정확히 적는다.** 학습 split의 gap은 이제 항상 음수가 아니지만, validation split에서는 여전히
$-0.096$이다. residual branch가 학습 split에 과적합하여 독립 split에서 base critic보다 나빠진다는 뜻이다.
따라서 조기 종료에 쓰는 신호는 아직 재학습 gap과 다르다. 포함관계는 구조로 보장되었고, 일반화 격차는 남아
있다.

**원인 귀속을 좁히지 않는다.** 관측된 gap은

$$
\text{학습된 critic의 위험 차이}
=\text{참 discrepancy}
+\text{base critic의 초과오차}
-\text{augmented critic의 초과오차}
$$

이므로, 재학습 gap이 0에 가깝다는 사실만으로 참 discrepancy가 작다고 말할 수 없다. 또한 `balance_feasible`은
**좋은 표현이 존재하는 후보라는 분류**이고 학습된 표현이 좋다는 판정이 아니다. TPR 1.00은 그 분류와의
일치율이다. 설계 F의 15개는 명시적 표현으로 뒷받침되지만, A--E의 14개 전부에 대한 존재성은 현재 판정
함수만으로 증명되지 않았다.

현재 자료가 지지하는 문장은 하나다.

> 후보 분류와 screen의 일치율은 개선되었고 nested 포함관계는 구조로 보장되었다. 학습 중 critic의 일반화
> 격차, 학습된 표현의 잔여 편향, 최종 회귀와 확률 추정 오차는 아직 분리되지 않았다.

### 7.9 amendment 후보

아래 1번과 2번은 이미 코드에 적용했고 그 근거를 7.5와 7.7에 기록했다. 계획 8.5, 8.7, 8.8, 8.9의 변경이므로 protocol amendment 승인이 필요하다. 3번과 4번은 적용하지 않았다.

1. **audit을 재학습 critic으로 정의한다.** 계획 8.5와 8.7을 고쳐 표현을 고정한 뒤 두 critic을 수렴까지
   학습하고 그 gap을 audit 값으로 삼는다. 위 표가 이 변경의 효과를 보여 준다. 계획 8.6의 restart 선택
   기준도 학습 중 gap이 아니라 재학습 gap을 쓰도록 함께 바꿔야 한다.
2. **처치확률 모형을 제약한다.** 참 처치확률이 $[0.05,0.95]$ 안에 있으므로 출력을 그 범위로 제한하거나
   보정한다. 그러면 overlap screen이 모형 오차가 아니라 실제 겹침을 재게 된다. 계획 8.8과 8.9의 변경이다.
3. **추정 단계를 먼저 점검한다.** 표현을 고정한 상태에서 후보별 추정치의 rotation 간 분산과 nuisance
   적합 품질을 측정한다. 선택을 고쳐도 3층 문제가 남기 때문에 이 점검이 선행되어야 한다.
4. 위 세 가지를 적용한 뒤에도 남는 경우에만 두 screen의 논리곱과 순서를 재검토한다.

어느 선택지든 계획 10.2의 규정대로 날짜가 찍힌 amendment로 기록하고 confirmatory를 새 seed로 다시 시작해야
한다.

## 8. 아직 주장하지 않는 것

모니터링 에이전트는 아래가 보고에 등장하면 근거를 요구한다.

1. PROBE가 raw보다 낫다는 어떤 형태의 주장. 아직 pilot도 돌지 않았다.
2. 어떤 DGP의 성능 수치. smoke run의 값은 규모가 작아 무효다.
3. 계획 1.3의 네 성공 기준 중 하나라도 충족했다는 주장.
4. exact ORC가 neural ERM의 성공이나 exact balance 달성을 함의한다는 진술.
5. 유한 stability constant $\Gamma$에 관한 진술.
6. 과거 s9의 절대오차 0.055가 새 Brier critic 구현의 증거라는 진술.

## 9. 다음 단계

1. aggregation boundary 사전 계산 (계획 12.3).
2. pilot 실행: $n\in\{6000,12000\}$, replicate id 10000--10004, DGP A부터. 계획 10.2의 다섯 점검 항목을
   기록하고 no-return과 tie 비율이 20%를 넘는지 본다.
3. pilot 결과로 계산 예산을 산출하고 confirmatory 규모를 용한과 확정한다.
4. 미구현 항목을 5절의 권장 순서로 채운다.
