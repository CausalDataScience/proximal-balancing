# single_image: 단일 이미지 PROBE

## Bounded decisive pilot v2

새 image-only 경로는 [corrected protocol v2](../../memo/2026-10-09-single-image-decisive-test-v2.md)의
`single_image.run_decisive_v2`다. Bounded link를 별도 module에 두어 아래 legacy generator와 결과를 보존한다.
Primary는 $\sigma=.3$, seed20261009, $n=48{,}000$이다. Five fixed/five MAE neural과 R-only 조건부 five random neural을 실행한다.
Actual frozen predictor와 exact phantom $p(L_{1:5},I)$의 fresh error를 사용한다. Empirical Bernstein과 Hoeffding의
finite discrepancy upper를 normal gap diagnostic과 구분한다. $M=45$, $\alpha=.05$, primary $\delta^2=.01$을 먼저 고정한다.
본 pilot은 source review와 smoke 뒤 실행을 마쳤다. Results는 별도 `results/single_image/decisive_v2/`에 남겼다.
완료한 stage/model의 평가·history·hash는 JSON에 보존한다. 6시간/12GiB cap이면 partial 상태를 보고한다.
ORC, $\Gamma$, actual plurality radius가 미확인이면 causal return은 None이다. Sampling interval은 causal certificate가 아니다.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -B -m single_image.run_decisive_v2 --n 48000 --sigma 0.3 --seed 20261009 --rounds 30 --encoder-steps 20 --reader-epochs 30 --mae-epochs 10 --critic-epochs 30 --raw-epochs 60 --mc 100000 --mc-batch 1000 --family-size 45 --alpha 0.05 --max-seconds 21600 --rss-limit-mib 12288 --device cpu --conditional-random
```

### 첫 bounded pilot의 실제 결과

2026-10-09 CDT에 $n=48{,}000$, $\sigma=.3$, seed20261009로 실행했다.
[JSON](../../results/single_image/decisive_v2/decisive_sigma0.3_seed20261009.json)과
[log](../../results/single_image/decisive_v2/decisive_sigma0.3_seed20261009.log)에 실제 command, versions,
patient-role hashes, source hashes, predictor epochs와 training history를 남겼다.
Runtime은 8,224.76초(137.08분), peak RSS는 2,338.95MiB였다. CPU 2threads를 사용했다.
$R$은 reader와 initialization을 정하는 12,000명이다. $D/C/P/S/E$는 각각 7,200명이다.
$D$는 encoder update, $C$는 checkpoint 선택, $P$는 fresh predictor와 outcome head fit,
$S$는 raw-image audit, $E$는 effect evaluation에 사용한다.
Encoder와 predictor 선택에는 $S/E$를 사용하지 않았다.

$U$는 hidden severity, $A$는 treatment, $Y$는 outcome이다. $L_{1:5}$는 image를 만드는 Gaussian measurements이고
$I$는 region5의 device signal이다. $Z$는 retained image의 encoder output이고 $T$는 held-out region의 raw image다.
$\widehat q_0$는 $Z$에서, $\widehat q_1$은 $(T,Z)$에서 treatment probability를 예측한다.

Fixed는 rank2 reader representation이다. MAE는 Masked Autoencoder로 초기화한 neural encoder다.
Five fixed와 five MAE model의 independent raw-image audit과 같은 $E$의 effect candidates를 계산했다.
Candidate는 gate와 관계없이 계산한 effect estimate다. 여기서는 known truth가 $\tau=1$이다.
아래 standard error(SE)는 fixed trained models에 대한 sampling uncertainty다. Causal confidence interval은 아니다.
Empirical Bernstein(EB) upper는 predictor error와 sampling uncertainty를 포함한 finite discrepancy upper다.

| Split | Fixed candidate (SE) | MAE candidate (SE) | MAE−fixed (paired SE) | Fixed EB upper | MAE EB upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 | 1.08906 (.03109) | 1.17121 (.03930) | .08214 (.02416) | .020423 | .021752 |
| S2 | 1.08564 (.03199) | 1.19205 (.03329) | .10641 (.01352) | .019515 | .023213 |
| S3 | 1.10008 (.03203) | 1.13903 (.03459) | .03896 (.01299) | .019363 | .020710 |
| S4 | 1.09286 (.03163) | 1.27661 (.03284) | .18375 (.01561) | .020338 | .033254 |
| S5 | 1.09191 (.03362) | 1.15278 (.03504) | .06087 (.01843) | .021157 | .024917 |

Normal gap upper는 predictor error를 포함하지 않은 approximation이다.
Primary $\delta^2=.01$에서 finite pass는 **0/10**이고 normal gap pass는 **9/10**이다.

| $\delta^2$ | Normal gap pass | Finite discrepancy pass |
| --- | ---: | ---: |
| .005 | 7/10 | 0/10 |
| .01 (primary) | 9/10 | 0/10 |
| .02 (sensitivity) | 10/10 | 2/10 |

$.02$에서 통과한 model은 **fixed S2/S3**다. Neural model은 0개다.
이는 finite discrepancy 판정이며 causal qualification이 아니다. Upper가 cutoff를 넘었다는 사실은
실제 population discrepancy가 cutoff를 넘었다는 증명이 아니다.

MAE S2는 P internal validation에서 reader/raw residual을 모두 epoch0으로 선택했다.
따라서 $\widehat q_1\approx\widehat q_0$이고 independent $S$ gap은 약0이다.
그러나 fresh reference의 $b_1=.017904$를 포함한 finite upper는 $.023213$이다.
작은 gap만으로 predictor quality와 balance를 동시에 확인했다고 말할 수 없다.
각 model을 고정한 뒤 fresh phantom 100,000명을 batch1,000씩 평가했다.
Reference $p_{LI}=P(A=1\mid L_{1:5},I)$와 비교한 error upper $b_1$은 exact Bayes approximation error가 아니다.
$b_1$에는 predictor approximation뿐 아니라 rendering과 representation에서 사라진 reference information의
uncertainty도 포함된다. 큰 $b_1$만으로 실제 approximation error나 $U$ information loss를 확정할 수 없다.
같은 fresh predictions에서 $\widehat q_0$의 reference error도 diagnostic으로 보고한다.

MAE training은 split별 30/10/15/10/15 rounds, 총80rounds와 1,600 encoder steps를 수행했다.
Selected rounds는 29/10/15/10/15다. S1은 30round cap에 도달했고, 다른 split은 C stop rule에서 멈췄다.
이는 mathematical convergence 또는 near-Bayes accuracy의 증명이 아니다.
Final P fit에서 MAE S1–S4의 추가 raw residual은 epoch0을 선택했고 S5는 epoch46을 선택했다.
추가 raw correction의 validation 이득이 없었던 경우와 predictor의 충분한 정확성은 구분한다.
R-only MAE hidden reconstruction MSE(mean-squared error)는 $.14908$로 mean-image baseline의 $.16827$보다 작았다.
따라서 사전 지정한 규칙대로 random initialization branch를 생략했다.
Reader features에서 $U$를 예측한 linear evaluation $R^2$는 $.93130$였다.
이는 neural $Z$의 $U$ 보존값이나 raw image의 optimal recovery를 측정한 값이 아니다.

같은 $E$의 full-image reader adjustment는 $1.05415$ (SE $.03599$), true-$L$ control은
$1.01970$ (SE $.02969$), known-$U$ oracle control은 $.99840$ (SE $.02988$), naive는
$2.01545$ (SE $.03163$)였다. Truth inputs는 명시한 evaluation controls에서만 썼다.
True-$L$ control은 matching posterior-adjustment benchmark다. 모든 image-based method의 information
ceiling이나 모든 encoder의 effect-bias lower bound가 아니다.
별도 matching bounded+I posterior reference의 population-adjustment target은 약 $1.0281$–$1.0286$이다.
그 target과 clipped finite fitted AIPW candidate는 같은 수치로 강제하지 않는다.

이번 pilot에서 MAE의 point absolute error는 five splits 모두 fixed보다 컸다.
JSON의 paired normal intervals는 fixed-score sampling diagnostic이며 three-seed noninferiority 판정이 아니다.
원인은 아직 분리하지 못했다. Representation target의 차이, propensity/outcome approximation과 sampling이
함께 작용할 수 있다. Current outcome head는 P에서 fit한 affine ridge regression이다.
Nonlinear $Z$에 대한 conditional outcome mean이 affine이라는 보장은 없다.
따라서 이 결과만으로 $U$ information loss 또는 outcome head가 실제 원인이라고 단정하지 않는다.

K1은 matching S5 posterior reference $D^2\approx.008474$가 primary $.01$ 안임을 확인하는 diagnostic으로
해석한다. S5의 primary pass를 자동 false pass로 부르지 않는다. K2의 primary finite pass는0개다.
K3/K4의 qualified plurality와 three-seed decision은 pending이다. K5의 별도 true-$L$ simulation sanity는
five splits 모두 5SE criterion을 통과했다. ORC와 $\Gamma$가 확인되지 않아 causal return과
qualified plurality는 `None`이다. Real CT validation과 추가 seeds/$\sigma$ grid는 실행하지 않았다.

Regression tests는 기존35개와 새12개, 총 **47개 통과**했다. $n=1200$/1round smoke도 완료했다.
본 run 종료 뒤 source24개, protected20개와 frozen protocol의 SHA256가 모두 실행 전과 일치했다.
Frozen protocol, historical JSON/log/checkpoints와 본 JSON/log는 결과 문서 작성 때 변경하지 않는다.

## Previous local pipeline

`run_pipeline_v2`는 환자당 이미지 한 장을 쓰는 이전 pure-probit Local 실행 경로다. Raw/corrected 각각 다섯
split의 fresh raw-image audit, corrected S1의 encoder–predictor 교대학습, 독립 audit와 효과 대조,
$\sigma=.3$ control을 연결한다. $X$ feature와 처치·결과 식의 $X$ 원인은 함께 제거한다.

```bash
cd research/papers/single_proxy_balancing/code
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -B -m single_image.run_pipeline_v2 --n 24000 --seed 20261009 --rounds 30 --encoder-steps 20 --device cpu
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -B -m pytest single_image/tests -q -p no:cacheprovider --basetemp=/tmp/probe-v2-tests
```

결과는 `results/single_image/pipeline_v2_local.json`과 `.log`다. 같은 이름이 있으면 학습 전에 중단한다.
기존 phase1 JSON/checkpoint, sealed source와 v1은 읽기만 한다. 새 모델 가중치는 저장하지 않는다.
재현에는 기록한 명령, source hash와 같은 device를 사용한다. 다른 결과 경로는 `--out-json`, `--log`로
지정한다. 새 출력의 부모 디렉토리는 이미 있어야 한다.

R은 이미지 전용 준비다. D 내부 validation은 predictor epoch를 고른다. Encoder 갱신 뒤 새 표현에서
두 predictor를 다시 적합하고, 그 뒤 C에서 dual과 checkpoint를 판정한다. 선택된 encoder는 고정한다.
P 내부 validation으로 새 predictor를 적합하고 S에서만 검사한다. E는 효과 평가에만 쓴다.
Fixed body reference는 모든 폐엽 밖에 있다. Raw/corrected 모두 그 scalar를 $Z$에 보존한다.
원본 held-out 픽셀은 raw $\widehat q_1$에 남긴다. 보정 때문에 채널 독립이나 causal completeness가 자동 성립하지 않는다.

Primary $\delta^2=.01$을 평가 전에 고정한다. .005/.02도 같은 upper로 함께 보고한다. Normal SE upper는
경험적 진단이다. Fresh phantom에서 실제 frozen $\widehat q_1$을 latent propensity와 비교한 보수적 error upper와
independent S의 finite risk upper를 별도로 계산한다. 이 error는 정확한 Bayes approximation error가 아니다.
고정된 $M=12$ family와 $m=5{,}000$에서는 sampling radius 자체가 .02보다 클 수 있다. 이를 predictor 실패와 구분한다.
$\widehat q_0$의 fresh latent-propensity error, calibration와 overlap tails도 보고한다.
Bounded head로 overlap을 강제하지 않는다. Tail 통과, propensity clipping, 학습의 round cap은 각각
uniform overlap, 인과 식별, 학습 수렴의 인증을 대신하지 않는다. 인증이 없으면 returned estimate는 None이다.
E의 후보 AIPW, full-reader/true-$L$/oracle-$U$/naive 값은 같은 E 환자의 diagnostic이다.

## Local v2 결과 (2026-10-09)

위 명령은 실제로 CPU에서 41.05분 실행했다. 새 seed의 환자 $n=24{,}000$, R6000과 D/C/P/S/E각3600이다.
Low-$\sigma$ control은 별도 seed의 $n=6{,}000$이다. 기존 보호15개 파일의 hash는 유지했다.

1. Raw/corrected 다섯 split의 normal gap-only 판정은 $\delta^2=.005$에서1개, $.01$에서4개, $.02$에서5개 통과다.
   S5 raw gap은 .012963/.012230이다. 참 $I$ control의 .015023/.014389와 구분한다.
   모든 split의 $\widehat q_0$ tail은 사전1% 기준을 넘었다. Gap-only 통과는 causal qualification이 아니다.
2. Reader-feature 선형 평가에서 $U$의 $R^2$는 raw.7921→corrected.8221, $G$는 .9975→−.0065였다.
   이 수치를 neural $Z$의 $U$ 보존값으로 옮기지 않는다. Body reference는 $Z$에 남긴다.
3. S1은 실제30round/600step을 수행했다. Ckeep1.0467→.5310은 요약 복원 목적 개선이다.
   C 적격 checkpoint가 없어 round30 diagnostic fallback을 골랐다. Cap은 convergence가 아니다.
4. Fresh S의 neural $\widehat q_0/\widehat q_1$ Brier는 .166202/.164165, gap=.002037, normal upper=.005329,
   $\widehat q_0$ tail6.81%다. P-validation은 $\widehat q_0$ epoch29, reader $\widehat q_1$ epoch9, raw residual epoch0을 골랐다.
   최종 raw branch는 reader prediction을 유지했다. Raw/Bayes predictor quality는 미인증이다.
   Fixed-reader S1의 $\widehat q_0/\widehat q_1$ Brier .161266/.160183 대비 독립 prediction이 개선된 결과는 아니다.
5. Fresh latent-propensity error는 $\widehat q_0$ .041133/$\widehat q_1$ .036454다. 정확한 approximation error가 아니다.
   Finite upper=.121904이고 모든12모델이 세 cutoff의 finite gate를 실패했다. S radius.058565와
   MC radius.024847 자체가 큰 보수적 검사다. 이 실패로 실제 $D^2$가 크다고 결론 내리지 않는다.
6. 같은 E의 후보는 fixed-reader1.44095(SE.04591), neural1.47219(SE.05083), fullreader1.38642,
   true-$L$1.27751, known-$U$1.00598, naive2.38051이다. 참 효과는 $\tau=1$이다. 효과 오차 감소는 확인하지 못했다.
   후보값을 accepted ATE로 반환하지 않는다. Linear outcome fitting/propensity clipping/latent residual을
   따로 구분하며, 참 $L$ 값을 모든 estimator의 정보 천장으로 부르지 않는다.

Known-$U$는 P-fitting 없이 E에서 known nuisance로 계산했다. E의 tail25.69%와 population reference23.835%는
다른 값이다. 이 DGP에서는 정확한 severity propensity도1%tail gate와 충돌할 수 있다. Hinge는 구간 밖
거리만 벌하므로 고정 $\mu=10$이1%tail을 강제하지 않는다. 이를 실제 gradient 실패의 단독 원인으로 단정하지 않는다.
Corrected MAE validation MSE.2028은 mean-image.1835보다 못했다. $\sigma=.3$에서도 .5468>.1589였다.
Low-$\sigma$ reader-feature $U$의 $R^2=.91765$는 true-$L$ population reference.98232와 구분한다. Low-$\sigma$ $\widehat q_0$ tail17.11%로
좋은 정보가 overlap gate를 자동 통과시키지는 않았다. Low-$\sigma$는 neural loop를 실행한 결과가 아니다.

실행 source는 결과 provenance에 보존했다. 사후 known-oracle metadata만 수정했고 수치4,633개는 보존했다.
Current corrected source hash도 별도 기록한다. 본 실행 전29개, 사후 관련8개, 최종 전체30개 테스트가 통과했다(최종72.20초).
다섯 neural loop, role 전체 rotation/plurality, 반복 grid, nonlinear/pixel-reconstruction 비교와 실제 CT 검증은
아직 실행하지 않았다. 상세 수치와 가정 범위는 v2 문서 §17을 따른다.

## Low-$\sigma$ 효과 후보 보완 (2026-10-09)

이전 low-$\sigma$ 결과에는 효과 계산 호출이 없었다. Gate가 후보값을 지운 것이 아니었다.
`run_low_sigma_effects`는 기존 $n=6{,}000$, seed20271010, $\sigma=.3$의 image-only 모형을 그대로 재현한다.
기존 $Z$, $\widehat q_0$, $\widehat q_1$, S 환자 hash와 scoring risk가 일치한 뒤에만 효과 후보를 계산한다.
기존 S900은 development reproduction이다. P에서 nuisance를 적합하고, 이전에 효과 평가에 쓰지 않은
E900에서 아래 값을 계산한다. Gate와 candidate 계산은 독립이다. Low-$\sigma$ neural loop는 실행하지 않았다.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -B -m single_image.run_low_sigma_effects --n 6000 --seed 20271010 --reader-epochs 30 --mae-epochs 10 --critic-epochs 30 --raw-epochs 60 --mc 5000 --out-json ../results/single_image/pipeline_v2_low_sigma_effect_followup.json --log ../results/single_image/pipeline_v2_low_sigma_effect_followup.log
```

| 같은 E900의 방법 | Candidate | Sampling SE |
|---|---:|---:|
| 고정 표현 S1 | 1.23372 | .11053 |
| Full-image reader 보정 | 1.01477 | .10184 |
| 참 $L$ 보정, 평가 전용 | .95677 | .08944 |
| Known $U$ propensity/outcome, 평가 전용 | .90418 | .08969 |
| 단순 비교 | 2.29617 | .08535 |

참 효과는 $\tau=1$이다. Full-reader의 point estimate가 더 가깝지만 paired 차이의 interval은 계산하지 않았다.
통계적 우월함이나 인과 인증으로 해석하지 않는다. 고정 S1의 sampling interval은 [1.01709, 1.45035]다.
이는 fitted 모델에 대한 sampling 진단이며 causal confidence interval이 아니다. S tail17.11%, E tail15.44%,
finite upper.166385로 accepted return은 None이다. 조건이 다른 pure-probit posterior reference1.06486에 맞추지 않는다.

실제 CPU 실행은109.69초(1.83분), peak RSS1079.13MiB였다. 기존 low scoring의 수치 차이는 모두0이었다.
기존 보호15개와 원 pipeline JSON/log의 총17개 hash를 보존했다. 신규5개 테스트 및 전체35개 테스트가 통과했다
(각2.76초/65.16초). 결과는 `results/single_image/pipeline_v2_low_sigma_effect_followup.json`과 `.log`다.
새 DGP, $n=96{,}000$, threshold, finite bound 방식이나 neural 재현 대조는 추가하지 않았다. 상세 정정은 v2 §18이다.

## 보존된 phase1 경로

Implements phase 1 of `memo/2026-10-08-single-image-probe-plan-v1.md`: section 6 (stage 1, fold R only),
section 8.1 (level A baseline) and sections 10.1-10.2 (Gaussian phantom and its oracle quantities).
이하 기존 phase1 경로에는 Stage2 교대학습이 없다. 새 경로와 결과를 구분한다. Fixed-encoder 진단은
full stage-3 ladder나 accepted PROBE estimate를 제공하지 않는다. The sealed project modules (`family_v2_probe.py`,
`hidden_proxy_recipe.py`, `run_twins.py`, `noise_scaled.py`, ...) are imported, never modified; their
SHA-256 is stored in every result file.

## Run

```bash
cd research/papers/single_proxy_balancing/code
export PYTHONDONTWRITEBYTECODE=1
nice -n 10 python3 -B -m single_image.run_phase1 --n 24000 --sigma 0.85 --seed 0 --image-only
nice -n 10 python3 -B -m single_image.run_phase1 --n 4000  --sigma 0.85 --seed 0 --image-only --quick
nice -n 10 python3 -B -m pytest single_image/tests -p no:cacheprovider --basetemp=<scratch dir>
```

Image-only 작업에는 `--image-only`가 필수다. 이 옵션은 predictor의 X input과 처치·결과 식의 X 계수를
함께 제거한다. 이미지의 random draws는 유지한다. 기존 X 실험 재현은 이 옵션을 생략한 legacy mode다.
두 모드는 결과 이름과 `feature_mode`로 구분한다. 기존 JSON이 있으면 학습·checkpoint 저장 전에 중단한다.

Options: `--variant nonlinear` (section 10.3), `--device cpu|mps|auto`, `--quick` (short training, tests
only), `--no-oracle-device`, `--overwrite`.  Threads are fixed at 2 (`OMP_NUM_THREADS`, `torch.set_num_threads`);
the process lowers its own priority (`os.nice(10)`).  Everything is deterministic given the seed on a fixed
device (checked for CPU and MPS; CPU and MPS differ from each other in the last digits).

## 수정된 진단 경로

```bash
cd research/papers/single_proxy_balancing/code
PYTHONDONTWRITEBYTECODE=1 python3 -B -m single_image.run_phase1 --diagnostic-only --device cpu
```

이 경로는 항상 image-only mode다. 결과는 `results/single_image/phase1/phase1_diagnostic_v2.json` 하나다.
기존 JSON과 reader checkpoint는 읽기만 한다. 새 checkpoint를 저장하지 않는다.

1. 기존 결과를 `.005`, `.01`, `.02`에서 다시 판정한다. Pooled rule과 네 rotation 모두 통과하는
   rule을 구분한다. 실제 reader, 참 L, 참 L+I positive control을 구분한다. 이 재판정은 legacy X 실험이다.
2. 새 image-only 데이터에서 기존 rank-2 encoder를 D 환자에 한 번 적합하고 고정한다. 영역 5를 보류한다.
3. 같은 Z, 같은 환자, 같은 q0에서 세 q1을 비교한다. Reader q1을 먼저 적합한다. Raw image와
   참 I control은 reader q1의 logit에 residual을 추가한다. Raw branch는 CNN과 고정 high-pass 특징을
   함께 받는다. 이 특징은 mask-normalized blur를 빼고 경계 3px를 제외한 제곱평균·고정 quantile이다.
   A/Y/U/I로 특징을 고르지 않는다. Residual이 0이면 앞선 predictor와 같다.
4. Predictor 적합 half의 내부 validation으로 epoch를 고른다. 반대 half는 최종 scoring에만 사용한다.
   두 half의 역할을 바꿔 반복한다. 새 확률은 sigmoid를 사용한다. `[.1,.9]`를 강제하지 않는다.
5. Brier risk, risk gap, predictor 간 차이, calibration, overlap 구간 밖 비율을 보고한다.
   이 관측 진단만으로 predictor가 Bayes optimum에 가깝거나 true discrepancy가 cutoff 이하라고 인증하지 않는다.
6. 참 L의 posterior mean을 쓴 별도 reference에서만 oracle predictor error를 계산한다.
   그 D2는 학습한 encoder의 값이나 모든 encoder의 noise floor가 아니다. Severity R2 ceiling은
   U 예측의 상한이다. 모든 causal effect estimator의 상한이 아니다. Population adjustment target과
   유한 표본 fitted AIPW 추정값도 따로 보고한다.

새 q0/q1 개선은 fixed-encoder 진단에 적용된다. 기존 PROBE 학습기는 sealed bounded-critic baseline이다.
그 baseline의 overlap flag는 link의 범위 때문에 자동 통과한다. 이를 overlap 인증으로 표시하지 않는다.
첫 raw CNN은 두 fold에서 validation을 개선하지 못했다. 고정 high-pass 특징을 추가하고 raw 학습만
최대 60 epoch, patience 12로 보완한다. 같은 scoring 환자를 다시 쓰므로 결과는 adaptive development pilot이다.
Checkpoint 선택에는 scoring을 쓰지 않는다. Raw branch가 validation을 개선하지 못하면 `unresolved`로 남긴다.
참 I는 evaluation positive control에만 쓴다.
이 phase1 진단 수정은 audit와 보고 오류를 고친다. 이 경로 자체는 alternating encoder나 causal effect 회복을 검증하지 않았다.

### 로컬 검증 결과

`phase1_diagnostic_v2.json`: image-only, n=24000, sigma=.85, seed=0, S5/rotation0,
scoring 4000명, cutoff=.01. Q0 Brier risk는 .160057이다.

| Q1 input | Scoring Brier risk | Risk gap | Empirical upper | .01 balance 진단 |
|---|---:|---:|---:|---|
| 압축 reader | .155141 | .004916 | .007120 | 통과 |
| Raw image + 고정 texture + reader | .149420 | .010637 | .014076 | 탈락 |
| 참 I positive control + reader | .146149 | .013908 | .017947 | 탈락 |

Raw branch의 내부 validation risk는 .166363에서 .163788, .161151에서 .149813으로 감소했다.
선택 epoch는 각각 2와 4다. Z/q0/reader/참 I predictor hash는 첫 pilot과 같았다.
Q0 확률의 4.65%는 [.05,.95] 밖이었다. Overlap을 인증하지 않는다.
참 L의 finite fitted AIPW는 1.247723, population adjustment target은 1.294086으로 구분한다.
이 값은 bias 해결이나 모든 image-only estimator의 정보 한계를 뜻하지 않는다.

## Files

| file | content |
|---|---|
| `regions.py` | fixed 64 x 64 lobe masks (1 RUL, 2 RML, 3 RLL, 4 LUL, 5 LLL), adjacency {1-2, 2-3, 4-5}, crops, `holdout_view` |
| `phantom.py` | generator of plan 10.1 (and the 10.3 variant behind `variant="nonlinear"`), eval-only truths |
| `summaries.py` | fixed chi_l, 16 per region; thresholds and scale fitted on R, then frozen |
| `reader.py` | region readers psi_k, sufficiency check, tetrad / one-factor diagnostic, eval-only R^2 |
| `mae.py` | convolutional masked autoencoder (8 x 8 patches, ratio 0.75) on R; `load_trunk` for stage 2 |
| `nets.py` | device and seeding, `ConvTrunk` (shared by the MAE and phi_S), `RegionEncoder` (phi_S) |
| `stage1.py` | `run_stage1(images_R, ...)`: summaries, readers, sufficiency, factor diagnostic, MAE |
| `folds.py` | R (25%) and the rest; `six_groups` / `roles` for D, C, P, S, E in stages 2-4 |
| `oracle.py` | population D^2 and tau_Z bias per split (10^6 Monte Carlo draws), comparator targets, section 3.6 closed forms |
| `level_a.py` | PROBE unchanged on reader blocks; comparators |
| `run_phase1.py` | CLI; writes `results/single_image/phase1/phase1_<variant>_n<n>_sigma<s>_seed<seed>.json` and weights |
| `tests/` | pytest suite |

## Design choices

* Masks: two ellipses (right lung on image left, radiological convention) cut by straight fissures; each
  lobe has 224-363 pixels.  Adjacency is a shared pixel edge, recorded in `ADJACENCY` and checked
  against the masks by a test.  N(k) has at least two regions for every k.
* Rendering: per lobe, a Poisson number of Gaussian blobs, mean 5 exp(0.45 v), opacity
  1.5 sigmoid(v) exp(0.35 xi), random sizes and positions, accumulated as 0.6 (1 - exp(-sum)) inside the
  lobe only (v = L_k, or tanh L_k in the nonlinear variant).  Count, opacity and coverage all rise with
  L_k, while positions, sizes, jitter, overlap and noise keep it from being read exactly (hand-made
  intensity features reach R^2 about 0.82 for L_k).  The device is a thin bright segment through a random
  point of lobe 5, multiplied by the lobe-5 mask.  Rendering is a batched separable-Gaussian matrix product:
  24,000 images take about 5 s.
* Each random component has its own generator from one SeedSequence, so turning one off (for instance
  the device) leaves every other draw unchanged.
* Readers see only their region (bounding-box crop, other pixels 0, plus the region mask) and are trained
  on fold R with images only.  `run_stage1`, `train_readers`, `train_mae` and `FixedSummaries.fit` take an
  image array and nothing else; a test spies on the runner to confirm only fold-R images reach stage 1.
* The g_k baseline of the sufficiency check is 512 fixed random convolution features (no pre-trained
  weights can be downloaded).  The region's own 16 summaries are reported as a second reference.
* Level A: `family_v2_probe.run_dataset(view, hidden_proxy_recipe.level(d_x, NAME), run_twins.THRESHOLD,
  seed)` then `noise_scaled.decide(...)`, exactly as `run_ihdp_acic.py` / `rhc_bootstrap.py`.  The level is
  the IHDP/ACIC/Twins one (rank-2 B, as section 8.1 asks); the threshold 0.0015 is SCM-1's sealed value and
  only matters for the sealed rule; seed = 20261008 + 1000 seed + `run_twins.FOLD_SEED_OFFSET`.  PROBE uses
  its own four folds inside the non-R rows.  Also run on the true burdens L_k ("oracle reader") and on L_k
  with I appended to block 5 (a positive control for the device screen).
* Comparators: `run_twins.baselines` (the sealed AIPW on the sealed rotations): naive, X, X with all
  blocks, X with U.

## Interfaces for stages 2 and 3

* `folds.six_groups(n, seed)` and `folds.roles(groups, rotation)`: R plus D, C, P, S, E.
* `regions.holdout_view(images, S)`: V_S in the two-channel format the MAE trunk was trained on;
  standardise channel 0 first with the stats returned by `mae.load_trunk`.
* `mae.load_trunk(path)` -> (`ConvTrunk`, stats); `nets.RegionEncoder(trunk, d_out=2)` is phi_S.
* `summaries.FixedSummaries.transform(images)` -> (n, 5, 16): the 80 targets of L_keep.
* `reader.ReaderBank.load(path).transform(images)`: psi_k for ladder levels 1-2.
* `oracle.population(cfg)`: D^2 and ordinary-adjustment bias of the fixed true-L posterior-mean reference;
  not the learned representation's D^2, not a global encoder noise floor.
* `run_phase1.stage2_cost`: micro-benchmark used for the stage-2 cost estimate.
