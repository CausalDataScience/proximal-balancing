# Single-image PROBE decisive pilot: corrected protocol v2

작성: 2026-10-09 CDT. Executor: single-proxy-balancing. Manager: Pao, read-only.
원 v1은 보존한다. 이 문서는 실행 전 승인된 correction과 local pilot의 고정값이다.

## Output Contract / PERR

- Deliverable: bounded $\sigma=0.3$, seed 20261009, $n=48{,}000$의 image-only local pilot. Five singleton splits의 fixed reader와 실제 alternating neural training, independent raw audit와 effect candidates를 보고한다.
- Consumer: 용한. Predictor quality와 balance가 동시에 가능한지 다음 실험을 결정한다.
- Form: Python code, regression tests, 이 protocol, 별도 JSON/log.
- Acceptance: new-$\phi$ critic refit 뒤 C 판단; P fit/internal validation 뒤 S score; frozen model 뒤 fresh reference evaluation; 같은 E의 candidates/controls/paired SE; source/roles/protected hashes; cap이면 partial 상태.
- Evidence: 실행 command, source SHA256, patient-role SHA256, epoch/history, test results, JSON/log.
- Boundaries: main local CPU만 쓴다. Original v1/protected artifacts/기존 JSON·log·weights를 보존한다. New dependency, remote action, Git mutation, memory write, 추가 seed/$\sigma$ grid는 제외한다.
- Necessity: 기존 teacher proxy와 signed effect criterion은 valid certification이 아니다. 이번 correction은 판정 대상과 finite bound를 실제 코드에 일치시킨다.
- Review: math-theory-agent는 read-only로 bound/code/results를 검토한다. 본 실행은 tests/smoke 뒤 manager code review pass를 받아 시작한다.

## 1. SCM과 input

환자당 feature는 image $W$ 한 장이다. $A$는 predictor training label이고 $Y$는 outcome estimation target이다.
Hidden severity $U$, region burdens $L_k=U+\sigma\epsilon_k$, device $I$, brightness $G$는 truth evaluation만 사용한다.
$U,G,\epsilon_k$는 standard normal이고 $\epsilon_k$는 iid다. $I$, outcome noise와 rendering noise는 이 변수들과 독립으로 생성한다. Treatment draw는 지정된 별도 uniform RNG stream을 쓴다.

$$
\pi(U,I)\triangleq P(A=1\mid U,I)=0.1+0.8\Phi(1.5U+I),\qquad
I\sim\operatorname{Bernoulli}(0.3),\qquad Y=A+U+\epsilon_Y .
$$

Legacy generator의 image와 $Y(0)$를 그대로 재사용한다. 새 지정 RNG stream에서 bounded $A$를 뽑고 $Y=Y(0)+A$로 바꾼다.
Legacy PhantomConfig/oracle/default는 바꾸지 않는다. New artifact는 link를 bounded로 명시한다.
Brightness correction은 모든 region 밖의 fixed body mask만 쓴다. 그 image-derived reference scalar는 $Z$에 남긴다.
Encoder의 learned dimension은 2이고 reference scalar를 포함한 total dimension은 3이다.
Held-out $T$는 correction 전 original pixels이다. Shared reference noise가 있으므로 ORC나 operator stability를 자동 주장하지 않는다.
$\sigma=0.3$의 true-$L$ severity $R^2$와 실제 reader/encoder의 information recovery는 별도다.

## 2. 고정 실행 값

- $n=48{,}000$, R 12,000; D/C/P/S/E 각 7,200; rotation 0.
- Primary $\delta^2=0.01$; sensitivity $0.005,0.02$도 모두 보고한다.
- Seed 20261009. 이 pilot 이후 primary 3 seeds의 최대 model family를 $M=45$, $\alpha=0.05$로 먼저 고정한다.
- Five fixed reader rank-2 encoders와 five MAE-initialized neural encoders를 실행한다.
- R-only MAE validation hidden-pixel MSE가 R mean-image baseline보다 나쁘면 five random-initialized neural encoders도 실행한다. S/E는 이 분기에 쓰지 않는다.
- Reader max 30 epochs, MAE 10 epochs, q0/reader q1 max 30 epochs, raw q1 max 60 epochs/patience 12.
- Neural max 30 rounds, B update 20 steps/round. 기존 architecture, fixed summaries, keep loss, dual rule, squared overlap hinge를 유지한다.
- Fresh reference sample 100,000, batch 1,000. Model을 고정한 뒤 별도 seeds로 생성한다.
- CPU 2 threads. Runtime cap 21,600 seconds. Peak RSS cap 12,288 MiB.
- Original v1, protected17, 기존 low-sigma followup JSON/log를 합친 20 files를 보존한다.

## 3. Training과 scoring 순서

R에서 image-only reader/MAE/summaries를 fit한다.
각 split은 retained pixels만 encode한다.
B update 뒤 새 $\phi$에서 두 predictors를 D에서 각각 proper Brier risk로 다시 fit한다.
그 뒤 D training diagnostic과 C risk를 계산하고 dual/stop/checkpoint를 정한다.
기존 1% propensity-tail gate는 이번 bounded pilot의 checkpoint 조건에서 제거한다.
Squared hinge는 range를 넘은 거리를 벌한다. Hard tail fraction을 강제하지 않는다.
C에서 선택한 encoder를 고정하고 P internal fit/validation으로 fresh predictors를 fit한다.
그 뒤 S에서만 audit을 한다. S/E는 epoch, encoder, initialization, threshold 선택에 쓰지 않는다.
Raw residual epoch 0이 선택되면 추가 raw 개선이 선택되지 않았다고 보고한다. Near-Bayes accuracy는 미확인으로 남긴다.

## 4. 실제 predictor의 valid conservative bound

$e(Z)\triangleq P(A=1\mid Z)$, $p(T,Z)\triangleq P(A=1\mid T,Z)$라고 정의한다.
$\varepsilon_b^2$는 fitted predictor와 해당 conditional probability의 mean squared difference이다.
Independent S의 unpenalized Brier gap은 다음 population identity를 따른다.

$$
g=D^2+\varepsilon_0^2-\varepsilon_1^2,\qquad
D^2\triangleq E\{p(T,Z)-e(Z)\}^2 .
$$

Renderer는 $L_{1:5},I,G$와 $A$에 독립인 rendering noise만 쓴다.
SCM의 독립 생성에 의해 $G$와 rendering noises는 $A$와 $(L_{1:5},I)$ 조건부로 독립이다.
Exact phantom reference $p_{LI}\triangleq P(A=1\mid L_{1:5},I)$는 Gaussian posterior를 적분해 계산한다.
Actual frozen raw $q_1$에 대해 fresh draws에서 계산한 error는 다음 upper이다.

$$
E(q_1-p_{LI})^2
=\varepsilon_1^2+E\{\operatorname{Var}(p_{LI}\mid T,Z)\}
\ge\varepsilon_1^2 .
$$

이는 exact $\varepsilon_1^2$가 아니다. Unobserved reference information 때문에 conservative하다.
Million-sample teacher의 heldout Brier/replicate agreement를 certified teacher error로 쓰지 않는다.
Latent $\pi$ 대비 error도 별도 diagnostic으로 보고한다. True-$L$ reference error를 learned-$Z$에 승계하지 않는다.

$V$는 ddof=1 sample variance, $L\triangleq\log(4M/\alpha)$다.
Maurer–Pontil empirical Bernstein Thm. 4를 각 fixed model의 S loss range $[-1,1]$와 fresh error range $[0,1]$에 rescale한다.

$$
r_S=\sqrt{\frac{2V_S L}{n_S}}+\frac{14L}{3(n_S-1)},\qquad
r_1=\sqrt{\frac{2V_1 L}{m}}+\frac{7L}{3(m-1)},\qquad
b_1=\overline{(q_1-p_{LI})^2}+r_1 .
$$

각 bound에 $\alpha/(2M)$를 배분한다. Primary 3 seeds의 최대 45 models를 함께 보호한다.
Full 3-$\sigma$ joint guarantee에는 $M=135$ 또는 추가 alpha allocation이 필요하다. 이번 pilot은 그 guarantee를 주장하지 않는다.
Finite discrepancy upper는 $\max(\widehat g,0)+r_S+b_1$이다.
Normal approximation과 Hoeffding upper도 구분해서 보고한다. 같은 upper의 세 cutoff 비교는 추가 multiplicity가 아니다.
Finite upper가 $\delta^2$를 넘으면 그대로 fail로 보고한다. Finite balance pass는 causal certificate와 구분한다.

## 5. Overlap과 effect candidates

Bounded true $\pi$의 conditional expectation도 $[0.1,0.9]$ 안이다. 이 SCM에서는 true conditional propensity의 uniform overlap를 확보한다.
Fitted predictors는 $(0,1)$ probability를 출력한다. Forced $[0.1,0.9]$ clamp로 tail pass를 만들지 않는다.
AIPW의 $[0.05,0.95]$ clipping은 numerical stabilization이다. Fitted tail, known-$U$ tail, latent-$UI$ tail을 구분한다.
Fitted tail 2%는 diagnostic flag이며 causal invalidity나 $Z$의 정보량을 판정하는 theorem이 아니다.

모든 encoder의 effect candidate를 gate와 독립으로 같은 E patients에서 계산한다.
Outcome nuisance는 P에서 fit한다. Full-image reader adjustment, true-$L$ evaluation control, known-$U$ propensity/outcome control, naive contrast를 함께 보고한다.
Paired score difference의 SE는 fixed fitted candidates의 sampling diagnostic이다.
Unmeasured confounding, outcome nuisance error, ORC, stability $\Gamma$는 sampling interval로 덮지 않는다.
Reference split-bias difference를 learned neural causal radius $\rho$로 쓰지 않는다.
Qualified causal plurality와 causal confidence interval은 해당 조건이 미확인인 동안 unresolved/None이다.

## 6. K1–K5의 수정된 의미

- K1: S5 additional-treatment signal을 matching bounded reference/true-I control와 비교한다. Bounded $\sigma=0.3$의 posterior reference $D^2_{S5}=0.00855518$은 primary 0.01 안이다. 따라서 S5의 primary pass를 자동 false pass로 부르지 않는다. 0.005 sensitivity도 그대로 보고한다.
- K2: Independent S/fresh reference finite discrepancy pass count를 보고한다. Single pilot은 3-seed criterion 완료가 아니다.
- K3: Candidate의 known $\tau=1$ error와 oracle difference/paired SE를 보고한다. Unqualified plurality point를 causal certificate로 부르지 않는다.
- K4: Signed neural-minus-fixed contrast의 방향으로 성능을 판정하지 않는다. Known-$\tau$ absolute error difference와 paired score sampling interval을 보고한다. Statistical noninferiority 및 3-seed decision은 pending이다.
- K5: New bounded+I true-$L$ posterior reference의 separate simulation sanity check를 수행한다. Learned reader/B와 old no-I pure-probit formula의 30% 일치를 요구하지 않는다.
- Failure 원인은 predictor/representation/threshold/bound/estimation을 구분해 기록한다. 한 criterion 실패를 하나의 원인으로 단정하지 않는다.

## 7. 실행과 보존

Command는 project code/에서 다음처럼 고정한다.

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -B -m single_image.run_decisive_v2 --n 48000 --sigma 0.3 --seed 20261009 --rounds 30 --encoder-steps 20 --reader-epochs 30 --mae-epochs 10 --critic-epochs 30 --raw-epochs 60 --mc 100000 --mc-batch 1000 --family-size 45 --alpha 0.05 --max-seconds 21600 --rss-limit-mib 12288 --device cpu --conditional-random

Output는 results/single_image/decisive_v2/decisive_sigma0.3_seed20261009.json과 같은 stem.log다.
JSON은 stage/model 완료마다 atomic checkpoint로 갱신한다. Timeout/RSS/source-change에서 partial state와 미완료 항목을 남긴다.
Original 결과를 덮어쓰지 않는다. Source freeze와 20 protected files의 SHA256를 실행 전후 확인한다.
Tests는 기존35개와 새 regression tests를 모두 실행한다. $n=1200$/1round smoke output은 /tmp만 쓴다.
이전 전체 run 41분 중 실제 neural loop는 약18분이었다. 이번 primary는 약3–4시간, random branch 포함은 약5–7시간으로 예상한다.
6시간 cap에서는 완료 항목을 보고하고 남은 항목을 완료라고 쓰지 않는다.
새 architecture, seed, threshold, truth input을 실행 중 바꾸지 않는다. Additional seeds/grid/DeltaAI는 이번 scope에 없다.

## 8. 실행 상태

아직 본 pilot을 실행하지 않았다. 기존35개와 새12개를 합친 tests47개는69.58초에 통과했다.
$n=1200$/1round smoke는65.73초, peak RSS1140.86MiB에15models를 끝까지 실행했다.
Smoke의 source/protocol/protected20 SHA는 모두 유지됐다. Smoke의 effect는 development 값이고 본 pilot을 대신하지 않는다.
Manager code review가 통과한 뒤 본 run을 시작한다.
본 run 시작 뒤 이 protocol은 변경하지 않는다. 실제 결과는 별도 JSON/log와 README에서 보고한다.
