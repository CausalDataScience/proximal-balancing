# SCM-1부터 SCM-5까지의 성적표

작성일: 2026-09-13
근거 파일: `results/probe_structured_scm_brier_g_confirm_summary_v1.json`,
`results/probe_structured_scm_population_checks_v2.json`, `results/probe_e8_population_v2.json`,
`results/probe_e9_hard4_component_dev_v1.json`, `results/probe_e9_five_gate_verdict_v2.json`
정의: `code/probe_structured_scm.py`의 `SCM_FAMILY`, `memo/2026-09-12-scm-family-and-experiment-plan.md` 1절

이 문서는 다섯 SCM에 대해 지금까지 저장된 모든 결과를 한자리에 모은 것이다. 새 실험은 없다.

---

## 0. 먼저 용어

$U$는 관측되지 않는 교란변수, $I$와 $C$는 proxy 블록 안에 관측되는 변수, $A$는 처치, $Y$는 결과,
$X$는 관측 공변량이다. 다섯 SCM은 같은 인과구조를 공유하고 한 축씩만 다르다. 참 평균 처치효과는
다섯 모두 $1$이므로, 아래에서 "오차"는 $\lvert\hat\tau - 1\rvert$을 뜻한다.

비교군은 셋이다. $X$만으로 조정한 것, $(X,W)$를 전부 넣어 조정한 raw, 그리고 관측 불가능한 $U$를
실제로 쓴 oracle이다. oracle은 도달 가능한 상한 기준일 뿐 현실에서 쓸 수 없다.

---

## 1. 한 줄 요약

**현재 확증 대상 다섯 설정은 모두 통과한다.** 이는 $m=T=30$ 후보를 전수 평가한 제한 Brier
critic/encoder와 rotate4 operational crossfit의 결과다. 모집단 계산에서는 균형을 완벽히 이루면서도
잘못된 값을 겨냥하는 후보가 존재했고, 실제 확증에서 이 후보가 screen을 통과했지만 Algorithm 1의
최대 연결성분에 들어간 경우는 100회 중 0회였다.

따라서 확인된 것은 **Algorithm 1의 graph–largest-component–median 핵심이 이 고정 실험군에서 실제로
작동했다**는 것이다. 원고의 literal one-shot Algorithm 1 정리나 임의의 $m$, neural/general encoder까지
확인한 것은 아니다.

---

## 2. SCM-1~5 확증 성적표

$n=6000$, 설정별 독립 base seed 20개, 역할 회전 4회다. 다섯 설정은 같은 20개 seed base를 쓰므로
설정 사이를 합쳐 100개의 독립 표본처럼 해석하지 않는다. 오차 열은 평균
$\lvert\widehat\tau-1\rvert$이다.

| SCM | 바뀌는 축 | naive 평균 | PROBE MAE | $X$ MAE | raw MAE | oracle MAE | PROBE p90 | 단일값 반환 | 판정 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SCM-1 | 기준 기본형 | −0.1709 | 0.0280 | 0.9514 | 0.1854 | 0.0052 | 0.0541 | 20/20 | **PASS** |
| SCM-2 | 결과 잡음이 $U$에 의존 | −0.1723 | 0.0292 | 0.9526 | 0.1843 | 0.0094 | 0.0597 | 20/20 | **PASS** |
| SCM-3 | 처치효과가 $U$에 의존 | −0.1963 | 0.0294 | 0.9688 | 0.1892 | 0.0062 | 0.0577 | 20/20 | **PASS** |
| SCM-4 | 블록별 측정 잡음이 다름 | −0.1709 | 0.0298 | 0.9514 | 0.2001 | 0.0052 | 0.0563 | 20/20 | **PASS** |
| SCM-5 | proxy 관측이 비선형 | −0.1709 | 0.0285 | 0.9514 | 0.1859 | 0.0052 | 0.0504 | 20/20 | **PASS** |

짝지은 95% 상한이다. 음수면 PROBE가 이긴다는 뜻이다.

| SCM | $X$ 대비 상한 | raw 대비 상한 | oracle 초과 상한 |
| --- | ---: | ---: | ---: |
| SCM-1 | −0.8907 | −0.1386 | 0.0332 |
| SCM-2 | −0.8903 | −0.1343 | 0.0308 |
| SCM-3 | −0.9056 | −0.1407 | 0.0342 |
| SCM-4 | −0.8887 | −0.1490 | 0.0355 |
| SCM-5 | −0.8905 | −0.1374 | 0.0333 |

읽는 법이다. **naive 평균이 다섯 모두 음수인데 참 효과는 $+1$이다.** 표본에서도 Simpson reversal이
20/20회 발생했다. PROBE의 $X$/raw 대비 paired 절대오차 상한은 모두 $0$ 아래이고, oracle 초과 오차의
상한은 $0.031$에서 $0.036$으로 사전 허용치 $0.05$ 안이다. 이 구간은 설정별 20개 base seed의
paired $t$ 구간이며 다중비교를 보정하지 않았으므로, 이 실험의 frozen empirical gate 판정으로만 읽는다.

**"oracle에 가깝다"의 정확한 뜻**을 적어 둔다. 허용 오차 기준을 만족한다는 뜻이고, 통계적 동률이
아니다. oracle이 여전히 더 정확하다.

---

## 3. 모집단 구조, 표본 없이 정확히 계산

처치 링크가 probit이고 모든 변수가 아홉 개 독립 표준정규의 선형결합이므로
$\mathbb E[\Phi(V)\mid W]=\Phi(m/\sqrt{1+s^2})$가 정확히 성립한다. Stein 항등식으로 $\tau_Z$까지
닫힌 형태가 된다. E8은 $W_0$가 complement에 있어 $I,C$를 쓸 수 있는 15개 후보의 모집단 값을
표본 없이 계산했고, 별도 population checker는 일곱 clean 후보와 oracle의 값이 $1$임을 재검산했다.

| SCM | 유효 후보 | 눈먼 후보의 $\tau_Z$ | 그 편향 | 균형 불가 후보 |
| --- | ---: | ---: | ---: | ---: |
| SCM-1 | 7 | −0.44320 | 1.4432 | 7 |
| SCM-2 | 7 | −0.44320 | 1.4432 | 7 |
| SCM-3 | 7 | −0.47206 | 1.4721 | 7 |
| SCM-4 | 7 | −0.44228 | 1.4423 | 7 |
| SCM-5 | 7 | −0.44320 | 1.4432 | 7 |

**"눈먼 후보"란 오염 블록 $W_4$ 하나만 held-out으로 빼는 후보다.** 이 후보는 $D^2$가
$2.3\times10^{-16}$에서 $2.4\times10^{-16}$으로, 수치 정밀도상 clean 후보와 마찬가지로 $0$이다.
그런데 겨냥하는 값은 $-0.443$ 부근이고 참값에서 약 $1.44$ 떨어져 있다. 따라서 이 DGP에서는
balance discrepancy만으로 clean 후보와 이 후보를 구별할 수 없다.

유효 후보에서 $r$을 뿌리 밖으로 밀면 discrepancy와 인과 편향이 함께 커지고, 그 비율
$\lvert\tau_Z-1\rvert/D$의 중앙값이 $8.53$이다. $D=0.01$이면 편향이 약 $0.085$다. 이 환산은 유효
후보에만 적용된다. 눈먼 후보는 $D=0$인데 편향이 $1.44$이므로 어떤 환산도 성립하지 않는다.

---

## 4. 두 단계가 서로 다른 일을 한다

확증 실행에서 20회를 합산한 후보 처리 결과다.

| SCM | screen이 남긴 정상 후보 | screen이 남긴 눈먼 후보 | 최종 집단의 정상 후보 | 최종 집단의 눈먼 후보 |
| --- | ---: | ---: | ---: | ---: |
| SCM-1 | 118 | 16 | 118 | **0** |
| SCM-2 | 118 | 16 | 118 | **0** |
| SCM-3 | 118 | 16 | 118 | **0** |
| SCM-4 | 125 | 18 | 125 | **0** |
| SCM-5 | 123 | 17 | 123 | **0** |

screen은 눈먼 후보를 설정별 16회에서 18회, 합계 83/100회 통과시켰다. 3절이 보인 대로 이것은
screen의 구현 오류라기보다 balance criterion 자체의 한계다. 그런데 선택된 최대 연결성분 진입은
합계 **0/100회**다. 이 실험에서는 눈먼 후보의 추정값이 clean 후보들의 추정값 무리와 분리되었고,
Algorithm 1의 반경 그래프와 최대 연결성분 단계가 이를 제외했다.

**정리하면 이 확증에서 screen은 mixed 및 $W_0$-held-out 후보를 모두 제거했고, 집계는 균형을 이루면서
잘못된 값을 겨냥하는 singleton을 제거했다.** 어느 한쪽만으로는 이 관측된 후보 집합을 처리하기에
충분하지 않았다.

---

## 5. E9: hard4 실패 원인

SCM-1, 결과 잡음 표준편차 $1.0$, 같은 총 표본 $n=6000$과 같은 20개 paired seed에서 자료 역할만
바꿨다. 네 fold를 $F_0,\ldots,F_3$라 할 때 H11은 기존 hard4, H12는 N/E를 두 방향으로, H21은
D/S를 두 방향으로, H22는 둘 다 두 방향으로 사용한다. R4는 네 역할을 모두 회전한다. 동일 evaluation
행에 여러 표현을 적용할 때는 score를 행별로 먼저 평균했으며, 한 행을 여러 독립 관측처럼 세지 않았다.

| arm | 고유 평가행 | MAE | p90 | 평균 $\rho$ | $\lvert\tau_Z-1\rvert$ 평균 | $\lvert\widehat\tau-\tau_Z\rvert$ 평균 | component mixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H11: single-fit hard4 diagnostic | 1500 | 0.1139 | 0.2552 | 0.2316 | 0.0750 | 0.0694 | 1.10 |
| H12: N/E 두 방향 | 3000 | 0.0965 | 0.1693 | 0.1646 | 0.0785 | 0.0351 | 1.00 |
| H21: D/S 두 방향 | 1500 | 0.0893 | 0.1782 | 0.2008 | 0.0549 | 0.0669 | 0.00 |
| H22: D/S와 N/E 두 방향 | 3000 | 0.0650 | 0.1032 | 0.1422 | 0.0549 | 0.0371 | 0.00 |
| R4: 네 역할 회전 | 6000 | **0.0302** | **0.0637** | 0.0994 | 0.0376 | 0.0315 | 0.00 |

H21/H22에서 mixed 후보가 최대 연결성분에서 완전히 사라졌는데도 사전 MAE와 p90 기준을 통과하지
못했다. 따라서 screen membership 하나가 실패 원인은 아니다. H22는 H11보다 paired 절대오차가 평균
$0.0489$ 작았고 보통 95% $t$ 구간은 $[-0.0919,-0.0058]$였다. 그러나 H22는 R4보다 오차가 평균
$0.0348$ 컸으며 구간은 $[0.0154,0.0543]$였다.

현재 근거가 지지하는 가장 좁은 결론은 **one-shot candidate와 score에 남은 변동을 네 역할 회전에서
평균하는 것이 필요했다**는 것이다. D/S arm은 representation과 screen을 함께 바꾸고, N/E arm은
nuisance와 evaluation을 함께 바꾸므로 그 안의 단일 구성요소까지 원인으로 분해하지 않는다. 공통
$\rho$도 같은 방향으로 줄었지만, radius 하나만의 인과효과는 식별하지 않았다.

E9의 네 원래 셀을 모든 frozen empirical gate로 다시 채점하면 결과 잡음 $1.0$과 $0.3$ 모두 hard4는
FAIL, rotate4는 PASS다. 20/20 단일 반환은 이 benchmark의 경험적 기준을 통과한 것이지 모집단 반환
확률 $0.95$에 대한 joint 95% certificate가 아니다.

---

## 6. Algorithm 1은 어디까지 실험했는가

### 6.1 구현되고 실제로 작동한 핵심

1. 다섯 proxy block의 nonempty proper split 30개를 만든다.
2. 각 split에서 data-only restricted Brier objective로 표현을 맞춘다.
3. 독립 screen fold의 zero-null nonrejection rule로 후보를 남긴다.
4. retained split마다 nuisance fold에서 nuisance를 맞추고, 겹치지 않는 evaluation fold에서 AIPW
   candidate를 계산한다.
5. 공통 반경 $\rho$로 $\lvert\widehat\theta_S-\widehat\theta_{S'}\rvert\leq2\rho$ graph를 만들고
   connected components를 계산한다.
6. 가장 큰 component의 median을 반환한다. 최대 component가 동률이면 각 median의 candidate set을
   반환하며, 임의의 한 점으로 tie-break하지 않는다.

여기서 규칙은 절대 과반을 요구하는 majority vote가 아니라 **largest-component plurality**다. 다만 이
실험의 선택된 component는 clean 후보만 포함했고, 잘못된 singleton은 83회 screen을 통과하고도 0/100회
선택됐다. 따라서 graph, largest-component, median 집계가 실제 자료에서 작동했다.

### 6.2 확인된 구현의 정확한 범위

최종 PASS는 다음의 **특수형**이다.

- 후보 예산은 임의의 $m$이 아니라 전체 $m=T=30$이다.
- encoder는 고정된 101점 scalar $r$ grid이고 critic은 bounded-probit linear class다.
- base critic은 zero start 하나, augmented critic은 lifted-base와 zero start에서 BFGS로 근사 최적화하며
  전역 ERM certificate는 없다.
- 네 번의 role rotation에서 모두 screen을 통과한 split만 남기고, 네 disjoint evaluation score를 합치는
  rotate4 operational crossfit이다.
- bootstrap $\rho$는 evaluation-score sampling 변동을 반영하지만 representation, screen, nuisance
  학습오차를 모두 덮는 원고의 uniform radius certificate는 아니다.

따라서 **Algorithm 1의 candidate–screen–AIPW–graph–largest-component–median 메커니즘은 end-to-end로
검증됐다.** 반면 literal one-shot Algorithm 1/theorem, arbitrary $m<T$에서 fresh learner를 다시 맞춘
무작위 search, neural/general encoder, theorem-level uniform radius와 separation은 검증되지 않았다.

---

## 7. 확인됨과 미확인

### 확인됨

- 다섯 설정 모두 표본과 모집단에서 naive 방향이 음수이고 ATE는 $1$이다.
- 모집단 clean 후보와 oracle target은 $1$이며 raw ordinary adjustment target은 약 $0.80$대다.
- balanced-but-wrong singleton이 존재하고, balance discrepancy만으로는 이를 제거할 수 없다.
- $n=6000$, 20회 restricted-Brier rotate4에서 다섯 설정이 frozen empirical gate를 모두 통과했다.
- 잘못된 singleton은 83/100회 남았지만 최대 연결성분에는 0/100회 들어갔다.
- E9에서 screen membership 정정만으로 hard4 실패가 해결되지 않았고, full rotate4가 필요했다.

### 미확인

- 원고 Section 4 one-shot finite-sample theorem의 직접 적용과 정리의 확률 lower bound
- 임의의 $m<T$에 대해 representation과 nuisance를 새로 맞추는 end-to-end random-search 성능
- neural encoder, 일반 critic class, 이미지·고차원 SCM-6/7의 학습 성능
- population $\Gamma$, uniform learned-representation radius, separation certificate
- 고정 101점 grid의 점근적 수렴
- E9의 D/S 내부에서 representation 대 screen, N/E 내부에서 nuisance 대 evaluation의 개별 기여

---

## 8. 원시 근거와 SHA-256

| 역할 | 파일 | SHA-256 |
| --- | --- | --- |
| SCM-1~5 Brier 확증 protocol | `results/probe_structured_scm_brier_g_confirm_protocol_v2.json` | `cf5b0f3b9e678af3763993e044783584015af60d22d69f57f5b781849795fa65` |
| SCM-1~5 확증 summary/verdict | `results/probe_structured_scm_brier_g_confirm_summary_v1.json` | `69c3c6e50c2d62ea5482d65583faf48028689003e659e534c0db413ec4b5a06b` |
| SCM-1 raw | `results/probe_structured_scm_brier_g_confirm_v2_g1.json` | `6e506f4b52268d6a80f56d3ee6f64c7c5288f7a3d727869909a17afc86dc8de1` |
| SCM-2 raw | `results/probe_structured_scm_brier_g_confirm_v2_g2.json` | `1c4691ef3bf57937d62891dacdb5de2e71060e590b496e6a8f45f8985f27568b` |
| SCM-3 raw | `results/probe_structured_scm_brier_g_confirm_v2_g3.json` | `99886bddc23b539fad5fc74f8495164e5b91fad176c2ae031dfefdc1bdb8139f` |
| SCM-4 raw | `results/probe_structured_scm_brier_g_confirm_v2_g4.json` | `a9c07ce97ca57c09b44b99f8687a6a569eb2dbf1b0e39262e91ae70f7f5d3af9` |
| SCM-5 raw | `results/probe_structured_scm_brier_g_confirm_v2_g5.json` | `1556eb7f9070fe2cda6ba947c7f0a2dc9ac4da18d19a649f10d89a5174f804dd` |
| population identification check | `results/probe_structured_scm_population_checks_v2.json` | `5207107770ef4e7d7e6fa71c4ef9179ff11dbce0e7bb8fd37a43ded4c8355083` |
| E8 population raw | `results/probe_e8_population_v2.json` | `8fe3227efd0d4dd95903ef06ec0bbbd19f717f3534d8ee0b2581a121d83c3604` |
| E9 component protocol | `results/probe_e9_hard4_diagnosis_protocol_v1.json` | `7284209ea2ee47e107d4d01075dc9d5159a009e7f761040b3f738a1112176511` |
| E9 component raw | `results/probe_e9_hard4_component_dev_v1.json` | `a3548c95ef670df88b4b4e7dbc5ec9b5b47c6c29e990e9cedf2d14465329cdc9` |
| E9 hard4 baseline sidecar | `results/probe_e9_hard4_baseline_rescore_v1.json` | `c9d0b45ffc5439ef317d3e15e3fec8c47021e48f816df4227b47656588cb9dac` |
| E9 full gate verdict | `results/probe_e9_five_gate_verdict_v2.json` | `e9bdf550b521b80a583ef04b9287ca20bac6640c07de5928d64ccd4b57c48efe` |

확증 summary는 `complete=true`, 설정별 20회, 총 표본 6000, 회전 4회를 기록한다. E9 component raw도
`complete=true`이며, 20개 모든 seed에서 생성 fold hash가 기존 hard4와 rotate4 raw 양쪽에 일치하고
H11 출력은 기존 hard4와 정확히 같았다. 기존 raw와 protocol은 이 문서를 작성하며 수정하지 않았다.
