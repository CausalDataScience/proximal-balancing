# E9 hard4 실패 원인과 최소 수정 결과

작성일: 2026-09-12  
상태: 완료  
범위: SCM-1, 총 표본 $n=6000$, 기존 20개 paired base seed

## 1. 결론

hard4가 실패한 직접 원인은 한 번만 학습한 표현·screen과 한 번만 맞춘 nuisance·평가 score의 변동이
후보 추정치에 함께 남기 때문이다. screen membership이나 공통 반경 $\rho$만 고쳐서는 충분하지 않았다.

- hard4(H11)의 MAE는 $0.1139$였다.
- D/S를 두 번 맞추고 N/E도 두 번 cross-fit한 H22는 MAE를 $0.0650$으로 낮췄지만, 사전 기준
  MAE $\leq0.05$와 p90 $\leq0.10$을 여전히 넘었다.
- 네 역할을 모두 회전해 동일 evaluation unit의 score를 먼저 평균한 rotate4는 MAE $0.0302$,
  p90 $0.0637$로 여섯 확인 항목을 모두 통과했다.

따라서 production 실험 모드는 **rotate4 operational crossfit**으로 지정한다. hard4는 한 번의
표현·screen·nuisance·평가 변동에 얼마나 민감한지 보는 진단 arm으로 보존한다. 이는 원고 Section 4의
one-shot conditional finite-sample theorem을 그대로 적용한 것이 아니라 별도의 operational extension이다.

## 2. 무엇을 고쳤는가

과학적 추정 규칙은 바꾸지 않았다. DGP, $t=3.5$ screen, 101점 $r$ 격자, 제한 Brier critic,
nuisance, 공통 반경, Algorithm 1의 최대 연결성분과 median 규칙은 모두 그대로다.

구현에서 바꾼 것은 다음뿐이다.

1. 모든 AIPW score에 D/S/N/E fold ledger를 붙이고, evaluation 행과 학습 행의 겹침을 fail closed했다.
2. 같은 evaluation unit에 여러 D/S 표현을 적용하면 score를 unit별로 먼저 평균한다. 같은 행을 여러
   독립 관측처럼 이어 붙이지 않는다.
3. raw, $X$-only, oracle baseline을 하나의 공통 helper로 계산하게 했다.
4. 일반 runner에서 `--mode`를 필수로 만들고 rotate4를 production operational mode, hard4를
   one-shot diagnostic으로 명시했다.
5. hard4의 누락 baseline은 저장된 fold SHA-256이 재생성 fold와 일치할 때만 별도 sidecar에 복원했다.
6. E9 판정기는 MAE, p90, unconditional return, $X$/raw와의 paired superiority, oracle-excess를 모두
   계산한다.

제한 Brier 선택은 D와 optimizer seed가 고정되면 결정론적이었다. 같은 입력을 서로 다른 optimizer
seed로 재생해 같은 결과가 나오는 테스트를 추가했다. 따라서 단순히 optimizer 반복 횟수를 늘리는
것만으로 이번 개선이 생긴 것은 아니다. 확인된 개입은 자료 역할 재사용과 회전 평균이다.

## 3. paired component ablation

네 고정 fold를 $F_0,\ldots,F_3$라 하자. 각 fold는 1500행이다.

| arm | D/S fit | N/E fit | 고유 evaluation 행 | 해석 |
| --- | --- | --- | ---: | --- |
| H11 | $(F_0,F_1)$ | $(F_2,F_3)$ | 1500 | 기존 hard4와 동일 |
| H12 | $(F_0,F_1)$ | $(F_2,F_3),(F_3,F_2)$ | 3000 | N/E만 두 방향 |
| H21 | $(F_0,F_1),(F_1,F_0)$ | $(F_2,F_3)$ | 1500 | D/S만 두 방향 |
| H22 | 두 방향 | 두 방향 | 3000 | D/S와 N/E 모두 두 방향 |
| R4 | 네 역할 전부 회전 | 네 역할 전부 회전 | 6000 | 기존 rotate4 |

H21/H22의 두 번째 D/S fit은 저장된 다른 rotation을 재사용하지 않고 실제로
$(D,S)=(F_1,F_0)$에서 다시 맞추고 audit했다. 모든 arm의 총 생성 행은 동일하게 6000이다.

### 3.1 결과와 signed 분해

선택된 component의 모집단 representation target을 $\tau_{Z}$, 최종 추정치를 $\widehat\tau$라 하면

$$
\widehat\tau-1=(\tau_Z-1)+(\widehat\tau-\tau_Z).
$$

아래 signed 평균은 이 항등식을 각 seed에서 계산한 뒤 평균한 것이다. 절대값 열은 보조 진단이며,
서로 더해지는 매개 백분율로 해석하지 않는다.

| arm | MAE | p90 | 평균 $\rho$ | signed total | signed representation | signed conditional estimation | mean $\lvert\tau_Z-1\rvert$ | mean $\lvert\widehat\tau-\tau_Z\rvert$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H11 | 0.1139 | 0.2552 | 0.2316 | +0.0460 | +0.0345 | +0.0115 | 0.0750 | 0.0694 |
| H12 | 0.0965 | 0.1693 | 0.1646 | +0.0442 | +0.0310 | +0.0132 | 0.0785 | 0.0351 |
| H21 | 0.0893 | 0.1782 | 0.2008 | +0.0179 | −0.0015 | +0.0194 | 0.0549 | 0.0669 |
| H22 | 0.0650 | 0.1032 | 0.1422 | +0.0176 | −0.0015 | +0.0191 | 0.0549 | 0.0371 |
| R4 | 0.0302 | 0.0637 | 0.0994 | +0.0003 | −0.0082 | +0.0085 | 0.0376 | 0.0315 |

H11 대비 paired 절대오차 변화의 보통 95% $t$ 구간은 다음과 같다. 이 구간은 진단용이며 사전
다중비교 성능 판정 구간이 아니다.

| 대비 | 평균 변화 | 95% 구간 |
| --- | ---: | --- |
| H12 − H11 | −0.0174 | [−0.0429, +0.0081] |
| H21 − H11 | −0.0245 | [−0.0591, +0.0101] |
| H22 − H11 | −0.0489 | [−0.0919, −0.0058] |
| R4 − H11 | −0.0837 | [−0.1247, −0.0427] |

R4 대비 H22의 절대오차는 평균 $+0.0348$ 더 컸고 95% 구간은 $[+0.0154,+0.0543]$였다.
두 방향만 쓰는 H22보다 네 회전의 candidate/score 평균이 추가로 필요했다는 근거다.

### 3.2 membership과 radius만으로 설명되는가

| arm | retained clean | retained bad | retained mixed | component clean | component mixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| H11 | 6.65 | 0.90 | 1.10 | 6.65 | 1.10 |
| H12 | 6.65 | 0.90 | 1.10 | 6.65 | 1.00 |
| H21 | 6.30 | 0.85 | 0.00 | 6.30 | 0.00 |
| H22 | 6.30 | 0.85 | 0.00 | 6.30 | 0.00 |

수치는 20개 base seed의 평균 후보 수다. $W_0$를 포함한 후보는 네 arm 모두 0이었다. D/S를 두
방향으로 audit하면 mixed 후보가 모두 사라졌지만 H21과 H22는 각각 MAE $0.0893$, $0.0650$이었다.
따라서 잘못된 membership 제거만으로 hard4 실패가 해결되지는 않는다. H22의 반경도 $0.1422$까지
줄었지만 p90이 $0.1032$로 기준을 조금 넘었다. 최종 통과에는 R4의 네 회전 candidate/score 평균과
더 작은 반경이 함께 필요했다.

이 ablation은 D/S 안의 representation과 screen을 서로 분리하지 않고, N/E 안의 nuisance와 evaluation도
서로 분리하지 않는다. 그러므로 더 좁은 단일 원인을 주장하지 않는다.

## 4. 전체 frozen empirical benchmark 판정

아래 upper bound는 각 셀의 20개 paired base seed를 추론 단위로 한 Bonferroni one-sided $t$ upper
bound이며 세 비교($X$, raw, oracle)에 대해 $\alpha=0.05/3$을 썼다.

| 잡음 SD | mode | MAE | p90 | return | $X$ upper | raw upper | oracle-excess upper | 전체 판정 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 | hard4 | 0.1139 | 0.2552 | 1.00 | −0.7184 | +0.0400 | +0.1191 | FAIL |
| 1.0 | rotate4 | 0.0302 | 0.0637 | 1.00 | −0.8621 | −0.1221 | +0.0163 | **PASS** |
| 0.3 | hard4 | 0.0988 | 0.1976 | 1.00 | −0.7504 | −0.0013 | +0.1285 | FAIL |
| 0.3 | rotate4 | 0.0201 | 0.0409 | 1.00 | −0.8794 | −0.1404 | +0.0211 | **PASS** |

확인 항목은 MAE $\leq0.05$, p90 $\leq0.10$, 단일값 반환률 $\geq0.95$, $X$와 raw 대비 paired
upper bound $<0$, oracle 대비 추가 절대오차 upper bound $\leq0.05$다. 20/20 반환은 이 데이터의
empirical gate를 통과한 것이지 모집단 반환확률이 0.95 이상이라는 joint 95% certificate가 아니다.

hard4 baseline은 별도 sidecar에서 복원했다. 잡음 SD $1.0$에서 $X$/raw/oracle MAE는 각각
$0.9232/0.1567/0.0403$, 잡음 SD $0.3$에서는 $0.9243/0.1619/0.0121$였다. 모든 fold hash가 기존
hard4 결과와 일치할 때만 이 값을 계산했다.

## 5. 증거와 재현

- 사전 계획: `memo/2026-09-12-e9-hard4-root-cause-fix-plan.md`
- frozen protocol: `results/probe_e9_hard4_diagnosis_protocol_v1.json`
  (`SHA-256 7284209ea2ee47e107d4d01075dc9d5159a009e7f761040b3f738a1112176511`)
- component raw: `results/probe_e9_hard4_component_dev_v1.json`
  (`SHA-256 a3548c95ef670df88b4b4e7dbc5ec9b5b47c6c29e990e9cedf2d14465329cdc9`)
- reconstructed hard4 baselines: `results/probe_e9_hard4_baseline_rescore_v1.json`
  (`SHA-256 c9d0b45ffc5439ef317d3e15e3fec8c47021e48f816df4227b47656588cb9dac`)
- five-gate verdict: `results/probe_e9_five_gate_verdict_v2.json`
  (`SHA-256 e9bdf550b521b80a583ef04b9287ca20bac6640c07de5928d64ccd4b57c48efe`)

component 실행은 protocol에 저장된 core/runner/test/E8 population source hash를 fail closed로 검사했다.
20개 모든 replicate에서 생성 fold hash가 기존 hard4와 rotate4 결과 양쪽에 일치했고, H11 최종값은
기존 hard4와 기계 정밀도로 같았다. 저장된 원시 JSON은 수정하지 않았다.

검증 명령과 결과:

```text
cd research/papers/single_proxy_balancing/code
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest test_probe_structured_scm.py test_analyze_e9.py
Ran 22 tests in 2.794s — OK (skipped=1)

PROBE_SLOW_SNAPSHOT=1 PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest \
  test_probe_structured_scm.StructuredSCMTests.test_rotate4_refactor_reproduces_stored_e9_seed
Ran 1 test in 37.502s — OK
```

slow snapshot test는 같은 저장 seed에서 fold hash, aggregation output, baseline을 모두 재현한다.
새 과학적 score/selection 로직을 만들지 않았으므로 fresh rotate4 확인은 반복하지 않았다. production
판정은 서로 독립인 기존 rotate4 confirm seed namespace 2,400,000과 E9의 9,100,000 결과에 근거한다.
