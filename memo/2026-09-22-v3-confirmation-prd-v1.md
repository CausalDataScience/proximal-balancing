# v3 전향적 재현 검증 PRD v1 (실행 전 수정안)

작성·수정: 2026-09-22 CDT. 상태: **PRD 수정만 승인됨. 실행·확증 완료가 아니다.** 구현 준비와 실행 승인을 받은 뒤 규약을 동결하고, 그 뒤에 여는 replicate의 결과를 보고 바꾸지 않는다.

## 0. 한 문장

자료의 잡음 크기에 따라 문턱을 정하는 v3 심사 규칙을 고정하고, 지정한 새 replicate 30건에서 반환율과 반환 조건부 오차의 재현성을 평가한다. 완료 후 판정에 따라 본문·부록 그림을 배치한다.

## 1. 정직한 이력 (원고 부록에 이 표 그대로 들어간다)

| 판 | 심사 규칙 | 언제 정했나 | 검증 |
|---|---|---|---|
| v1 (봉인) | rotation 네 개 각각 $\max(\text{gap},0)+z\,\mathrm{SE}\le t$, $t$는 개발 자료에서 고정한 상수 | SCM-1~4 확증 전 | SCM-1~4 확증 80건 PASS |
| v2 pooled | 네 rotation의 gap 평균과 SE 합성으로 한 번 판정 | RHC v1이 무응답인 것을 **본 뒤** | 봉인 80건에서 추정값 넷째 자리까지 불변 |
| v3 noise floor | $t=\operatorname{median}_S\, z\,\mathrm{SE}_S$, 자료마다 | RHC, RHC 심기, ACIC, IHDP가 무응답인 것을 **본 뒤** | 봉인 80건에서 추정값 최대 변화 $0.0051$; 봉인 상수가 각 자료의 잡음 바닥과 비 $0.93$, $1.11$, $0.94$, $1.02$ |
| **이 PRD** | v3 고정 | 열지 않은 replicate를 열기 **전** | 아래 3절 |

숨기지 않는 사실: v2와 v3는 실패를 본 뒤 정한 새로운 운용 규칙이다. 봉인 상수와 잡음 크기가 비슷하다는 관찰이나 봉인 80건의 작은 변화는 사후 민감도 분석이지 새 규칙의 독립 확증이 아니다. 이 PRD는 과거를 사전 등록으로 되돌리지 않으며 **앞으로 여는 반복에 대해서만 전향적으로 규칙을 고정한다.**

## 2. v3 규칙 (고정)

후보 $S$의 rotation $r=1..4$에서 봉인 실행이 저장한 $\text{gap}_{S,r}$과 $\mathrm{SE}_{S,r}$에 대해
$$\bar g_S=\max\!\Big(\tfrac14\textstyle\sum_r\text{gap}_{S,r},\,0\Big),\qquad \mathrm{SE}_S=\tfrac14\sqrt{\textstyle\sum_r\mathrm{SE}_{S,r}^2},\qquad z=\Phi^{-1}(1-0.05/30)$$
$$t=\operatorname{median}_{S}\ z\,\mathrm{SE}_S,\qquad \text{보류}\iff \bar g_S+z\,\mathrm{SE}_S\le t.$$
이 규칙 안에서 추가로 튜닝할 계수는 없다. 보류된 집합의 공통 반경과 최대 성분 중앙값은 기존 집계 코드를 사용한다. 구현 출발점은 `code/noise_scaled.py`의 `decide`, 검사는 `code/test_noise_scaled.py`다. 아래 완결성·overlap 검사를 구현한 뒤 전체 의존 코드를 함께 동결한다. 현재 코드를 이미 동결된 최종 구현이라고 부르지 않는다.

**해석 제한:** 위 $\mathrm{SE}_S$는 rotation 간 공분산을 생략한 working SE다. 행 역할과 학습 자료를 공유하므로 정확한 표준오차나 동시 신뢰상한이라고 주장하지 않는다. 후보별 SE의 중앙값을 문턱으로 삼으면 gap이 같아도 SE가 작은 후보가 유리하고, 정확히 균형인 후보도 SE가 중앙값보다 크면 탈락할 수 있다. 따라서 이는 잡음에 맞춘 경험적 선택 규칙이지 타당한 후보를 모두 보존한다는 보증이 아니다. 이 제한 때문에 새 반복의 반환과 오차를 실제로 확인한다.

수치 판정에 더해 네 rotation의 `overlap`이 모두 참인지 확인한다. NaN/Inf, 음의 SE, 없는 필드, 누락·중복 split/rotation, 기대 30개 후보와 다른 이름, 잘못된 score 행렬은 실행 결함으로 거부한다. 현재 bounded critic은 통상 overlap 범위 안이지만 검사 자체를 제거하지 않는다.

## 3. 확증 자료 (고정, 아직 열지 않음)

| 자료 | 지금까지 연 것 | 확증에 열 것 | seed 규칙 (기존과 같음) |
|---|---|---|---|
| Twins | 0 (개발), 1–10 | **11–20** | `98,210,000 + 1000r`, $w$ seed `98,200,000` 고정 |
| RHC 심기 | 0 (개발), 1–10 | **11–20** | `98,300,000 + 1000r`, $w$ seed `98,300,000` 고정 |
| IHDP | 1 (개발), 2–11 | **12–21** | `98,400,000 + 1000r`, $w$ seed `98,400,777` 고정 |
| ACIC | 1 (개발), 2–10 | **없음.** 배포된 10개를 다 썼다 | |

fold seed는 자료 seed $+7$이다. 반복 단위는 자료별로 다르다.

- Twins: 고정된 실제 $X$, 숨긴 변수와 잠재결과 위에서 proxy 잡음과 처치 배정을 새로 생성한다.
- RHC 심기: 실제 $X$, 숨긴 변수, **proxy와 결과 기준값도 고정**하고 처치 배정만 새로 생성한다. 결과는 고정 기준값에 심은 처치효과를 더한다.
- IHDP: `ihdp_acic_data.ihdp(r)`의 **0-based 배열 인덱스** $r$를 사용한다. benchmark 잠재결과 realization도 바뀌며, 그 위에서 proxy와 처치를 생성한다. 고정 잠재결과의 배정 반복이라고 쓰지 않는다.
- ACIC: 이미 사용한 자료이며 새 검증에 넣지 않는다. **v3를 정할 때 본 탐색 자료**로 부록에만 남긴다.

## 4. 실행 (고정)

- 봉인 `family_v2_probe.run_dataset`, 자료 모듈, recipe, 비교군의 수학적 계산은 무수정. 실행 범위는 별도 manifest로 지정한다. IHDP의 기존 기본 범위를 몰래 덮어쓰지 않고 명시적 ID 입력 경로를 준비한다.
- fresh scorer는 Twins `[11,12,13,14,15,16,17,18,19,20]`, RHC 심기 같은 목록, IHDP `[12,13,14,15,16,17,18,19,20,21]` **각각만** 읽는다. `replicate_*.json` 전체 glob이나 기존 summary의 고정 범위를 사용하지 않는다.
- 각 shard의 자료명·replicate·seed·source hash·완결성을 manifest와 대조한다. 예상 10건이 모두 완결되어야 채점한다. 기존 결과나 개발 자료를 섞지 않는다. `noise_scaled.decide`로 새 출력을 계산하고, 모든 비교군은 같은 fresh shard의 값을 읽는다.
- CPU만, worker 2개, BLAS 스레드 2개, `nice -n 10`. 예상 약 1시간.
- 결과는 `results/<자료>/replicate_r.json` (새 $r$)과 `results/<자료>/summary_v3_fresh.json`, 모두 write-once.
- 실행 오류·누락·비유한 결과는 `incomplete`로 기록하며 PASS를 주지 않는다. 정상 계산 뒤의 통계적 `no_return`은 별도 상태다. 기존 파일이 있다는 이유만으로 무결성 검사 없이 건너뛰지 않는다.

## 5. 보고 항목과 판정 (사전 선언)

자료마다 새 replicate 10건에서: v3 반환율, **반환한 자료에서의 조건부 평균 절대오차와 p90**, 비교군 전부(naive, $X$만, raw, oracle, P2SLS 둘, CEVAE; Park은 Twins에서만 ATE 변환으로), 보류 수, 잡음 크기, 짝지은 한쪽 상한(oracle, $X$만, raw, CEVAE 대비)을 보고한다.

반환율의 분모는 항상 지정한 10건이다. 무반환에 임의 오차를 대입하지 않는다. 비교군 자체의 전체 10건 성적을 함께 표시하고, PROBE와의 paired 비교는 **두 방법 모두 반환한 동일 ID subset**에서 계산하여 ID와 쌍 수를 공개한다. 기존 paired-$t$ 산식은 표본평균 차이 $+t_{0.95,k-1}s_d/\sqrt{k}$이며 $k<2$면 산출 불가로 기록한다. 이 조건부 비교를 전체 10건에 대한 우월성으로 표현하지 않는다.

TPR/FPR은 해당 생성 설계에서 후보의 타당성 정답이 뒷받침될 때만 사용하고 근거를 명시한다. **실제 RHC 및 RHC 심기의 실제 proxy에는 합성 SCM의 일곱 TARGETS를 그대로 적용하지 않으며 TPR/FPR은 N/A**다. 대신 후보별 유지 여부와 연결 성분을 보고한다.

**통계적 관문은 기존 두 개를 유지한다.** 자료마다 (i) 반환이 10건 중 8건 이상, (ii) 반환 조건부 평균 절대오차가 그 자료의 기존 replicate(1–10 또는 2–11)에서의 v3 오차의 1.5배 이내.

**관문 (i)이 무엇을 보는지 미리 적는다.** 문턱을 후보별 $z\,\mathrm{SE}_S$의 중앙값으로 잡으면 정의상 약 절반이 그 이하이므로 보류 집합이 비는 일은 거의 없다. 실제로 RHC 실제 자료에서 30개 후보의 pooled gap이 전부 0으로 잘리고 보류는 정확히 15개였다. 따라서 (i)은 심사의 검정력이 아니라 **집계 단계의 동점 여부**를 보는 관문이고, 정보를 담은 관문은 (ii)다. 통과를 "심사가 후보를 가려냈다"로 읽지 않는다. 세 자료가 모두 충족하면 **"v3 전향적 재현 관문 통과"**다. 절대적 정확성, oracle 근접, 비교군 우월성을 통과 조건에 새로 넣거나 통과만으로 주장하지 않는다.

기준은 현재 `results/noise_scaled_v3.json`의 기존 ID 목록과 저장 수치로 고정한다. 파일 SHA256은 `a0565c5f2e07a93baec374e730300f1575c74e94650c9fa61b6b2f9dfe1aeb8d`다.

| 자료 | 기존 v3 조건부 MAE (저장값) | 새 관문 상한 (표시용 반올림) |
|---|---:|---:|
| Twins | 0.009557489874420181 | 0.01433623 |
| RHC 심기 | 0.45349871582366097 | 0.68024807 |
| IHDP | 0.33618081767248315 | 0.50427123 |

실제 판정은 위 반올림 열이 아니라 **참조 JSON의 원래 수치 × 1.5**로 한다. 해당 수치·기존 ID·참조 파일 해시를 protocol에 복사하며 새 자료와 합쳐 기준을 다시 계산하지 않는다.

하나라도 실패하면: 실패한 자료를 포함해 **있는 그대로 보고**하고, 실제 자료 그림은 본문에서 부록으로 내리며, 본문은 모의실험과 RHC(참값 없음, 시연)만 싣는다. 규칙, seed, 자료, 문턱을 바꿔 다시 돌리지 않는다.

## 6. 그림 (고정)

- **본문 그림 A** `realdata-groundtruth`: 1×3. Twins, RHC 심기, IHDP의 **새 지정 replicate만** 그린다. 각 패널에 반환 건수/10과 무반환 ID를 표시한다. 줄과 색은 Figure 3과 같고 축은 "추정값 $-$ 참값"이다. ACIC는 별도 부록 탐색 그림으로 둔다.
- **본문 그림 B** `rhc-comparison`: v3의 점추정 약 $-1.29$, standard DR, proximal DR, 무조정 대비. 현재 그림의 `0.30`은 **보류된 구성원 하나의 AIPW 표준오차**이고 최종 집계량(보류 집합의 최대 성분 중앙값)의 표준오차가 아니다. 부록 설명만으로는 그림이 오해를 남기므로 둘 중 하나로 고친다.
  - (a) 저장된 score 교차곱에서 다변량 정규 표본을 뽑아 각 표본에 같은 집계를 다시 적용하고, 그 분포의 표준편차와 2.5/97.5 분위수를 PROBE 줄의 막대로 쓴다. 산식을 caption 한 줄과 부록에 적는다.
  - (b) PROBE 줄의 구간 막대를 빼고 점만 두며, caption에 "이 점에는 최종 집계량의 표준오차를 붙이지 않는다"를 적는다.
  구현 시 (a)를 먼저 시도하고, 집계가 동점이나 성분 이동으로 불안정하면 (b)로 내린다. 어느 쪽을 썼는지 결과에 기록한다. RHC에는 참값이 없으므로 정확도 검증이 아닌 시연이다.
- 부록: 무엇이 실제이고 무엇이 생성인지 표(RHC, RHC 심기, Twins, IHDP, ACIC), 1절의 이력 표, 자료별 요약표, 확증 replicate 대 기존 replicate 대조, 그리고 아래 6.1의 RHC 보류 집합 해석.

### 6.1 RHC 보류 집합을 무엇이 갈랐는가 (고정 서술)

v3가 RHC에서 보류한 15개는 $W_1=(\texttt{pafi1},\texttt{paco21})$를 남기는 후보 전부이고 탈락한 15개는 그것을 빼는 후보 전부다. 이 분리를 만든 것이 무엇인지 정확히 적는다.

- **심사 fold에서는 SE가 갈랐다.** 30개 후보 모두 pooled gap이 0으로 잘렸고, 보류 여부는 $z\,\mathrm{SE}_S$가 중앙값 아래인지로만 결정됐다. $W_1$을 빼는 후보의 평균 $\mathrm{SE}$는 $0.00247$, 남기는 후보는 $0.00143$이다.
- **학습 fold에서는 불균형 자체가 갈랐다.** 같은 실행이 저장한 `gap_fit`(표현 학습이 실제로 도달한 잔여 불균형)은 두 집단을 겹침 없이 나눈다. $W_1$을 남기는 120개 (후보, rotation) 값의 최댓값이 $0.00346$이고, 빼는 120개의 최솟값이 $0.00436$이다. 중앙값은 각각 $0.00073$과 $0.00715$다.

따라서 "심사가 이름표 없이 경계를 그었다"고 쓰지 않는다. 쓸 수 있는 문장은 이것이다. **서로 다른 두 양이 같은 경계에 동의한다.** 심사 fold의 보류는 정밀도로 갈렸고, 학습 fold의 잔여 불균형은 그와 독립적으로 같은 두 집단을 갈랐다. 두 번째가 $W_1$이 균형을 거부한다는 실질 증거이며, 원고에는 두 사실을 갈라 적는다.

## 7. 원고 문장 (실행 결과를 확인한 뒤 반영할 초안)

**Algorithm 1과의 관계:** 후보 추정·연결·유일 최대 성분의 중앙값이라는 집계 구조를 유지한다. 입력 문턱의 자료적응 규칙과 pooled working-SE screen은 v3의 operational 구현으로 명시한다. $t$가 입력이라는 이유만으로 원고의 모든 확률 보증을 이 규칙에 자동 적용하지 않는다.

**부록 D.3의 v1 설명은 유지한다.** SCM-1~4의 기존 모의실험 숫자는 실제 사용한 봉인 상수·rotation별 판정으로 설명한다. 아래 v3 문단을 별도 추가하며 기존 절차를 소급해 바꾸지 않는다.

> For the v3 operational analysis, we average the four rotation-specific gaps and use $\tfrac14(\sum_r\mathrm{SE}_r^2)^{1/2}$ as a working scale that omits cross-rotation covariance. The threshold is the median candidate-specific noise term. This data-adaptive rule was developed after observing nonreturns and favors more precisely estimated candidates; it is not a new confidence guarantee. We froze it before the designated prospective replicates. Their return rates and errors conditional on return are reported separately from the original sealed simulations.

**본문의 문턱 설명은 실험군을 구분한다.** 기존 모의실험의 고정 문턱 문장은 유지하고 v3 결과에만 다음 문장을 붙인다.

> For v3, the threshold is data-adaptive, with its rule frozen before the designated prospective replicates were opened.

**봉인 80건 재채점은 사후 민감도 분석으로만** 보고한다. 최대 변화는 `0.005086718371059118` (약0.005087)이며 0.005 이하가 아니다. 문턱 비율은 $t/\text{floor}$ 기준 0.93, 1.11, 0.94, 1.02이고(저장 파일의 `floor`/`t`로는 1.08, 0.90, 1.06, 0.98이다) SCM-2가 11%이므로 모두 10% 이내라고 쓰지 않는다. 어느 방향의 비인지 항상 함께 적는다. 작은 변화는 v1 결과를 보존할 이유를 대체하지 않는다. 새 검증 결과가 나오기 전에는 "confirmed"라는 완료 문장을 넣지 않는다.

## 8. 하지 않을 것

- 확증 결과를 본 뒤 $U$, 계수, seed, 블록, 문턱 규칙, 자료 목록을 바꾸지 않는다.
- ACIC를 다시 돌리지 않는다.
- 판정 (i)(ii) 외의 기준을 사후에 더하지 않는다.
- 실패한 자료를 그림에서 빼지 않는다.

## 9. 확정 절차

다음은 별도의 구현·실행 승인 후 수행할 절차다. **이번 요청은 PRD 수정만 승인한 것이다.**

1. fresh manifest scorer, finite/schema/overlap 검사, write-once와 실패 분모 검사를 준비한다. 통계적 규칙과 두 관문은 바꾸지 않는다.
2. protocol에 이 PRD, `noise_scaled.py`, `test_noise_scaled.py`, **새 scorer**, 세 runner의 `SOURCES` 합집합과 실제 추가 의존 파일의 SHA256을 넣는다. 특히 `rhc_pooled.py`, 집계 코드, 자료 생성기·recipe·비교군·runner를 빠뜨리지 않는다. 입력 benchmark 파일, 기존 참조 shard·summary·`noise_scaled_v3.json`의 해시, ID/seed manifest와 수치 관문도 기록한다.
3. `results/v3_confirmation_protocol.json`을 write-once로 만들고 동결 시각을 남긴다. 지정 새 ID의 선행 결과가 없음을 확인한다. 있으면 자동으로 fresh라고 부르지 않고 사용 이력을 확인한다.
4. 그 뒤에만 지정 자료를 실행한다. scorer는 protocol과 source·입력 해시, 30건 완결성을 다시 확인한다. 실행 오류가 남으면 incomplete이며 통계 관문 채점 완료를 주장하지 않는다.
