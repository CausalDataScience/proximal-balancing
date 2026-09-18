# Section 5 스토리라인과 그림 설계 (초안, 2026-09-14)

## 0. 한 문장

하나의 인과 시스템을 세 가지 proxy 표현으로 렌더링하면 표준 조정은 모두 실패하고 PROBE만 참값을
회복한다. 그 이유는 Section 3이 예측한 그대로다. held-out balance는 가정이 성립하는 split의 bias를
증명하고, plurality 집계는 가정이 깨진 채 균형을 이루는 split을 걸러낸다.

## 1. 실험 설계: 하나의 SCM, 세 렌더링

리뷰어가 가장 먼저 묻는 것은 "그림마다 설정이 다르지 않은가"다. 세 실험은 **잠재 구조와 상수가
바이트 단위로 같고**, proxy를 렌더링하는 방식만 다르다. 코드에서 상수를 대조해 확인했다.

$$
X=U+2\varepsilon_X,\quad K_j=U+0.85\,\xi_j\ (j=0,\dots,4),\quad K_4\leftarrow K_4-0.6\,I,
$$
$$
A\sim\mathrm{Bern}\{0.1+0.8\,\Phi(U+2.5I+0.2X+0.3C)\},\qquad
Y=A-2.5U+0.2X-0.5C+0.3\varepsilon,\qquad \tau=1.
$$

블록 0은 $(I,C)$를 싣고, 블록 1–3은 $U$의 순수한 측정이며, 블록 4는 처치 전용 원인 $I$에 오염되어 있다.

| 이름 | proxy 렌더링 | raw $W$ 차원 |
|---|---|---:|
| SCM-1 | 채널 $K_j$를 스칼라 그대로 | 7 |
| SCM-6 | 256차원 직교 혼합에 잡음 255차원 | 1,282 |
| SCM-7 | MNIST 템플릿의 공간 변형, $96\times96$ | 46,082 |

naive 차이는 $-0.18$로 **부호가 뒤집힌다**(Simpson reversal). 30개 oriented split의 모집단 사실은
정확히 계산된다. 8개가 균형에 도달하고, 그중 7개(S1, S2, S3, S12, S13, S23, S123)가 $\tau$를 겨냥하며,
S4 하나는 정확히 균형을 이루면서 $-0.443$을 겨냥한다.

## 2. 네 질문, 네 정리, 네 그림

### 2.0 용어 (그림과 caption에 쓰는 말)

| 용어 | 뜻 |
|---|---|
| 블록 | proxy $W$를 미리 나눈 다섯 조각 $W_0,\dots,W_4$ |
| held-out 선택 | 비워 둘 블록의 조합 하나. 예를 들어 $\{W_1\}$이나 $\{W_1,W_2\}$. 모두 30가지 |
| 깨끗한 블록 | $U$에 잡음만 더한 블록. $W_1,W_2,W_3$ |
| 오염된 블록 | $U$와 함께 처치 전용 원인 $I$도 섞인 블록. $W_4$ |
| 후보 표현 $Z$ | 비워 두지 않은 블록과 $X$로 만든 요약 |
| 남은 불균형 $D_{\mathrm{res}}$ | $Z$를 알고 난 뒤에도 비워 둔 블록이 처치를 얼마나 더 예측하는지 |
| 균형 검사 | $D_{\mathrm{res}}$가 충분히 작은지 보는 screen |
| 다수결 | 균형 검사를 통과한 추정값 중 서로 가까운 가장 큰 무리의 중앙값. PROBE의 최종 답 |

그림 안에는 내부 이름(S4, S12 같은 split 번호)을 쓰지 않는다.

### 2.1 색과 표시

한 그림 안에서 색은 한 가지 구분만 한다. **파랑은 PROBE이거나 깨끗한 블록, 주황은 오염된 블록, 회색은 다른
방법이나 기준선.** 점선은 참값 $\tau=1$, 검은 세로 막대는 여러 자료에 대한 평균이다. 모든 표시의 뜻은 그림
안의 범례나 직접 붙인 글자로 적는다. 글자가 겹치지 않는지는 그림을 만들 때 코드가 글자 상자를 재서 검사한다.

### Q1. 숨은 교란이 표준 조정을 깨뜨리는가, PROBE는 고치는가

- **이론:** Theorem `thm:exact-proxy-identification`, Algorithm `alg:probe-search`의 출력.
- **그림 1:** 세로축은 방법 여섯, 가로축은 추정된 처치효과, 열 셋은 proxy 형태 셋. 점 하나가 모의 자료 하나.
- **읽는 법:** 점선에 붙어 있는 줄이 참값을 맞추는 방법이다. 파란 PROBE 줄만 세 열 모두에서 점선 위에 있다.
  맨 위 naive 줄은 0보다 왼쪽이라 효과의 부호까지 틀린다.
- **이미지 열에 대해 정직하게 적을 것:** 이미지 확증은 동결 관문 13개 중 8개만 통과해 FAIL이다. 답한 16개 자료는 모두 옳은 무리에서 나왔고 평균 오차 0.048이지만, 네 자료에서 답을 보류했다. 그림에는 "answered 16 of 20"을 적는다.
- **정직하게 적을 것:** $n=12{,}000$에서 oracle과의 짝지은 차이 상한 $0.053$은 관문 $0.05$를 넘었다. 같은 자료를 두 배로 키운 $n=24{,}000$에서는 $0.045$로 관문 안에 들어온다. learned target 오차도 중앙값 $0.054$에서 $0.036$으로 줄어 원래 관문을 통과한다. 다만 최종 오차의 짝지은 감소는 자료 10개로 유의하지 않다.

### Q2. 왜 되는가: 불균형이 0이면 bias도 0인가

- **이론:** Proposition `prop:zero-discrepancy`, Theorem `thm:approximate-proxy-bias-bound`.
- **그림 2:** 두 칸. 가로축은 남은 불균형, 세로축은 조정한 효과의 bias. 점 하나가 후보 표현 하나이고, 적합 없이
  모집단 값을 정확히 계산했다. (a)는 비워 둔 블록이 깨끗한 경우, (b)는 오염된 블록을 비워 둔 경우.
- **읽는 법:** (a)에서 점들이 원점으로 모인다. 불균형을 없애면 bias도 없어진다는 정리 그대로다. (b)에서는 불균형이
  0인 점의 bias가 1.44다. 오염된 블록을 비워 두면 균형 검사가 속는다.
- **논증:** (b)에서 일관성, 교환가능성, held-out 분리, 중첩, 정확한 균형이 모두 성립하는데 효과가 틀리므로,
  Theorem `thm:exact-proxy-identification`의 대우로 outcome-relevant completeness가 깨진다.

### Q3. 어느 블록이 오염됐는지 모르면 어떻게 피하는가

- **이론:** Theorem `thm:unknown-valid-block-identification`, Corollary `cor:target-majority-median`.
- **그림 3:** (a) 모의 자료 하나에서 균형 검사를 통과한 추정값 여덟. 깨끗한 held-out 선택 일곱은 1 근처, 오염된
  선택 하나는 $-0.43$. 그 여덟의 평균은 $0.86$, PROBE의 다수결은 $1.04$. (b) 모의 자료 30개에서 같은 비교.
- **읽는 법:** 오염된 선택도 균형 검사를 통과한다. 평균은 그 하나에 끌려가고 다수결은 끌려가지 않는다.
- **수치:** 오염된 선택이 검사를 통과한 자료는 30개 중 23개. 평균 절대오차는 평균 방식 $0.175$, PROBE $0.042$.
  과반이 성립하는 설계라 단순 중앙값도 $0.041$이다.

### Q4. 유한표본 이론은 실제 행동을 예측하는가

- **(a) 이론:** Theorem `thm:probe-search-error`. **그림:** 가로축은 시도한 held-out 선택의 수 $m$, 세로축은 추정값이
  참값에서 $\rho=0.20$ 안에 들어올 확률. 회색 선은 정리가 보장하는 하한, 파란 점은 정확히 풀리는 모형에서
  2,000번 돌려 관측한 비율. **읽는 법:** 파란 점이 항상 회색 선 위에 있다. 보장은 지켜지고 $m$과 함께 오른다.
- **(b) 이론:** Corollary `cor:learned-aipw-consistency`. **그림:** 가로축은 표본 크기, 세로축은 평균 오차, 로그-로그.
  **읽는 법:** 표본이 커지면 PROBE는 $0.056$에서 $0.024$로 줄고 $X$와 $W$를 모두 넣는 조정은 $0.2$ 근처에 머문다.

## 3. 부록으로 보낼 것

- SCM-1–5 성적표, SCM-6와 SCM-7의 mechanism 표(screen TPR/FPR, component 순도, learned target 오차)
- E11-v2 정확한 positive control의 상태별 수치
- root recovery와 선택된 $r$의 분포(진단, 관문 아님)
- SCM-6 v1 실패 진단과 수리 과정
- SCM-7 렌더러 검증, raw-CNN 비교군의 결과 head $R^2$, 판별 검사
- 계산 시간과 GPU 사용량

## 4. 아직 채워지지 않은 칸

| 항목 | 상태 |
|---|---|
| 그림 1의 이미지 열 | qualification 2개 자료로 미리보기. DeltaAI 확증 20개 진행 중 |
| 그림 4 왼쪽 | $m$-curve 계산 중(로컬, 2,000 반복) |
| SCM-6 $n=24{,}000$ | 10개 중 진행 중(로컬) |

## 5. 결정이 필요한 것

1. 그림 넷을 본문에 모두 넣을지, 그림 4를 부록으로 보낼지.
2. 5.tex의 현재 내용(Honest Approximate SCMs, Tracks A/B, WSC)을 이 구성으로 대체할지. 대체한다면 기존
   내용은 빨간 취소선으로 남기고 새 내용은 파랑으로 넣는다.

## 6. 첫 정리의 실험 증거 (2026-09-15 추가)

### 무엇을 보이나

Theorem `thm:exact-proxy-identification`은 미리 정한 블록 하나에 대한 정리다. 그래서 여러 held-out 선택을 탐색하고
다수결로 모으는 그림 1은 이 정리의 직접 증거가 아니다. 이 정리의 증거는 블록 하나를 고정하고, 그 블록을 균형 맞춘
표현으로 조정한 추정값이다.

### 자료와 재현 검증

- 봉인된 E12 확증(SCM-1, $n=12{,}000$, seed 20개)을 같은 fold와 같은 Brier 선택으로 다시 돌렸다. 저장된 PROBE 출력,
  기준선 넷, 균형 검사를 통과한 rotation의 추정값이 모두 **차이 0**으로 재현됐다. 봉인 뒤 `probe_structured_scm.py`가
  바뀌었지만 이 계산에는 영향이 없다는 뜻이다.
- 그 위에서 각 블록을 네 rotation 모두에서, 그 rotation이 고른 $r$로 평가했다. 균형 검사, 블록 탐색, 재적합은 없다.
- 그림에 쓸 블록($W_1$, $W_4$, 기준선 $X$)과 보고할 선택 여덟은 추정값을 계산하기 전에 스크립트 설명에 적었다.
- 코드 `code/run_t1_fixed_block_replay.py`, 결과 `results/t1_fixed_block_scm1_v1.json`,
  그림 `figures/section5/section5-thm1-fixed-block.pdf`와 `.png`(`code/make_section5_figures.py`의 `fig_thm1_fixed_block`).

### 블록 이름

원고 2장이 블록을 $W_1$부터 세므로, Section 5에서는 $W_1,W_2,W_3$을 깨끗한 블록, $W_4$를 오염된 블록,
$W_5=(I,C,M_0)$을 코드의 블록 0으로 부른다. 코드의 블록 $j=1,\dots,4$는 원고의 $W_j$와 같다.
2.0절 용어표의 $W_0,\dots,W_4$ 표기는 이 이름으로 바꿔 읽는다.

### 모집단 검산 (SCM-1, 적합 없이 계산)

표현 가족은 $Z_j=(X,C,\bar M_{-j}+rI)$이다. 각 블록을 정확히 균형 맞추는 $r$은 둘씩 있다.

| held-out | 탐색 범위 $[-1,1]$ 안의 균형 $r$ | 조정값 | 범위 밖 균형 $r$ | 조정값 |
|---|---:|---:|---:|---:|
| $W_1$ | $0.2245$ | $1.0000$ | $2.5755$ | $1.0000$ |
| $W_4$ | $-0.6502$ | $-0.4432$ | $2.5502$ | $0.9530$ |

두 균형 $r$에서 잔여 불균형은 모두 $10^{-16}$ 수준이다. 깨끗한 블록은 두 균형 표현이 모두 참값을 준다. 오염된 블록은
두 균형 표현이 서로 다른 값을 주고, 둘 다 참값이 아니다. 대우에 의해 $W_4$에서는 completeness가 깨진다.
나머지 깨끗한 선택 여섯도 두 균형 표현 모두 $1.0000$이다.

### 결과

| 조정 | 평균 | 표준편차 | 평균 절대오차 | E12 균형 검사를 네 rotation 모두 통과 |
|---|---:|---:|---:|---:|
| $X$만 | $0.035$ | $0.023$ | $0.965$ | |
| held-out $W_1$ | $1.009$ | $0.044$ | $0.038$ | 20개 중 15 |
| held-out $W_4$ | $-0.446$ | $0.026$ | $1.446$ | 20개 중 15 |
| $W_2$, $W_3$, 깨끗한 블록의 합집합 넷 | | | $0.024$–$0.046$ | 20개 중 14–19 |

오염된 $W_4$도 깨끗한 $W_1$만큼 균형 검사를 통과한다. 균형 검사만으로는 오염을 걸러낼 수 없으므로, 이것이 다음
정리 `thm:unknown-valid-block-identification`과 다수결의 동기가 된다.

### 원고 반영

- 5.tex 앞부분에 새 opening 세 문단과 새 소절 "Identification with a Prespecified Held-Out Block"을 파란색으로 넣었다.
- 옛 소절 "Questions, Comparators, and Reporting Rules"는 제목과 문단을 빨간 취소선으로 남겼다.
- 옛 소절 셋(Honest Approximate SCMs, Tracks A/B, WSC)은 그대로 두고, 차례로 바꿀 예정이라는 보라색 코멘트를 달았다.
- 원래 5.tex는 세션 scratchpad에 백업했다. 컴파일은 scratchpad 출력 폴더에서 했고 추적 중인 `main.pdf`는 건드리지 않았다.

### SCM-1–5로 넓힌 결과 (2026-09-15 추가)

- 봉인된 SCM family 확증(`restricted-brier-rotate4-g-confirm-v2`, $n=6{,}000$, SCM마다 자료 20개)의 적합을 그대로 썼다.
  fold를 다시 만들어 저장된 해시 100개가 모두 일치했고, 균형 검사를 통과한 rotation에서 저장된 추정값이 차이 0으로
  재현됐다. 표현은 다시 고르지 않고, 저장된 $r$로 AIPW만 계산했다.
- 코드 `code/run_t1_fixed_block_family.py`, 결과 `results/t1_fixed_block_scm_family_v1.json`.

| SCM | $X$ 조정 | $(X,W)$ 조정 | held-out $W_1$ 평균 (평균 절대오차) | held-out $W_4$ 평균 | 나머지 깨끗한 선택의 평균 절대오차 |
|---|---:|---:|---:|---:|---:|
| SCM-1 | $0.049$ | $0.815$ | $0.999$ ($0.048$) | $-0.442$ | $0.027$–$0.040$ |
| SCM-2 | $0.047$ | $0.816$ | $0.997$ ($0.048$) | $-0.443$ | $0.029$–$0.040$ |
| SCM-3 | $0.031$ | $0.811$ | $0.999$ ($0.050$) | $-0.471$ | $0.029$–$0.042$ |
| SCM-4 | $0.049$ | $0.800$ | $0.996$ ($0.053$) | $-0.443$ | $0.029$–$0.052$ |
| SCM-5 | $0.049$ | $0.814$ | $1.000$ ($0.048$) | $-0.440$ | $0.027$–$0.044$ |

- held-out $W_4$의 모집단 값은 SCM-1, SCM-2, SCM-4에서 $-0.443$, $-0.443$, $-0.442$이고 SCM-3에서 $-0.472$다.
  SCM-5는 관측이 sinh로 비선형이라, 같은 선형 계산으로 얻은 $-0.443$은 참고로만 둔다.
- 다섯 SCM 모두 같은 모양이다. 깨끗한 블록은 참값, 오염된 블록은 자기 모집단 값, $X$ 조정은 0 근처,
  $(X,W)$ 조정은 $0.8$ 근처다.
- 고차원 렌더링에는 블록별 AIPW 추정값이 균형 검사를 통과한 자료에만 남아 있다(벡터 30개 중 22–23, 이미지 20개 중 4–6).
  학습한 표현이 겨냥하는 값(네 rotation의 중앙값, 큰 새 표본에서 계산)은 모든 자료에 있다. 벡터에서 held-out $W_1$
  평균 $1.042$, $W_4$ 평균 $-0.461$이고, 이미지에서 $1.026$, $-0.436$이다.
- 그림 이름표는 원고 용어에 맞춰 "PROBE, held-out block $W_j$"로 바꿨고, 사용자 요청으로 $(X,W)$ 조정 줄을 더했다.
- 사용자 결정(2026-09-15): 첫 정리 그림은 SCM-1–5의 봉인 확증 자료 100개($n=6{,}000$)를 한 판에 모은다. opening도
  "같은 구조를 공유하는 다섯 SCM"으로 고치고, 벡터와 이미지 렌더링은 SCM-1에 대한 것으로 적는다. 모집단 계산에 기댄
  문장(정확한 균형 근과 조정값)은 SCM-1에 한정한다. SCM-1 $n=12{,}000$ 결과(`results/t1_fixed_block_scm1_v1.json`)는
  보조 기록으로 남긴다.


## 7. Section 5를 skeleton부터 다시 쓴다 (2026-09-15, 용한 결정)

- 그림 규칙: 모든 figure는 3–5칸이다. 한 칸짜리 그림은 wrapfigure로 둔다.
- 첫 정리 그림은 SCM마다 한 칸인 1×5로 그린다(`fig_thm1_fixed_block`). 다섯 SCM을 한 판에 모은 설계는 폐기했다.
- 5.tex는 정리 하나에 소절 하나와 그림 하나를 짝짓는 skeleton으로 다시 시작했다. 옛 원문은 끝에 빨간 취소선으로 남겼다.
- 아래는 skeleton 전에 쓴 파란 초안이다. 수락되지 않았으므로 5.tex에서는 뺐고, prose 단계에서 다시 쓴다.

```latex
\textcolor{blue}{We test the theory on data whose causal effect is known. Five structural causal models, SCM-1 to SCM-5, share one causal structure and the true effect $\tau=1$. SCM-2 to SCM-5 vary SCM-1 in the outcome noise, the treatment-effect heterogeneity, the proxy noise, and a nonlinear view of the proxy, respectively. The latent confounder $U$ is strong: the naive difference in means is about $-0.2$, the wrong sign, and adjustment for the observed covariate $X$ alone gives about $0.04$.}

\textcolor{blue}{The proxy has five blocks. Blocks $W_1$, $W_2$, and $W_3$ are noisy measurements of $U$. Block $W_4$ is a noisy measurement of $U$ shifted by $I$, an observed cause of treatment with no effect on the outcome. Block $W_5$ records $I$, an observed confounder $C$, and one more noisy measurement of $U$. Each measurement is one number, so the proxy has $7$ coordinates. For SCM-1 we also render each measurement as a $256$-dimensional vector or a $96\times96$ image, keeping $I$ and $C$ as numbers, which gives $1{,}282$ or $46{,}082$ coordinates with the same causal structure.}

\textcolor{blue}{Each subsection pairs one theorem with one figure. For every experiment, the seeds, sample sizes, learners, and pass criteria were sealed before its data were generated, and every criterion is reported.}

\subsection{\texorpdfstring{\textcolor{blue}{Identification with a Prespecified Held-Out Block}}{Identification with a Prespecified Held-Out Block}}
\label{sec:exp-fixed-block}

\textcolor{blue}{Theorem~\ref{thm:exact-proxy-identification} fixes one held-out block. If the block satisfies the theorem's assumptions, adjustment for a representation that balances it recovers $\tau$.}

\textcolor{blue}{With $7$ proxy coordinates, PROBE searches the representations $Z_j=(X,C,\bar M_{-j}+rI)$ over $r\in[-1,1]$, where $\bar M_{-j}$ averages the measurements outside the held-out block, after a rank transform in SCM-5. In SCM-1, each of $W_1$ and $W_4$ is balanced exactly at one $r$ in this range. Block $W_1$ satisfies the theorem's assumptions, with completeness from the additive-noise case of Proposition~\ref{prop:orc-sufficient}, and its balancing representation targets exactly $\tau=1$. Block $W_4$ meets every other condition of the theorem, including exact balance, yet its balancing representation targets $-0.443$. By the theorem, completeness must fail for $W_4$: because $W_4$ mixes $U$ with $I$, a representation can remove the treatment information in $W_4$ while $U$ still carries some.}

\textcolor{blue}{Figure~\ref{fig:thm1-fixed-block} pools the $20$ data sets of each of SCM-1 to SCM-5 from their sealed confirmation runs with $n=6{,}000$. Across the five models, held-out $W_1$ gives average estimates from $0.996$ to $1.000$ and mean absolute errors from $0.048$ to $0.053$; holding out $W_2$, $W_3$, or a union of clean blocks gives mean absolute errors from $0.027$ to $0.052$. Held-out $W_4$ gives average estimates from $-0.47$ to $-0.44$. Adjustment for $X$ gives $0.03$ to $0.05$, and adjustment for $(X,W)$ gives $0.80$ to $0.82$. Each block is evaluated alone, at the $r$ the sealed run selected, with no balance screen and no search over blocks.}

\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{../figures/section5/section5-thm1-fixed-block.pdf}
\caption{\textcolor{blue}{Each dot is one data set, $20$ from each of SCM-1 to SCM-5 ($n=6{,}000$). Dashed line: the true effect. Holding out the clean block $W_1$ recovers the effect. Holding out the contaminated block $W_4$ does not, although both blocks can be balanced.}}
\label{fig:thm1-fixed-block}
\end{figure}
```

### 상태 갱신 (2026-09-15)

- SCM-7-v4 이미지 확증이 관문 13개를 모두 통과했다(감사 메모 C.6). 4절 표의 "그림 1의 이미지 열" 칸은 이제 채워졌고, 그림 1은 v4 자료 15개로 다시 그렸다.
- skeleton 5.4 소절의 이미지 자료 메모를 "준비됨"으로 고쳤다.

## 8. 결정 기록 (2026-09-15)

- Theorem 1 그림 확정: `figures/section5/section5-thm1-fixed-block.pdf`(1×5, SCM-1–5, 줄은 $X$ 조정, $(X,W)$ 조정, held-out $W_1$, held-out $W_4$).
- 5.2는 Theorem 2 대신 Algorithm 1의 성능을 보인다. 대상은 합성 SCM, 고차원, 이미지 전부다. 후보 그림 A(1×3 추정값), B(1×5 합성), C(1×3 절대오차 로그 눈금)를 그려 검토한다.
- 사용자 결정(2026-09-15): 두 그림을 채택해 저장했다.
  - **Algorithm 1 성능 그림(1×3):** `figures/section5/section5-alg1-performance.pdf`와 `.png`, 코드 `code/make_alg1_performance_figure.py`.
    맨 윗줄은 각 렌더링의 $W$ 한 표본(생성기 seed 20260915)이다. 아래 점은 봉인 결과로, E12 SCM-1 $n=12{,}000$ 자료 20개,
    SCM-6-v3 확증 30개, SCM-7-v4 확증 15개다. 줄은 $X$ 조정, $X$와 $W$ 전부 조정, 균형 검사 없는 학습 요약, PROBE, oracle이다.
  - **PROBE 메커니즘 흐름도(1×4):** `figures/section5/probe-mechanism.pdf`와 `.png`, 코드 `code/make_probe_mechanism_figure.py`.
    (c)와 (d)는 SCM-7-v4 확증 자료 하나(seed 95300000)의 실제 값이다. (c)는 첫 rotation에서 두 critic의 검증 Brier 오차로,
    held-out $W_1$은 $0.2303\to0.2299$, held-out $W_5$는 $0.2478\to0.1422$다. (d)는 통과한 선택의 추정값과 반환값
    $\widehat\tau=0.9715$다. (a)와 (b)의 이미지는 같은 생성기로 그린 예시 한 사람이다. 위치는 Section 4.3 또는 논문 첫 그림 중에서 정한다.
  - **캡션에 적을 것:** 위 자료 출처, 그리고 (c)의 실제 판정은 다른 fold에서 신뢰상한으로 비교하며 네 rotation 모두 통과해야 남긴다는 규칙.

## 9. 2026-09-15 피드백 수용 기록

용한이 `5.tex`에 남긴 표시를 모두 처리했다.

### 빨간 표시 수용 (삭제 확정)

옛 Section 5 전체를 실제로 지웠다. 없어진 소절은 네 개다.

- Questions, Comparators, and Reporting Rules
- Honest Approximate SCMs (그림 `figures/final/figure_approximate_scms.pdf`)
- Exact-Assumption Tracks A and B (그림 `figures/final/figure_exact_tracks.pdf`)
- WSC Observational Benchmark (그림 `figures/final/wsc_benchmark.pdf`)

원문은 git HEAD의 `main/5.tex`에 그대로 있고, 그림 파일도 `figures/final/`에 남겨 두었다.
지운 본문을 참조하는 문장은 원고 어디에도 없었다(`WSC`, `Track A/B`, `cross-mask` 전수 검색).
이 삭제로 원고에서 실데이터 결과가 빠졌다. 되살리려면 git HEAD에서 꺼내면 된다.

### 대괄호 피드백 3건

1. `[(wrapfigure) - (draw a diagram instead of the placeholder)]`: placeholder 상자를 실제 인과 그림으로 바꿨다.
2. `[(Divided by paragraph... to choppy. Also no subsection at this point.)]`: 도입부의 Goal/Design/Reporting 굵은 라벨을 없애고 이어지는 산문 두 문단으로 다시 썼다. 같은 지적이 각 소절의 Claim/Theory/Evidence 라벨에도 그대로 해당하므로 다섯 소절 모두 산문으로 합쳤다.
3. `[(Goal 문단) - (delete)]`: Goal 문단을 지웠다.

### 새 그림: SCM 가족의 공통 구조

- 원본: `figures/2026-09-15-scm-family-structure.tex` (standalone TikZ), 산출물 `.pdf`.
- 스타일 출처: `figures/2026-08-27-proximal-balancing-causal-graphs.tex`와 동일한 node/edge 스타일, 색 `coltreat` `#0066CC`(처치), `colout` `#D62828`(결과), `collatent` `#A6A6A6`(숨은 변수).
- 구조 출처: `code/probe_structured_scm.py`의 구조식 주석과 `generate_role`.
  - $U\to X, W_{1:3}, W_4, W_5, A, Y$
  - $I\to W_4, W_5, A$ ($I$는 $Y$로 가지 않는다)
  - $C\to W_5, A, Y$
  - $X\to A, Y$, $A\to Y$
- 깨끗한 블록 셋은 `3.tex`가 이미 쓰는 표기 $W_{1:3}$로 묶어 선을 줄였다.

### 레이아웃 판정

- wrapfigure `[12]`, 폭 `0.40\linewidth`. 그림과 캡션이 19쪽 안에 들어가고, 20쪽으로 좁은 줄 하나만 이어진다.
- 절 끝의 `\clearpage`를 뺐다. Section 5는 19–21쪽, Section 6은 22쪽에서 시작한다.
- 컴파일 오류 0, 미정의 참조 0.

### 백업

- 수정 직전 `5.tex`: 세션 scratchpad의 `5.tex.before-feedback`.

### 9a. 구조 그림 재설계 (2026-09-15, 용한 피드백 2차)

첫 시도는 아홉 변수를 한 판에 그린 단일 그림이었다. 용한의 판정은 두 가지였다.
$W_1$부터 $W_5$, $C$, $I$가 무슨 뜻인지 그림만 봐서는 알 수 없고, Figure 1을 기본 template으로 쓰라는 것이다.

그래서 Figure 1(`2026-08-27-proximal-balancing-causal-graphs`)의 판형을 그대로 따랐다.

- panel 세 개, 각 panel은 proxy 블록 한 종류가 참여하는 구조만 보여준다. 골격 $U\to X,A,Y$, $X\to A,Y$, $A\to Y$는 세 panel이 공유한다.
- (a) $W_{1:3}$는 $U$의 깨끗한 측정, (b) $W_4$는 $I$가 밀어놓은 측정, (c) $W_5$는 $I$와 $C$를 함께 담은 블록이다.
- panel 아래 두 줄 라벨과 caption이 $W$, $I$, $C$의 뜻을 직접 적는다.
- node style, 색, arrowhead, `(a)`~`(c)` 라벨, `width=0.86\textwidth`, 공유 caption을 Figure 1과 같게 맞췄다.
- panel마다 $I$는 블록 바로 위, $C$는 블록 바로 아래에 두어 `$I$가 $W_4$를 민다`, `$W_5$가 $I$와 $C$를 담는다`가 눈으로 보이게 했다. 선 교차는 0이다.

wrapfigure 대신 3칸 figure가 되면서 19쪽의 좁은 줄 문제도 사라졌다. Section 5는 19–22쪽, Section 6은 23쪽에서 시작한다.

### 9b. 구조 그림 3차: $I$, $C$ 기호를 $W_j$로 흡수 (2026-09-15, 용한 지시)

용한의 지적은 두 단계였다. 첫째, $I$와 $C$를 관측 변수로 그리려면 실제로 adjusting 또는 representing 해야 하고,
아니면 $U$에 흡수해야 한다. 둘째, 직접 쓰는 변수라면 그것은 곧 proxy $W$의 일부이므로 $I$, $C$ 같은 별도 기호 대신
$W_j$ 번호로 흡수해야 한다.

첫째 질문의 답은 코드에 있다. 세 구현이 같은 representation을 쓴다.

- `probe_structured_scm.py:312`, `probe_highdim_w.py:173`, `e8_population.py:66`
- $Z_S = (X,\ C,\ \bar M_{-S} + r\,I)$ where $\bar M_{-S}$는 유지된 블록들의 측정 평균이다.
- 즉 $C$는 좌표로 직접 들어가고, $I$는 균형 탐색이 고르는 가중치 $r$로 들어간다.
- 블록 0(원고의 $W_5$)이 held-out이면 $Z_S=(X,\ \bar M_{-S})$이고 둘 다 없다. 그 선택은 균형 검사를 통과하지 못한다
  (SCM-1 모집단에서 균형 가능한 8개 선택에 블록 0이 없고, 이미지 확증에서 S0는 0.2478 → 0.1422로 탈락했다).

그래서 둘을 $U$로 흡수하는 것은 틀리다. 대신 둘째 지시를 따라 기호를 없앴다.

- 그림은 블록 단위 그래프가 됐다. panel (a) $U\to W_{1:3}$, (b) $U\to W_5$와 $W_5\to A$, $W_5\to Y$ 추가,
  (c) $W_5\to W_4$와 $U\to W_4$ 추가. panel (c)가 전체 구조다.
- $W_5$의 두 좌표가 하는 일은 caption과 본문이 말로 적는다. 기호는 쓰지 않는다.
- `5.tex`에서 `$I$`와 `$C$` 표기를 모두 제거했다. 검색으로 확인했다.

한 가지 제약을 남긴다. 코드의 held-out 선택은 블록 다섯 개에 대한 부분집합이고 블록 0은 세 좌표가 한 덩어리로
묶여 있다. 그래서 두 좌표를 $W_6$, $W_7$ 같은 별도 최상위 블록으로 승격하면 탐색 공간 서술이 실제 실험과 달라진다.
지금 그림은 블록 다섯 개 구조를 그대로 지킨다.

## 10. 블록 분할을 원고 수준으로 되돌린 재실험 (2026-09-16)

### 왜 다시 돌리나

용한의 질문은 "왜 원고에 충실하게 실험하지 않았나"였다. 원고와 코드를 대조한 결과는 다음과 같다.

- Algorithm 1은 블록 분할 $W=(W_1,\dots,W_J)$를 입력으로 받고 후보 수가 $T=2^J-2$다. 옛 코드는 $J=5$, split 30개였고 $2^5-2=30$이다. **열거는 원고와 어긋나지 않았다.**
- 3.tex는 $\Phi_j$를 "prespecified class"로 둔다. 옛 코드의 $Z_S=(X,C,\bar M_{-S}+rI)$도 미리 정한 class이므로 허용 범위 안이다.
- 진짜 문제는 두 가지다. Section 5가 블록 구성과 $\Phi_j$를 **한 번도 적지 않았고**, 옛 분할이 교란 변수를 측정값과 한 블록에 묶어 두어 **살아남는 split이 언제나 교란 변수를 갖고 있었다.** 탐색이 "교란 변수는 남겨야 한다"를 스스로 발견할 기회가 없었다.

### 새 설계

- 관측 proxy 좌표 하나가 블록 하나다. $W_1,W_2,W_3$는 깨끗한 측정, $W_4$는 오염된 측정, $W_5$는 측정 하나 더, $W_6$은 처치만 일으키는 좌표, $W_7$은 교란 좌표다. $J=7$, $T=126$.
- 예산은 Algorithm 1 1행 그대로 $m=30$을 균등 비복원 추출한다(`budget_permutation(seed_base+7)`의 앞 30개).
- 한 replicate에서 126개를 모두 계산한 뒤 중첩된 앞부분을 읽어 $m\in\{5,10,20,30,60,126\}$ 곡선을 함께 낸다. 계산은 한 번이고, 주 결과는 $m=30$ 하나로 미리 못박았다. 곡선은 관문 없이 무조건 보고한다.

### 봉인 전 개발 진단 (seed 777000, 확증 seed 아님)

| 범주 | 통과/전체 |
|---|---|
| clean | 15 / 15 |
| bad_singleton ($W_4$만 held-out) | 1 / 1 |
| mixed | 0 / 15 |
| drops_cause | 0 / 32 |
| drops_confounder | **17 / 63** |

교란 변수를 버린 split 17개가 균형 검사를 통과한다. 즉 검사만으로는 걸러지지 않고 집계 규칙이 실제로 일을 해야 한다.
유지된 33개 중 참값을 겨냥하는 것은 15개로 **과반이 아니다**. 그래서 중앙값 따름정리의 과반 조건은 성립하지 않고,
가장 큰 동의 성분 규칙이 작동해야 한다. 옛 분할보다 훨씬 강한 시험대다. 예산 곡선은 1.0143, 0.9681, 0.9800, 0.9801, 0.9809, 0.9775였다.

### 봉인

- 규약 `results/probe_structured_scm_blockwise_j7_confirm_protocol_v1.json`, 소스 스냅샷 `..._v1_source/`.
- seed_start 3100000은 이 프로젝트의 어떤 실행에서도 쓰인 적이 없다. 관문은 옛 규약과 동일하다.
- 봉인 직후 한 번 수정했다. CLI가 비교하는 `mode_scope` 키가 빠져 다섯 실행이 모두 검증에서 멈췄고 결과 파일은 하나도 만들어지지 않았다. 그 키만 더했고 seed, 명세, 예산, 관문, 소스 해시는 그대로다.

### 실측 시간

- 구조 SCM: 옛 분할 seed 하나 36.6초, 새 분할 전수 126개 계산 164초. 5개 SCM × 20 seed를 5-way 병렬로 약 1시간.
- 고차원 SCM-6: seed 하나 약 520~735초. 재실행은 밤 단위 작업이다.
- 이미지 SCM-7: 15 seed에 GPU 3시간 3분. DeltaAI 10시간 예산 중 46분만 남아 지금은 불가능하다.

용한의 결정으로 이번에는 CPU 실험만 먼저 돌린다.

### 10a. 7블록 확증 결과 (2026-09-16, 규약 blockwise-j7-brier-rotate4-confirm-v1)

판정은 **FAIL**이다. 다섯 설정 중 하나만 모든 관문을 통과했다.

| 설정 | 평균 | MAE | p90 | 반환율 | vs $X$ | vs $(X,W)$ | oracle 초과 | 관문 |
|---|---|---|---|---|---|---|---|---|
| SCM-1 | 0.9638 | 0.0443 | 0.0891 | 1.00 | -0.8828 | -0.1301 | 0.0512 | FAIL |
| SCM-2 | 0.9621 | 0.0463 | 0.0972 | 1.00 | -0.8826 | -0.1271 | 0.0492 | PASS |
| SCM-3 | 0.9638 | 0.0443 | 0.0899 | 1.00 | -0.8997 | -0.1337 | 0.0517 | FAIL |
| SCM-4 | 0.9597 | 0.0465 | 0.1045 | 1.00 | -0.8815 | -0.1435 | 0.0541 | FAIL |
| SCM-5 | 0.9615 | 0.0489 | 0.0974 | 1.00 | -0.8794 | -0.1267 | 0.0551 | FAIL |

걸린 관문은 거의 전부 oracle 초과분이다. 기준 0.05에 대해 0.0492부터 0.0551이다. SCM-4는 p90도 0.1045로 0.10을 넘겼다.
MAE, 반환율, $X$ 대비, $(X,W)$ 대비는 다섯 설정 모두 통과한다.

원인은 회계에 그대로 드러난다. 주 예산 $m=30$ 기준 합계다.

| 범주 | 유지된 수 | 반환된 성분 안에 든 수 |
|---|---|---|
| clean | 273 | 273 |
| bad_singleton | 33 | 0 |
| mixed | 0 | 0 |
| drops_cause | 0 | 0 |
| drops_confounder | 177 | **169** |

교란 좌표를 버린 split 177개가 검사를 통과했고, 그중 169개가 반환된 성분 안으로 들어갔다. 숫자로 보면 이렇다.

| SCM | $\rho$ | 반환값 | clean만의 중앙값 | 교란 버린 것만의 중앙값 | $|\widehat\tau-\tau|\le\rho$ 비율 |
|---|---|---|---|---|---|
| SCM-1 | 0.0848 | 0.9638 | 0.9852 | 0.8596 | 85% |
| SCM-2 | 0.0881 | 0.9621 | 0.9834 | 0.8566 | 85% |
| SCM-3 | 0.0868 | 0.9638 | 0.9854 | 0.8588 | 85% |
| SCM-4 | 0.0875 | 0.9597 | 0.9809 | 0.8518 | 85% |
| SCM-5 | 0.0839 | 0.9615 | 0.9869 | 0.8594 | 85% |

교란 좌표를 버린 split의 추정값 0.86과 깨끗한 split의 0.985는 0.127만큼 떨어져 있는데, 연결 기준은 $2\rho\approx0.17$이다.
그래서 두 무리가 **하나의 성분으로 이어지고**, 반환된 중앙값이 0.985에서 0.962로 끌려 내려간다.

정리와의 관계는 이렇다. Theorem 6이 약속하는 것은 $\rho$ 안의 반환이고, $\rho\approx0.085$에 대해 85%의 replicate가 그 안에 든다.
즉 이번 실패는 정리 위반이 아니라, 옛 설계(교란 좌표가 언제나 유지되던 설계)에 맞춰 잡아 둔 프로젝트 자체 관문을 아슬아슬하게 못 넘긴 것이다.

예산 곡선(관문 없음, 다섯 설정 평균):

| $m$ | 반환율 | MAE | p90 | 평균 유지 수 |
|---|---|---|---|---|
| 5 | 0.44 | 0.0504 | 0.1037 | 0.7 |
| 10 | 0.65 | 0.0515 | 0.1177 | 1.4 |
| 20 | 0.90 | 0.0461 | 0.0962 | 3.1 |
| 30 | 1.00 | 0.0460 | 0.0956 | 4.8 |
| 60 | 1.00 | 0.0468 | 0.0985 | 9.6 |
| 126 | 1.00 | 0.0400 | 0.0924 | 21.1 |

주 결과는 봉인대로 $m=30$이다. $m=126$이 더 좋아 보인다는 이유로 주 결과를 바꾸지 않는다.

산출물: `results/probe_structured_scm_blockwise_j7_confirm_v1_g{1..5}.json`, 채점 `results/probe_structured_scm_blockwise_j7_confirm_summary_v1.json`, 채점기 `code/score_blockwise_j7.py`.

### 10b. 진단 그림: 실패 경로가 닫혔다 (2026-09-16, 새 시뮬레이션 없음)

`figures/diagnostics/j7-screen-and-linking.pdf`, 생성 코드 `code/make_j7_diagnostic_figure.py`.
세 panel 모두 봉인된 결과 파일에서만 읽었다. 주 예산 $m=30$이 뽑은 후보 3,000개, 유지된 비-clean 후보 186개, replicate 100개다.

**(a) 검사는 무엇을 통과시켰나.** 통계량은 rotation 네 개 중 최악의 $\text{signed gap}/(2\,\mathrm{se})$이고, 통과 경계는 1이다.

| 후보 유형 | 중앙값 | p10 | 통과율 |
|---|---|---|---|
| clean | 0.50 | -0.07 | 85.3% |
| $W_4$ out | 0.51 | 0.14 | 94.3% |
| mixed | 2.39 | 1.97 | 0% |
| $W_6$ out (처치 좌표) | 8.58 | 8.11 | 0% |
| $W_7$ out (교란 좌표) | **2.81** | **0.97** | **11.3%** |

교란 좌표를 버린 후보의 통계량 중앙값은 경계의 2.8배다. 즉 검사는 이 위반을 대체로 **잡아낸다**.
통과한 11%는 아래 꼬리가 경계를 넘어온 것이다. 설계상 맹점이 아니라 유한표본 검정력 문제다. 다섯 SCM의 중앙값이 서로 포개진다.

**(b) 왜 이어졌나.** 유지된 후보마다 가장 가까운 clean 후보와의 거리를 $2\rho$로 나눈 값이다. 1 이하면 연결된다.

| 후보 유형 | 중앙값 | 1 이하 비율 |
|---|---|---|
| $W_4$ out | 8.10 | **0%** |
| $W_7$ out | 0.22 | **94.8%** |

오염 블록을 버린 후보는 깨끗하게 분리되고, 교란 좌표를 버린 후보는 거의 전부 연결된다.
선택 효과가 겹쳐 있다. 검사를 통과한 교란-버림 후보는 곧 추정값이 clean 무리에 가장 가까운 것들이다.

**(c) 결과를 얼마나 끌어내렸나.** 반환된 성분 안의 교란-버림 후보 비율과 $\widehat\tau-1$의 상관은 $-0.32$다.

| 성분 내 비율 | replicate 수 | 평균 오차 |
|---|---|---|
| 0 | 45 | -0.0206 |
| (0, 0.34) | 11 | -0.0652 |
| [0.34, 0.67) | 35 | -0.0473 |
| [0.67, 1] | 9 | -0.0537 |

비율이 0인 replicate도 -0.021만큼 낮다. 교란 오염을 없애도 오차가 0이 되지는 않는다. 대략 절반으로 준다.

**판정.** math-theory-agent의 결정 규칙에 따르면 전자의 경우다. 검사 구현이나 통과 방향의 결함이 아니라 경계에서의 검정력 문제다.
따라서 권고 경로가 성립한다. 이번 $J=7$ 결과는 stress test로 보존하고, 알려진 관측 교란 좌표를 $X$로 옮긴 $J=6$ 규약을 새 seed로 봉인한다.

## 11. 설계를 원고 변수로 되돌린다 (2026-09-16, 용한 + math-theory-agent 지시)

### 중단과 보존

$J=7$ 실행은 여기서 멈춘다. 결과 다섯 개와 채점, 진단 그림은 그대로 둔다.
지위는 **잘못 지정된 $X/W$ 분할에 대한 negative-control stress test**다. 본문 핵심 결과가 아니라 appendix의 failure analysis 자리다.
$C$만 $X$로 옮기는 $J=6$은 바로 돌리지 않는다. 그렇게 해도 $I$, 특수 블록, 역할을 아는 encoder가 남는다.

### 무엇이 잘못이었나

원고의 관측 자료는 $(X,W,A,Y)$이고 잠재변수는 $U$다. 그런데 시뮬레이션은 역할이 미리 알려진 두 좌표를 $W$ 블록으로 만들고,
`CAUSE_BLOCK`과 `CONFOUNDER_BLOCK`이라는 이름으로 표현 코드가 그 역할을 알고 다르게 처리했다.
그래서 실험이 "데이터에서 좋은 표현을 배우는가"가 아니라 "잘못 지정된 $X/W$ 분할을 복구하는가"를 추가로 풀었다.

### 새 설계 (모집단에서 확인함)

관측: $X=(X_1,X_2)$, $W=(W_1,\dots,W_J)$, $A$, $Y$. 잠재: $U$(교란), $I$(처치만 일으키는 원인, 관측되지 않음).

$$X_1 = U + 2\eta_1,\quad X_2=\eta_2,\quad W_j = U + \sigma_j\varepsilon_j + a_j I$$
$$P(A=1\mid\cdot)=0.1+0.8\,\Phi(U+\gamma I+0.2X_1+0.3X_2),\quad Y=A(1+hU)-cU+0.2X_1-0.5X_2+\text{noise}$$

표현 class는 역할을 모른다. $\Phi_S=\{(x,w)\mapsto (x_1,x_2,\beta^\top w_{-S}) : \lVert\beta\rVert=1\}$.
어떤 좌표가 깨끗한지, 오염됐는지, 쓸모없는지 class 안에 들어 있지 않다.

블록 구성: $W_1,W_2,W_3$은 $U$를 재고, $W_4,W_5$는 $U$를 재지만 잠재 $I$에 밀려 있으며, $W_6,W_7$은 $U$ 정보를 담지 않는다.

모집단 계산 결과($\sigma=0.85$, $a=(0,0,0,-0.6,0.4,0,0)$, $\gamma=2.5$, $c=2.5$, $h=0$):

| held-out 유형 | 균형점 $\beta$ | 표적 |
|---|---|---|
| 정보 있는 깨끗한 블록 | 13개 발견 | **전부 정확히 1.0** |
| 깨끗한 블록 둘 | 11개 | **전부 1.0** |
| $I$에 밀린 블록 | **없음** | 균형 불가, 검사가 탈락시킨다 |
| 정보 없는 블록만 | 아무 $\beta$나 ($D^2\approx2\times10^{-16}$) | **-0.23, -0.10, 0.74 등 흩어짐** |
| 정보 있는 블록 + 정보 없는 블록 | 12개 | **1.0** |

이것이 Theorem 1의 내용 그대로다. 균형에 completeness가 더해지면 어떤 균형 표현을 골라도 표적이 $\tau$다.
held-out 블록이 정보를 담지 않으면 균형 조건이 공허해지고 표적이 임의가 된다. 학습기는 어느 쪽인지 모른다.

$J=7$일 때 예상 census: 균형 가능 31개 중 28개가 $\tau$를 겨냥한다(90.3%). $W_4$ 또는 $W_5$를 포함한 95개는 균형 불가다.
공허한 3개의 표적은 서로도 흩어지므로 경쟁 성분을 이루지 못한다. 분리는 균일 $\beta$ 기준 $|0.736-1|=0.264$이고 직전 실행의 $2\rho\approx0.17$보다 크다.

다음 단계는 126개 split 전수 census(약 30분, seed 소모 없음)와 그 뒤의 fresh-seed 최소 확증 계획 봉인이다.

### 11a. 11절의 주장 철회 (2026-09-16)

11절의 "모집단 계산 결과"는 성립하지 않는다. 네 가지가 틀렸다.

1. 식 $W_j=U+\sigma_j\varepsilon_j+a_jI$에는 $U$ loading이 없는데 설명은 일부 블록이 $U$ 정보를 담지 않는다고 했다. 식과 설명이 모순이다.
2. 126개 split 전수 census artifact가 없다. 조합 계산만 있었다.
3. 목적함수가 평평한 split에서 $\beta$ tie-break가 정의되지 않았으므로, 보고한 $-0.23$, $-0.10$, $0.74$는 Algorithm 1의 candidate value가 아니다.
4. separation 기준은 $2\rho$가 아니라 $4\rho$다. $0.264<4\rho\approx0.34$이므로 통과하지 못한다.

또한 정보가 전혀 없는 블록은 minimizer 집합이 구면 전체가 되어 $\theta_S$ 범위가 $\tau$를 포함할 수 있고 $g_{\min}=0$이 된다.
따라서 그 장치를 설계에서 뺀다. 잠재변수는 $U=(U_{\mathrm c},U_{\mathrm t})$ 벡터 하나로 쓰고 별도 $I$ 표기를 없앤다.

수정된 계획은 `memo/2026-09-16-population-census-prd-v1.md`에 있다. 그 검토가 끝나기 전에는 fresh seed를 쓰지 않는다.

### 11b. census PRD v2 제출 (2026-09-16)

PRD v1은 FIX REQUIRED 판정을 받았다. 열두 승인 조건을 반영한 v2는 `memo/2026-09-16-population-census-prd-v2.md`다.
핵심 변경: 연속 구면 대신 유한하고 치환에 닫힌 $\{-1,0,1\}^d$ library, 임의 tie-break 대신 동점 집합 보존과 불일치 시 FAIL,
$D_S^2$와 $\theta_S$의 완전한 정의, 인증 구간과 보수적 병합에 의한 plurality, 새 DGP 전용 $\rho$(theorem arm과 operational arm 분리),
모든 후보 쌍의 $4\rho$ 분리, 16 cell ledger, 전체 파이프라인 치환 검사. 검토 전에는 구현도 실행도 하지 않는다.

### 11c. census PRD v3 제출 (2026-09-16)

v2도 FIX REQUIRED였다. v3는 `memo/2026-09-16-population-census-prd-v3.md`다. 핵심 변경: ORC 라벨을 $e_{\mathrm c}\in\operatorname{rowspan}(B_S)$로 교체,
exact arm을 주 arm으로 택하고 $D_S^2=0\iff B_S\gamma(\beta)=0$의 닫힌 특성으로 target family와 wrong family의 지정 library 원소가 정확한 균형점이 되도록
$\alpha_{\mathrm t}$와 $b_{6t}=b_{7t}$를 푸는 exact-hit cell 구성, $p_\tau$를 ORC 인증 target class 비율로 정의, 반경을 원고 Section 4의 유한표본 공식으로 교체,
"certified interval"을 수치 안정 구간으로 격하, split 내부 selection margin $m_S$ gate, ternary alphabet 하나로 고정. 검토 전에는 구현도 실행도 하지 않는다.

### 11d. PRD v4: positive-control audit로 단순화 (2026-09-16)

math-theory-agent의 지적대로 v3는 census 체계 자체가 목표가 되어 있었다. v4는 `memo/2026-09-16-population-census-prd-v4.md`이며 통제 문장 하나로 묶인다.
SCM 하나(격자 없음, $h=0$), population preflight 한 번(정확한 균형은 $B_S\gamma(\beta)=0$으로 대수 판정, 세 사실만 확인, 통과 시 즉시 동결),
fresh seed에서 Algorithm 1 문자 그대로 실행(정확한 nuisance, $m=T=126$ 주 설정, 실행마다 $\widehat{\mathcal B}$로 $N,\Delta,g_{\min},\rho$ 재계산, 세 조건 판정, 성공률 대 하한).
보탠 것은 넷뿐이다: 치환 단위 테스트, $m=T$ 주 설정, $\sigma_{S,\beta}$ preflight 표, 실패 시 두 줄 기록. 검토 전에는 구현도 실행도 하지 않는다.

### 11e. PRD v4 구현 1단계: population preflight PASS (2026-09-16)

코드 `code/positive_control_scm.py`, `code/run_positive_control_preflight.py`, `code/test_positive_control.py`.
산출물 `results/positive_control/preflight_v1.json`. 테스트 11개 통과.

**푼 계수.** 목표족은 $\gamma_c$가 $\alpha_t$에 affine이므로 닫힌 식으로 $\alpha_t=-1.2225$를 얻었다.
오답족은 지정 원소 $W_4+W_5-W_7$에서 $\gamma_t=0$을 bracket 근으로 풀어 $b^\circ=3.1011054687$을 얻었다.
$b^\circ$는 목표족 지정 원소 $W_4+W_5$의 $(p,q,v)$에 들어가지 않으므로 목표족 균형을 흔들지 않는다(코드로 확인).

**126 split census.** population candidate는 9개다.

| 부류 | split | ORC | 균형 원소 수 | $\theta$ |
|---|---|---|---|---|
| target | $\{W_1\},\{W_2\},\{W_3\}$와 그 합집합 7개 | valid | 각 1 | **정확히 1.00000** |
| wrong | $\{W_6\}$, $\{W_7\}$ | invalid | 각 1 | 0.35877 (한 값) |

나머지 117개는 균형 원소가 없고 최소 잔차가 $8.8\times10^{-2}$ 이상이다. 균형/비균형 간격이 12자릿수다
(균형점 $\lVert B_S\gamma\rVert\approx10^{-16}$, $D^2\approx6\times10^{-16}$; 비균형 최소 $D^2=5.3\times10^{-4}$).
후보마다 균형 원소가 정확히 하나이므로 동점 모호성이 없다.

**세 사실.** $p_\tau=7/9=0.7778$, $p_\star=2/9=0.2222$, $\Delta=0.5556>0$. wrong class $L=1$. $g_{\min}=0.64123$.
치환 검사 최대 차이 $3.7\times10^{-16}$.

**반경.** AIPW score 표준편차는 목표 후보 3.5995, 오답 후보 5.3860이다($10^6$ 표본, seed 8100000, 확증 구간과 분리).
원고 Section 4 식 $\rho=\sigma_{\max}\sqrt{N/(n_{\mathrm E}\delta)}$, $N=9$, $\delta=0.05$에서
$4\rho<g_{\min}$이 되는 최소 평가 표본은 $n_{\mathrm E}=203{,}191$이다. Chebyshev형이라 보수적이며, 이것이 사실이다.

$m=T$일 때 원고 하한은 $1-\delta-L e^{-N\Delta^2/2}=0.95-e^{-1.389}=0.7006$이다.

세 사실이 모두 성립하므로 PRD 3.4에 따라 **DGP를 동결한다.**

### 11f. 구현 2단계: Brier screen 개발 연구와 설계 수정 (2026-09-16)

편공분산 screen 제안은 거절됐다. 원고의 Brier-risk screen $\widetilde D_S^2=\max\{\widehat R_{0,S}-\widehat R_{1,S},0\}$으로 구현했다.
critic은 probit-linear이고, augmented가 base를 포함하도록 lifted start를 넣어 통계량이 구성상 음수가 되지 않는다.

**첫 개발 실행에서 드러난 결함.** $n_D=5{,}000$에서 $\beta$ 복원이 0.5/9였다. 원인은 split 내부 selection margin
$m_S=\min_{\beta\notin\mathcal T_S}D_S^2-\min_{\beta\in\mathcal T_S}D_S^2$이다. PRD v3에 있다가 v4에서 빠진 항목이다.
ternary 전체 library(단일 split에서 364개)에서 $m_S$는 단일 split 기준 $2.3\times10^{-6}$이고, 유한표본 잡음 바닥 $\sim10^{-4}$보다 작다.
선택이 모집단 최소점을 놓치면 학습된 표현의 $\theta_S$가 $\tau$에서 미세하게 벗어나고, 그러면 정리의 분리 조건
$4\rho<\min|c-c'|$이 목표 split끼리도 깨진다. 실제로 $n_D=100{,}000$에서 5회 중 1회가 0.0285만큼 벗어났다.

**치료.** 잡음을 줄이는 대신 margin을 키운다. $(\alpha_c,\alpha_t)$ 확대는 probit 포화 때문에 2배밖에 못 키운다.
효과적인 지렛대는 library 밀도다.

| library | 크기($d=6$) | $m_S$ 단일 | $m_S$ 쌍 | $m_S$ 삼중 |
|---|---|---|---|---|
| ternary 전체 | 364 | $2.3\times10^{-6}$ | $7.0\times10^{-5}$ | $8.9\times10^{-4}$ |
| 비영 2개 이하 | 36 | $8.0\times10^{-5}$ | $1.2\times10^{-4}$ | $8.9\times10^{-4}$ |
| **비영 1개** | **6** | $\mathbf{2.3\times10^{-3}}$ | $3.4\times10^{-3}$ | $4.0\times10^{-3}$ |

표현 class를 "유지된 블록 하나를 channel로 쓴다"로 고정했다. $\Phi_S=\{(x_1,x_2,w_j):j\notin S\}$이며 치환에 닫혀 있고 역할 정보가 없다.
이 class에서는 계수 풀이도 독립적이다. 블록 3이 목표족의 균형 channel이고 블록 4가 오답족의 균형 channel이므로,
$\alpha_t$는 $\gamma_c(e_3)=0$에서, $b_{4t}$는 $\gamma_t(e_4)=0$에서 서로 영향 없이 풀린다.

**동결된 DGP.** $J=7$, $b_{\cdot c}=(1,1,1,1,1,0,0)$, $b_{\cdot t}=(0,0,0,-0.6,-3.433555,1,1)$, $\sigma=0.85$,
$\kappa=2$, $\alpha_c=1$, $\alpha_t=-1.804167$, $c=2.5$, $s=0.3$, $\tau=1$.

**모집단.** candidate 10개다. 목표 7개(블록 1~3의 공집합 아닌 부분집합, $\theta$가 정확히 1), 오답 3개($\{W_6\},\{W_7\},\{W_6,W_7\}$, $\theta=0.35358$ 한 값).
$p_\tau=0.70$, $p_\star=0.30$, $\Delta=0.40$, $L=1$, $g_{\min}=0.64642$. 치환 검사 $3.5\times10^{-18}$. 다섯 사실 모두 PASS.
$\sigma_{\max}=5.9345$이고 $4\rho<g_{\min}$이 되는 최소 평가 표본은 $n_{\mathrm E}=269{,}710$이다.

**개발 사다리(개발 seed 8200000 구간, 5회 반복).** $\beta$ 복원은 $n_D=2{,}000$부터 모두 10/10이다.

| $n_D$ | 의도한 후보 목적함수 최대 | 나머지 116개 최소 | $t\in[10^{-4},2\times10^{-3}]$ |
|---|---|---|---|
| 2,000 | $5.6\times10^{-4}$ | $1.8\times10^{-3}$ | $t=10^{-3}$에서만 정확히 10 |
| 5,000 | $5.7\times10^{-4}$ | $1.8\times10^{-3}$ | $t=10^{-3}$에서만 정확히 10 |
| **20,000** | $7.0\times10^{-5}$ | $2.0\times10^{-3}$ | **전 구간에서 정확히 10, $\Delta=0.400$** |

가장 작은 안정적 값으로 $n_D=20{,}000$, $t=5\times10^{-4}$를 고른다. $t$는 의도한 최대의 7배이고 나머지 최소의 1/4이다.
$m=T=126$일 때 원고 하한은 $0.95-e^{-10\cdot0.16/2}=0.5007$이다.

### 11g. 구현 3단계: positive-control 확증 완료 (2026-09-16)

규약 `results/positive_control/confirmation_protocol_v1.json`(봉인 뒤 runner 해시만 추가, 그 시점에 확증 seed 미사용).
결과 `results/positive_control/confirmation_v1.json`. runner `code/run_positive_control_confirmation.py`.

**구조.** 원고 정리는 학습 표본에 조건부다. 그래서 독립 학습 상태 5개를 만들고, 상태마다 screen과 retained set을 한 번 고정한 뒤
평가 표본만 200회 다시 뽑았다. 학습 seed는 5300000 구간, 평가 seed는 5400000 구간이다.

**모든 상태에서 같은 결과가 나왔다.**

| 항목 | 값 |
|---|---|
| $N$ | 10 (모집단 후보와 정확히 일치) |
| $\Delta$ | 0.4000 |
| $L$ | 1 |
| $g_{\min}$ | 0.64642 |
| $\rho$ | 0.16135 |
| $4\rho$ | 0.64539 |
| 세 조건 | plurality PASS, separation PASS, retained set 일치 PASS |
| 반환 | 200/200 unique largest (동점 0, 무반환 0) |
| uniform radius | 1.000 |
| 성공률 | **1.000** |
| 원고 하한 | **0.5007** |

선택된 $\beta$는 열 후보 모두 모집단 균형점과 일치했다. screen 분리는 유지된 것의 최대 $2.5\times10^{-5}$ 대 탈락한 것의 최소 $2.9\times10^{-3}$로 116배다.
1,000회 평가에서 평균 오차 0.00069, 최대 절대오차 0.02327이며 모두 $\rho$ 안에 들었다.

**정직하게 남길 것 둘.**
첫째, 분리 조건의 여유가 얇다. $4\rho=0.64539$ 대 $g_{\min}=0.64642$로 0.001 차이다. 봉인한 $n_{\mathrm E}=270{,}000$이 최소값 269,710 바로 위이기 때문이다.
조건은 다섯 상태 모두에서 성립했지만, 후보가 하나라도 더 유지되면 $N$이 커져 $\rho$가 오르고 조건이 깨질 수 있다.
둘째, 학습 상태 다섯 개가 모두 같은 retained set을 냈다. $n_D=20{,}000$에서 선택이 사실상 결정적이라는 뜻이며,
조건부 구조의 변동을 보려면 더 작은 $n_D$가 필요하다.

**결론.** 관측 성공률 1.000이 원고 하한 0.5007을 넘는다. 정리의 하한은 이 설계에서 약 2배 보수적이다.
이 결과의 정확한 명칭은 "제한된 역할 비인지적 표현 class를 사용한 Algorithm 1과 유한표본 정리의 positive-control verification"이다.
일반적인 representation learning 성공을 주장하지 않는다. 그 역할은 SCM-6과 SCM-7이 맡는다.

### 11h. Figure 4 교체: 다섯 설정 1×5 (2026-09-16)

`figures/section5/section5-five-settings.pdf`, 생성 코드 `code/make_section5_five_settings_figure.py`.
5.1의 고정 블록 그림을 대신한다. 옛 그림은 `section5-thm1-fixed-block.pdf`로 남겨 두었다.

**칸 선택.** 합성 다섯 모형의 수치가 거의 같다(PROBE 평균 0.9859–0.9913, MAE 0.0280–0.0298). 본문에 다섯을 다 둘 이유가 없다.
기본, 효과가 잠재변수에 따라 변하는 경우, proxy를 비선형으로 보는 경우 셋만 둔다.

**칸 제목.** SCM 번호를 뺐다. 설계의 축이 둘이므로 두 줄로 나눈다. 윗줄은 proxy가 무엇인지(7 numbers / 1,282 numbers / five images),
아랫줄은 모형 변형(base / varying effect / nonlinear proxy)이다. 그래서 앞 세 칸은 한 proxy에서 모형을 바꾸고, 뒤 두 칸은 기본 모형에서 proxy를 키운다.
EXP-$k$ 방식은 찾아볼 표가 생기므로 쓰지 않았다.

**행 이름.** 셋째 행을 "Learned summary (all $W$), no balance check"로 확정했다(용한 제안). 이 행은 블록마다 스칼라 요약을 만들고,
어느 것도 빼 두지 않으며, 균형을 검사하지 않는다. 비교자는 봉인된 채점이 `best_no_balance`로 지정한 것이다.
SCM-6은 `blockwise_pca`(코드: `[x, I, C] + [H_j @ pc_j for j in range(5)]`), SCM-7은 `cnn_channels`(코드: `[x, cc, block]`, `block`은 다섯 블록의 CNN 채널)다.
둘 다 $X$와 관측 교란 좌표를 그대로 받고도 0.830과 0.605에 머문다. SCM-7의 CNN은 채널 상관 0.946으로 표현 학습 자체는 성공했다.
즉 학습의 실패가 아니라 표현을 고르는 기준의 문제다. 같은 행이 있는 `make_alg1_performance_figure.py`도 같은 이름으로 맞췄다.

**수치.**

| 칸 | $X$ | $X$와 $W$ 전부 | 무균형 요약 | PROBE | MAE | oracle |
|---|---|---|---|---|---|---|
| 7 numbers, base | 0.049 | 0.815 | — | 0.987 | 0.0280 | 0.999 |
| 7 numbers, varying effect | 0.031 | 0.811 | — | 0.987 | 0.0294 | 1.000 |
| 7 numbers, nonlinear proxy | 0.049 | 0.814 | — | 0.988 | 0.0285 | 0.999 |
| 1,282 numbers, base | 0.051 | 0.773 | 0.830 | 1.029 | 0.0290 | 1.000 |
| five images, base | 0.059 | 0.194 | 0.605 | 1.008 | 0.0304 | 1.001 |

**본문 변경.** 5.1의 blue 문장에서 옛 그림의 행을 설명하던 부분을 고쳤다. 깨끗한 블록 대 오염된 블록 대비는 이제 그림이 아니라 숫자로 적는다.
held-out $W_1$이 0.999, 0.999, 1.000이고 held-out $W_4$가 $-0.442$, $-0.471$, $-0.440$이다. 모집단 값 $1.0000$과 $-0.4432$, $0.9530$도 그대로 둔다.
purple 메모에 옛 그림의 경로와, 심사자가 그 대비를 그림으로 보고 싶어 하면 되살리는 선택지를 적었다.

### 11i. Section 5의 이름을 Sim으로 통일 (2026-09-17)

용한 지시로 Section 5에서 SCM 표기를 없앴다. 5.tex에 SCM 언급이 하나도 남지 않았다(파일 경로 제외). 다른 절에는 원래 없었다.

| 이름 | 내용 |
|---|---|
| Sim-1 | 기본. 측정값 하나가 숫자 하나, proxy 7좌표, $n=6{,}000$ |
| Sim-2 | 처치 효과가 잠재 교란에 따라 변함 |
| Sim-3 | proxy를 비선형 렌즈로 봄 |
| Sim-4 | 기본 모형, 측정값을 256차원 벡터로 렌더, 1,282좌표, $n=24{,}000$ |
| Sim-5 | 기본 모형, 측정값을 $96\times96$ 이미지로 렌더, 46,082좌표, $n=12{,}000$ |

단순 치환이 아니라 구조를 바꿨다. 옛 본문은 합성 모형 다섯을 SCM-1~5로 부르고 그중 셋만 그림에 썼다.
이제 도입부가 공유 구조를 먼저 말하고, 그다음 문단이 다섯 이름을 한 번에 소개한다. 그림에 넣지 않은 합성 변형 둘(결과 잡음, proxy 잡음)은
번호 없이 purple 메모로 남겼다. 다섯 합성 모형이 평균 추정값과 평균 절대오차에서 0.002 이내로 일치하므로 셋만 본문에 둔다는 근거도 그 메모에 적었다.

바꾼 곳은 일곱 군데다. Figure 3 캡션("every simulation in this section"), 도입부 두 문단, 5.1의 purple 메모("per simulation"),
5.2의 본문과 배치 계획, 5.3의 모집단 문장, 5.4의 자료 메모다. 5.4 메모는 설정 이름을 Sim으로 바꾸고 실제 산출물 경로만 \texttt{}로 남겼다.
컴파일 깨끗, 미정의 참조 없음. Section 5는 19–23쪽.

## 11j. 2026-09-17 — 흐름도(probe-mechanism) 감사와 재작성

용한이 "A 흐름도가 지금 구현과 맞는지 확인"을 요청했다. 네 칸을 Algorithm 1과 실제 코드에 대조한 결과, 둘은 맞고 하나는 축소,
하나는 틀렸다. 옛 그림은 `results/scm7_v4/shards/confirmation_seed95300000_n12000.json` 한 데이터셋을 썼다.

- (d) 정확. 통과 split {S1,S2,S3,S4,S13,S123}, $\rho=0.06151$, 최대 덩어리 {S1,S2,S3,S13,S123}, 중앙값 $0.97152$가 파일의 최종 답과 일치.
  S4는 $-0.4077$로 고립. 그림의 점·별이 모두 이 숫자였다.
- (c) 심사 규칙은 원고와 같다(`probe_full_target.screen_full_target`, $\max\{R_0-R_1,0\}+z\cdot SE<t$). 다만 그림이 그린 막대는
  `nesting.val_risk_base/augmented`(critic 자기 검증 행)이고 통과 판정은 $S$ 조각에서 났다. 같은 이야기, 다른 행.
- (a) 축소. `oriented_splits()`는 $2^5-2=30$개 부분집합 전부를 뺀다. (d)의 파란 점 다섯 중 둘(S13, S123)이 다중 블록이다.
- (b) 틀림. 두 가지다. (i) CNN 몸통이 split 선택 **전에** 다섯 블록 이미지를 전부 쌓아 한 번 학습된다(`probe_image_w.fit_encoder`),
  그래서 "held-out 블록을 절대 보지 않는다"가 encoder 단계에서 거짓이다. (ii) 표현이 `Z=(X, C, mean(kept)+r I)`이고 블록 0이 빠지면
  `(X, mean)`이다(`probe_highdim_w.representation`). $C$와 $I$를 이름으로 건네받고 배우는 것은 스칼라 $r$ 하나뿐이다.
  블록 0의 특별 취급이 세 곳(표현식, `probe_pixel_critic.image_target`, 후보 격자)에 손으로 박혀 있다. 이것이 전에 지적된 role-aware 설계다.

용한 승인("그대로 진행해") 아래 네 가지를 실행했다.

1. (b)를 encoder 그림에서 "$\phi$를 골라 $\widetilde D_S^2$를 최소화한다"로 바꿨다. 목적함수가 그림에 보인다.
2. (a)의 문구를 "블록 하나"에서 "블록 한 묶음"으로 바꾸고 블록 수를 7개로 맞췄다. "$2^7-2=126$ groups in all"을 적었다.
3. (c)와 (d)의 숫자를 positive-control 감사로 갈아끼웠다. 이 실행만이 역할을 모르는 표현류
   (`positive_control_scm.representation` = $(X_1,X_2,W_j)$)와 원고의 심사·연결·다수결을 그대로 쓴다.
4. MNIST 그림은 (a),(b)에 예시로 남겼다. 캡션에 예시임을 적어야 한다.

표시용 수치는 `code/replay_positive_control_display.py`가 만든다. 봉인된 protocol과 학습 seed에서 (i) 126개 split 전부의 심사 통계,
(ii) 통과·탈락 예시 두 개의 $R_0,R_1$, (iii) 평가 draw 0의 추정값을 다시 읽는다. 결정은 하나도 바뀌지 않는다.
재생이 봉인된 결과를 그대로 재현했다: 126개 중 10개 통과, 이는 `confirmation_v1.json` state 0과 같다. 산출물은
`results/positive_control/screen_display_v1.json`.

그림이 쓰는 숫자:

| 양 | 값 |
|---|---|
| $W_1$ 뺀 경우 | $R_0=0.199209$, $R_1=0.199208$, 통계 $3.71\times10^{-7}$, 통과 |
| $W_4$ 뺀 경우 | $R_0=0.161418$, $R_1=0.158050$, 통계 $3.37\times10^{-3}$, 탈락 |
| 문턱 | $t=5\times10^{-4}$ |
| 통과 통계 범위 | $3.7\times10^{-7}$ ~ $2.5\times10^{-5}$ |
| 탈락 통계 최소 | $2.88\times10^{-3}$ |
| 평가 draw 0 | 7개가 $0.99599$, 3개가 $0.35002$, 반환 $0.99599$ |
| $2\rho$ | $0.32270$, 두 무리 간격 $0.646$ |

overlap 조건 $\eta\le\widehat e_S\le1-\eta$는 (c)의 규칙 문구에 넣었다. 이 설계의 성향점수는 $0.1+0.8\Phi(\cdot)$ 형태라
$[0.10,0.90]$ 안에 구성적으로 들어가고, 실제로 잰 범위도 그렇다. 즉 규칙을 적어도 통과 여부가 달라지지 않는다.

아직 원고 어디에도 이 그림을 넣지 않았다. Section 4의 Algorithm 1 옆이냐 Section 5 맨 앞이냐가 미결이다.

## 11k. 2026-09-17 — 흐름도를 wrapfigure로 원고에 넣음

용한 지시: "이 그림도 일단 draft 에 추가하자. wrapfigure 형태로 일단."

1x4 줄 배치는 폭 6.5인치 본문에서 wrapfigure가 될 수 없다. 0.5\linewidth로 줄이면 글자가 4pt 아래로 내려간다.
그래서 같은 네 칸을 2x2로 다시 짜서 `probe-mechanism-wrap`(3.22 x 3.56인치)을 만들었다. 패널 글자 크기는 6.2pt로 유지된다.
`make_probe_mechanism_figure.py`가 이제 두 배치를 모두 낸다. `LAYOUTS` 사전이 폭, 높이, 패널 크기, 열 수를 담고
`render(stem)`이 같은 그림 코드를 두 번 돈다. 전체폭 float가 필요해지면 `probe-mechanism.pdf`가 그대로 있다.

배치 문제가 하나 있었다. Section 5가 19쪽 바닥에서 시작해서 wrapfigure가 페이지 경계를 넘어 깨졌다. 그림은 19쪽에 남고
글줄만 20쪽으로 흘러서, 그림 없는 좁은 단이 생기고 \texttt{} 경로가 든 purple 메모가 단어 사이로 찢어졌다. 세 가지를 고쳤다.

- `5.tex` 맨 앞에 `\clearpage`를 넣어 Section 5가 새 쪽에서 시작하게 했다. 이게 없으면 wrapfigure가 다시 깨진다.
- 순서를 바꿨다. 절 제목 → wrapfigure → 도입 문단 두 개 → 설계 그래프 float → 나머지.
- purple 메모를 wrap 영역 밖(도입 문단들 뒤)으로 옮겼다. 좁은 단에서 `\texttt{}` 경로는 줄바꿈이 안 되어 글자 간격이 벌어진다.

결과: 20쪽에 절 제목, 오른쪽에 흐름도, 왼쪽에 본문. 21쪽 위에 설계 그래프. 41쪽 그대로, 미정의 참조 0, overfull hbox 0.

남은 흠 하나. 선언 순서 때문에 흐름도가 Figure 3, 설계 그래프가 Figure 4가 되었는데 본문은 Figure 4를 먼저 부른다.
번호가 역순이다. 절을 다시 짤 때 한 줄로 고칠 수 있다.

## 11l. 2026-09-17 — wrapfigure를 1x4 top float로 되돌림

용한 지시: "그냥 1by4 로 figure top 으로 바꾸자." 11k의 wrapfigure 배치를 되돌렸다.

- `probe-mechanism-wrap.pdf`(2x2) 대신 `probe-mechanism.pdf`(1x4)를 `\begin{figure}[!t]`에 `width=\linewidth`로 넣었다.
- 11k에서 wrapfigure를 살리려고 넣었던 `\clearpage`를 뺐다. top float는 페이지 경계에 걸리지 않으므로 필요 없다.
- 선언 순서를 설계 그래프 먼저, 흐름도 나중으로 두어 11k의 번호 역순 문제를 없앴다. 이제 Figure 3이 설계 그래프,
  Figure 4가 흐름도이고 본문이 부르는 순서와 같다.
- 흐름도를 본문에 묶기 위해 파란 문장 하나를 "Each subsection below pairs..." 문단 끝에 붙였다.
  "Figure~\ref{fig:mechanism} shows the algorithm itself, carrying the numbers of the positive-control audit."
- purple 메모의 마지막 문장이 뒤집혔으므로 고쳤다. 이제 half-width wrap용 2x2 판이 `probe-mechanism-wrap.pdf`에 있다고 적는다.

두 배치 모두 `make_probe_mechanism_figure.py`가 계속 낸다. 결과: 20쪽 위에 Figure 3과 Figure 4가 차례로 놓인다.
41쪽 그대로, 미정의 참조 0, overfull hbox 0, Section 5는 다시 19쪽에서 시작한다.

## 11m. 2026-09-17 — 두 그림을 낮추고 caption을 최소화

용한 지시: "Figure 3 너무 길어. 훨씬 줄일 수 있으니까 더 작게 줄여. Figure 4도 훨씬 작게 줄여. 지금 너무 위아래로 길어.
그리고 caption 은 늘 minimize."

**Figure 3 (설계 그래프).** `figures/2026-09-15-scm-family-structure.tex`에서 세 가지를 바꿨다.
y 단위를 1cm에서 0.78cm로 줄이고, 패널 설명을 두 줄에서 한 줄로 줄이고, 그 위치를 $-1.85$에서 $-1.45$로 올렸다.
패널 설명은 "(a) $W_{1:3}$: $U$ only", "(b) $W_5$ acts on $A,Y$", "(c) $W_5$ shifts $W_4$"다.
standalone 상자가 135pt에서 100pt로 낮아졌고 폭은 그대로다. 본문에서 같은 `0.86\textwidth`로 넣으면 2.73인치에서 2.02인치가 된다.
노드가 겹치지 않는지 200dpi로 확인했다. $W_4$와 $W_5$가 가장 가깝고 y 간격 0.66cm로 노드 지름 0.6cm보다 넓다.

**Figure 4 (흐름도).** 6.5 x 2.50인치에서 6.5 x 1.55인치로, 세로 38% 줄였다. 바꾼 것은 넷이다.
- 패널 높이 2.40 -> 1.43인치.
- (a)의 블록 일곱 개를 두 줄에서 한 줄로 폈다. 두 줄일 때 잡아먹던 세로 공간이 사라진다.
- (c)와 (d)의 밑줄을 두 줄에서 한 줄로 합쳤다. "$\widetilde D_S^2$ per group; keep below $t$"와
  "one dot per group; dashed $\tau$"다. 짧은 패널에서 두 줄은 반드시 붙는다.
- 글자를 8.0/6.5pt에서 7.2/6.2pt로 낮췄다.
(b)는 사다리꼴 안 두 줄이 붙어서 전체를 다시 쌓았다. thumbnail 0.64, 라벨 0.625, 화살표 0.560-0.525,
사다리꼴 0.515-0.285, 글 0.450과 0.340, 화살표 0.275-0.240, $Z$ 막대 0.140-0.230, 밑줄 0.04.
`save()`의 text-collision 검사가 깨끗하다. 쓰이지 않게 된 2x2 wrap 판과 그 layout 코드는 지웠다.

**Caption 최소화.** Section 5의 세 caption을 한 줄로 줄였다. 이건 앞으로도 지킬 규칙이다.

| Figure | 지금 caption |
|---|---|
| 3 | The structure shared by every simulation in this section. Only $U$ is hidden. |
| 4 | Algorithm 1 in four steps, with the numbers of the positive-control audit. |
| 5 | What the algorithm returns in five simulations; each dot is one data set. |

정보를 버리지는 않았다. Figure 3의 패널별 설명은 그림 안 패널 라벨과 도입 문단이 이미 담는다.
Figure 4의 숫자는 전부 패널 안에 그려져 있다. Figure 5 caption에 있던 셋째 행 설명
("summarises every block, holds none of them out, and never tests balance" 및 비교자 선택)은
5.1의 purple 메모 앞으로 옮겼다. 지운 wrap 판을 가리키던 purple 문장도 뺐다.

결과: 미정의 참조 0, overfull hbox 0, 전체 41쪽에서 40쪽으로 줄었다. 두 그림이 모두 20쪽 위에 놓이고
그 아래로 본문이 충분히 들어간다.

## 11n. 2026-09-17 — Sim-5 v2(손글씨 숫자 중증도) PRD v1 제안

용한이 이미지 설계를 바꾸자고 했다. $U$는 MNIST 글자 값(1–9, 중증도), $W$는 비튼 이미지 한 장, $X$는 기저 공변량, 블록은 픽셀 영역.
PRD는 `memo/2026-09-17-sim5-v2-digit-proxy-prd-v1.md`에 있다. 검토 대기, DGP seed 안 씀.

사전 측정(원본 MNIST, DGP seed 없음): 원본 $28\times28$의 4픽셀 테두리는 잉크 3.7%인데도 숫자를 0.567 맞힌다. 원본에 빈 영역이 없다.
그래서 $36\times36$ 캔버스에 잡음만 있는 여백 $R$을 붙여 ORC 실패 부류를 만들고, 표식 $M$은 여백 귀퉁이에 둔다. $J=6$, split 62개 전부 열거.
사분면 정확도(원본 $10\times10$): 하나 0.767, 둘 0.905, 셋 0.946. 정확한 균형은 불가능하고 Theorem 3 regime이다.

계산: 4,000행 MLP 한 epoch 0.01초 CPU. 학습 상태 하나 ≈ 15분, 연구 전체 11개 상태 ≈ 3시간 CPU, DeltaAI GPU 0분.
seed 블록 6.1M/6.3M/6.4M은 results, code, memo 어디에도 없음을 단어 단위 grep으로 확인했다.

가장 큰 위험 둘을 PRD 12절에 적었다. (R1) 지렛대 $I$가 $Z$에 없으니 정확한 균형은 없고 잔여가 남는다.
(R2) 전체 이미지 CNN 조정이 PROBE와 비슷하거나 더 좋을 수 있어, Sim-5의 주장은 정확도 우위가 아니라 역할 비인지 실행과 certificate가 된다.
