# ICLR Writing Review (Sections 1-6, writing only)

Created: 2026-09-18 CDT

- 대상: `ICLR/paper/main.pdf` (2026-09-18 14:08 CDT build, 31쪽, SHA-256 `ecfb5e9c93871ae49de1e32dd749a6ae7db6c0599f05a693a37aa78902a58126`)
- 범위: 수학과 코드는 제외하고, ICLR 리뷰어 관점에서 writing만 점검했다. `memory/durable/writing/`의 Lean floor와 Alexander ceiling을 함께 적용했다.
- 처리 방침 (2026-09-18 용한 지시): Section 5는 보류하고 Section 1-4부터 고친다.
- 경로는 모두 project root `research/papers/single_proxy_balancing/` 기준이다.

## 한 줄 총평

문장은 짧고 정직하며, 형식 결과의 뼈대(가정, 정리, 경계 문장)가 잘 보인다. 다만 리뷰어가 본문 9쪽만 읽으면 이름이 둘인 방법, 부록에 정의를 맡긴 기호와 실험 용어, 중간에 끊긴 정리 상자 때문에 clarity 점수를 깎을 가능성이 크다. 9쪽이 절반 비어 있으니 고칠 공간은 있다.

## A. 리뷰어가 가장 먼저 지적할 큰 문제

**A1. 방법 이름이 두 개다.** 제목과 abstract는 "Proximal Balancing", 본문, 그림, 알고리즘, 결론은 "PROBE"다. "proximal balancing"은 `ICLR/paper/sections/03-why-it-works.tex:4`에서 identification method로 한 번 정의되는데, abstract에서는 representation을 학습하는 절차 전체를 가리킨다. 결론에는 "proximal balancing"이 나오지 않는다. intro 두 번째 문단에서 두 이름의 관계를 한 문장으로 정하고, 결론에서 한 번 되짚는다.

**A2. Section 4는 본문만으로 읽히지 않는다.**
- `ICLR/paper/sections/04-learning-and-search.tex:19`: $\mathcal E_{\mathrm D,j}$를 부록 식 (18)의 우변으로만 정의한다. Theorem 4의 $\sqrt{\mathcal E_{\mathrm D,j}}$가 본문에서 뜻이 없는 기호가 된다.
- `ICLR/paper/sections/04-learning-and-search.tex:23`: Theorem 4 가정에 "the appendix learning conditions hold"가 들어 있다. 자기완결성 위반이고 `YonghanTaste.md`의 self-contained theorem 규칙에도 어긋난다.
- `ICLR/paper/sections/04-learning-and-search.tex:5`: 첫 문단이 명령문 한 문장이다. D, N, E가 무엇의 약자인지(discrepancy, nuisance, evaluation) 본문에 없다.
- `ICLR/paper/sections/04-learning-and-search.tex:12`의 critics, `:16`의 Brier risks와 $n_{\mathrm D}$, `:32`의 $\mathtt{UIF}$가 본문에서 풀이 없이 나온다.

**A3. Section 5의 protocol 용어가 정의 없이 결과의 단위가 된다.** sealed confirmation, exact replay, population calculation, development runs, 13 sealed gates, oracle-gap bound, return rate, role-blind class, learning states. 특히 "passed 8 of 13 gates"가 주요 결과인데 gate의 정의가 본문과 부록 D 어디에도 없다. Section 5 첫머리에 protocol vocabulary 문단을 두는 것을 권한다.

**A4. Theorem 6 상자가 6쪽과 7쪽으로 쪼개지고, 그 사이에 Algorithm 1이 끼어 있다.** 6쪽 맨 아래 "...with"에서 끊기고 7쪽에서 알고리즘 다음에 "every tie counted as failure, obeys (10)"이 이어진다. 가장 먼저 눈에 띄는 조판 결함이다.

**A5. 본문에서 참조되지 않는 float가 셋이다.** Figure 2 (`fig:probe-mechanism`), Table 2 (`tab:verified-results`), Algorithm 1 (`alg:probe-search`)의 본문 `\ref` 횟수가 0이다.

**A6. Abstract가 본문과 어긋나는 곳이 있다** (`ICLR/paper/abstract.tex:1`).
- "Simpson reversals"라는 말이 본문과 부록 어디에도 없다.
- 가정 이름 여섯 개를 나열한 문장이 벽처럼 읽힌다. 핵심 둘(common channel, outcome-relevant completeness)만 남기고 나머지는 standard conditions로 묶는다.
- 마지막 문장(positive-control success rate 1.00 vs 0.50 lower bound)은 맥락 없이 뜻이 안 통하는 약한 마무리다.
- $X$가 정의 없이 나온다.

## B. Section별 중간 크기 문제

### Introduction (`ICLR/paper/sections/01-introduction.tex`)
- B1. `:6` 두 번째 문단이 $Z_j$, $X$, $W_j$를 쓰는데, 이 기호들은 `:8` 세 번째 문단의 의료 영상 예시에서 처음 소개된다.
- B2. `:10` "The dimension makes both obvious shortcuts unsafe."의 "both shortcuts"가 세 문단 앞 `:4`를 가리켜 너무 멀다. "The dimension"도 무엇의 차원인지 불분명하다.
- B3. `:4` "Raw adjustment can be unstable". 실험과 결론이 보여주는 것은 bias다. 단어를 맞춘다.
- B4. `:12` "exactly three"는 방어적으로 읽힌다. `:19` 한 문장 문단은 bullet을 되풀이한다.
- B5. `:75` "the two-page evidence audit"은 논문 안의 쪽수 메타 발언이다.

### Related Work와 1.2
- B6. Table 1에서 세 행이 전부 ×이고 PROBE만 전부 ✓다. PROBE가 ×를 받는 열을 넣어 균형을 잡거나 표를 문단으로 흡수한다.
- B7. `:60` 1.2 첫 문단이 주장 없는 이음말 한 문장이다 (Lean Rule 4.5).
- B8. `:69` "non-nesting"이 정의 없이 먼저 나온다. 뜻은 `:73`에서야 나온다.
- B9. Figure 1 caption이 세 panel의 차이를 말하지 않는다. 그림 소스 기준으로 (b)는 (a)에 $W_j\to Y$를, (c)는 (b)에 $W_{-j}\to Y$를 더한 것이다 (`research/AGENT.md` 9a caption 규칙).

### Section 2 (`ICLR/paper/sections/02-probe.tex`)
- B10. `:6` 중복 문장: "An encoder $\phi_j$ produces $Z_j=\phi_j(V_j)$, and $Z_j=\phi_j(V_j)$ does not take $W_j$ as an input."
- B11. `:4` $W$가 $U$의 proxy라는 말이 본문 setup에 없다.

### Section 3 (`ICLR/paper/sections/03-why-it-works.tex`)
- B12. `:37`, `:60`의 "images"(사상의 상)가 running example의 medical image와 부딪힌다. "induced laws"로 바꾼다.
- B13. `:16` "Theorem 1 formalizes why clinical relevance alone is insufficient." Theorem 1은 clinical relevance에 대해 말하지 않는다.
- B14. `:23` "Consistency links potential and observed outcomes."는 가정 이름을 풀어 쓴 것 이상의 정보가 없다.
- B15. `:119` Assumption 4의 $\mathcal Q_{j,z}^\phi$가 본문에 정의되지 않는다. 가정이 부등식 없이 말로만 적혀 있어 $\Gamma_j(\phi)$를 본문만으로 해석할 수 없다.
- B16. `:67`~`:69` 3.2 첫 두 문단이 같은 말을 두 번 하고, 두 번째 문단의 "therefore"가 앞 문장에서 따라 나오지 않는다.
- B17. `:71` held-out 단위가 block $j$에서 split $S$로 설명 없이 바뀐다.
- B18. `:101`~`:108` 같은 대상을 residual treatment discrepancy, incremental treatment signal, additional treatment information 세 이름으로 부른다.

### Section 4 (`ICLR/paper/sections/04-learning-and-search.tex`)
- B19. 제목 "Estimation"이 learning, estimation, search 세 내용을 덜 담는다.
- B20. `:48` $T=2^J-2$가 audit variable $T_j$와 겹친다. 같은 문단의 $N$은 $\mathcal I_{\mathrm N}$과, $R$은 $\widehat R_{0,\phi}$와, $L$은 $L_2$와 겹친다.
- B21. `:59` 두 문장에 새 기호 여섯 개가 들어 있고, $\theta_S=\tau_{\widehat Z_S}$가 식 (3)의 $\theta_S$를 다시 정의한다.
- B22. `:41` Theorem 5가 "With conditional probability ..., the bound below holds under [가정]" 순서라 읽기 어렵고, 무엇에 조건부인지 statement 안에 없다.
- B23. `:63` "on one event of probability at least $1-\delta$ every $S$ satisfies"는 "there is an event ... on which every $S$ satisfies"로 푼다.

### Section 5 (`ICLR/paper/sections/05-experiments.tex`)
- B24. image threshold 이야기가 `:19`, `:45`, `:47`, `:53`에서 네 번 나온다.
- B25. baseline 이름이 Figure 3 범례와 본문 prose에서 다르다.
- B26. scalar 설정 개수가 흔들린다 (Figure 3은 3개, `:15`는 three, `:39`는 five, Table 2는 $5\times20$). Figure 3의 Sim 번호가 부록의 SCM 번호와 다르다.
- B27. Table 2의 Mean error 열에 서로 다른 통계량(mean range, median, p90, not primary)이 섞여 있다.
- B28. `:17` 24,000 결과를 먼저, earlier 12,000 결과를 뒤에 말하는 역순이고 "improves with more data"는 동어반복이다.
- B29. `:39` "minus 0.47"처럼 음수를 글자로 쓴다.
- B30. 5.1 "Protocol discipline"은 부록 잔재처럼 보이고 제목만 sentence case다.

### Conclusion과 뒷부분 (`ICLR/paper/sections/06-conclusion.tex`, `ICLR/paper/main.tex`)
- B31. `06-conclusion.tex:4` "learns without one prespecified block"은 "with one prespecified block withheld"가 정확하다.
- B32. Limitations 문단이 없다.
- B33. AI use statement의 "will review ... before submission"은 미래형이다.
- B34. Reproducibility statement의 "can be supplied"는 코드가 없다는 신호로 읽힌다.

## C. 작은 것들
- C1. Abstract의 "46,082-pixel image": $5\times96\times96=46{,}080$이고 나머지 2는 픽셀이 아니다.
- C2. Figure 2 panel 글씨가 각주보다 작다.
- C3. "boundaries"와 "sealed"가 논문 고유 어휘처럼 반복된다.
- C4. Section 3 제목이 길고, 3.1은 문장형 제목, 3.2는 명사구 제목이다.
- C5. Theorem 4의 label이 `cor:representation-causal`이다.
- C6. 같은 `causal-graphs.pdf`가 본문 Figure 1과 부록 그림으로 두 번 들어간다.

## D. 고치는 순서 제안
1. A4, A5 (기계적이고 즉시 눈에 띔)
2. A1 (문장 두 개)
3. A2, A3 (본문 자기완결성)
4. A6와 Section 5 scalar 개수, 이름 통일
5. 나머지 B 항목

## 처리 상태

2026-09-18: Section 1-4를 `YonghanTaste.md` Editing Markup으로 수정했다 (삭제는 빨간 취소선, 추가는 파랑, 코멘트는 보라). 용한이 수락하기 전까지 원문과 표시를 보존한다. 취소선을 위해 `ICLR/paper/main.tex`에 `\usepackage[normalem]{ulem}` 한 줄을 추가했고, 수락 시 함께 지운다.

| 항목 | 상태 |
|---|---|
| A1 | intro 두 번째 문단에 "PROBE implements proximal balancing" 문장 추가. abstract와 conclusion은 범위 밖이라 보류 |
| A2 | D, N, E 표본, critic, Brier risk, $\mathtt{UIF}$, $\mathcal E_{\mathrm D,j}$ 풀이 추가. Theorem 4의 "appendix learning conditions"는 수학 확인이 필요해 보라 코멘트로 남김 |
| A4 | 수락본 조판에서 Theorem 6 상자가 7쪽 안에 온전히 들어감 (Algorithm 1은 그 위). float 설정은 바꾸지 않음 |
| A5 | Figure 2, Algorithm 1 본문 참조 추가. Table 2는 Section 5라 보류 |
| B1~B5, B7~B14, B16~B19, B21~B23 | 수정 |
| B6 | Table 1 균형 문제는 인용 방법별 내용 판단이 필요해 보라 코멘트 |
| B15 | $\mathcal Q_{j,z}^\phi$ 정의 추가. 부등식 복원 여부는 보라 코멘트 |
| B20 | $T, N, R, L$ 충돌은 부록 증명과 같은 글자를 써서 보라 코멘트 |
| C4, C5 | 손대지 않음 |
| A3, A6, B24~B34, C1~C3, C6 | 보류 (Section 5, abstract, conclusion, 그림) |

검증 (scratchpad 복사본에서 compile, repo의 `ICLR/paper/main.pdf`는 다시 만들지 않음):
- 표시본: 32쪽, error 0, overfull 0, undefined reference 0. 취소선, 파랑, 보라가 수식과 쪽 경계에서 유지됨을 렌더링으로 확인.
- 수락본 (표시를 모두 수락한 복사본): 31쪽, error 0, overfull 0. Section 4는 6쪽, Experiments는 7쪽, Conclusion은 9쪽, References는 10쪽에서 시작하여 본문 9쪽 한도를 지킨다.
- 수정 전 원본: session scratchpad `backup-pre-markup/`에 복사해 두었고, 표시본 자체에도 빨간 취소선으로 남아 있다.

## 수학 점검 (2026-09-18, 용한 지시 "수학이 잘못된 곳이 있다면 고쳐")

방법: 식별 부분(Section 2~3, 부록 B, 증명 E.1)과 학습·추정·탐색 부분(Section 4, 부록 C, 증명 E.2, 수치)을 두 read-only 감사로 나누어 검토하고, 지적 사항은 모두 원문에서 직접 다시 확인한 뒤에만 고쳤다. 표시 규칙은 위와 같다 (빨간 취소선, 파랑, 보라).

핵심 정리는 맞다. Theorem 1, 3, 4, 5, 6과 Proposition 1을 독립적으로 다시 유도했고, Theorem 3의 상수 $\Gamma_j(\phi)/\{\eta(1-\eta)\}$, Theorem 5의 $q_{e,j}(q_{0,j}+q_{1,j})/\eta$와 Chebyshev 항, Theorem 6의 Hoeffding 항 $L\,\mathbb E\{e^{-R\Delta^2/2}\}$와 연결 반경 논리가 모두 성립한다.

| 위치 | 문제 | 수정 |
|---|---|---|
| `appendix/02-identification.tex` canonical completeness 식 | 두 arm을 한꺼번에 조건화한 형태라, Example 2에서 $p_{z,1}-p_{z,0}=40/91,\,40/99\ne0$이 되어 이 정의로는 completeness가 성립한다. 증명의 "completeness fails"가 거짓이 된다 (오류) | 문헌의 arm별 형태로 교체: for every $(a,x)$, $\mathbb E\{g(U)\mid Z^{\mathrm{tr}},A=a,X=x\}=0$ a.s. $\Rightarrow g(U)=0$ a.s. (Cui et al. Assumption 8, Zotero `S5AVGWGY`) |
| Section 1.2 "Neither condition implies the other" | 부록은 ORC가 canonical completeness를 함의하지 않는 방향만 보였다 | Proposition 3에 "neither ... implies the other"를 추가하고, Example 1에서 $E_W$를 독립 Bernoulli(1/2)로 바꾸는 반례로 반대 방향을 증명 |
| 부록 Proposition 3 | "a designated proxy split"가 어떤 representation인지 불분명 | $W_{-j}=Z^{\mathrm{tr}}, W_j=Z^{\mathrm{out}}$, every representation $\phi_j$로 명시 |
| 부록 B Kuroki 문장 | rank 조건 없이는 거짓이고, Section 1.1 참조는 존재하지 않는 논의를 가리킴 | rank 조건 추가, 참조 삭제 |
| Assumption 4 (본문) | 부등식이 없어 "it bounds ... by the norm"이 수학적으로 불완전 | 한 줄 부등식과 $\mu^\phi_{a,j,z}$ 정의로 교체 |
| Theorem 3 (본문) | $\eta$ 범위 없음 | $\eta\in(0,1/2]$ 추가 |
| Assumption 5 (본문) | "for every $\phi$ ... for one $\eta$" 양화사 순서가 부록과 반대 | "There is one $\eta$ such that, for every $\phi$" |
| Theorem 4 (본문) | "uniformly over the encoder class"는 $\Gamma_j$가 하나인 것처럼 읽힘 | "for every encoder in the class" |
| 부록 C AIPW 정리 | 추가 가정이 Theorem 4 전체 가정을 요구해 본문 Theorem 5와 어긋남 | $\widehat\phi_j$에서의 Assumption 1, 2, 4와 overlap으로 교체 |
| 부록 C 탐색 논의 | condition (b)의 근거가 $|\widehat\tau-\tau|$ 정리를 가리킴 | Equation (eq:conditional-aipw-error)로 교체 (두 곳) |
| 부록 C 근사 band 문단 | 쓰인 그대로는 condition (a), (c)가 성립하지 않음 | band를 target class로 세고, 분리 조건을 $4(\rho+b)$로 명시 |
| 부록 C screen 문단 | 더 엄격한 screen이 $\Delta$를 늘린다는 주장은 거짓 ($\{\tau,\tau,\tau,c\}$에서 $\tau$ 둘을 빼면 $\Delta$가 $1/2$에서 $0$) | "$L$, $p$를 늘리지 않고, 비표적 split이 빠질 때만 $\Delta$가 커질 수 있다" |
| 부록 E Theorem 3 증명 | $Q^\phi_{b,j,z}\in\mathcal Q^\phi_{j,z}$ 확인 누락, Cauchy-Schwarz 근거 위치 오표기 | 한 문장 추가, 근거 위치 수정 |
| abstract, 부록 D | "46,082-pixel"은 $5\cdot96^2=46{,}080$ 픽셀과 스칼라 2개 | "46,082-dimensional", 부록 D에 $2+5\cdot96^2$ 문장 추가 |
| 부록 D positive control | 0.5007은 $\delta=0.05$, $m=T=126$일 때만 성립하는데 값이 없음. $4\rho$ 반올림 | 두 값 명시, $\rho=0.161348$, $4\rho=0.645390$ |

오류가 아니어서 고치지 않은 것 (보라 코멘트 또는 보고만):
- Theorem 2는 Definition 2를 다시 말할 뿐이다. 더 강한 형태를 보라 코멘트로 제안했다.
- Theorem 4의 "appendix learning conditions" 문구 (기존 보라 코멘트 유지).
- $T, N, R, L$ 기호 충돌 (기존 보라 코멘트 유지).
- Section 5 positive control은 $m=T$라 split 표집 항이 실제로 시험되지 않는다. $4\rho$가 간격의 0.16% 안에 있으므로 $\widehat\sigma_S$의 Monte Carlo 오차를 보고할 필요가 있다. Section 5 본문은 건드리지 않았다.

검증: 표시본 32쪽, 수락본 31쪽. 둘 다 error 0, overfull 0, undefined reference 0. 수락본은 Conclusion 9쪽, References 10쪽, 부록 11쪽 시작, Theorem 6은 7쪽에 온전히 있다. 8쪽 여유는 약 1줄이다.
