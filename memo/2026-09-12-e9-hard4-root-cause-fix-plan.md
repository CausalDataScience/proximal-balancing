# E9 hard4 실패 원인과 최소 수정 PERR 계획

작성일: 2026-09-12  
상태: 실행 전 승인된 계획

## Necessity Gate

실제 목표는 SCM이나 기준을 바꾸지 않고 hard4 실패의 원인을 자료 역할 수준에서 분리하고, 이미 두 번
재현된 rotate4를 production experimental mode로 명확히 지정하는 것이다. 최소 완전해법은 같은
$n=6000$ 자료에서 D/S 재사용과 N/E cross-fitting을 2×2로 절제하고, E9의 누락된 hard4 baseline과
다섯 frozen empirical benchmark gate를 다시 계산하는 것이다. 새 SCM, 새 threshold, 원고 수정은
필요하지 않다.

## Plan

1. 네 fold $F_0,\ldots,F_3$를 공유하는 H11, H12, H21, H22와 기존 R4를 구현한다.
2. H21/H22의 두 번째 표현은 반드시 $(D,S)=(F_1,F_0)$에서 다시 맞추고 audit한다.
3. 같은 evaluation unit에 여러 표현이 적용되면 score를 unit별 평균한 뒤 한 행으로 저장한다.
4. 모든 score에 D/S/N/E fold ledger와 hash를 붙이고 row overlap을 fail closed한다.
5. 선택 component에 대해 signed total error를 population target error와 conditional estimation error로
   정확히 분해한다. 정확한 target을 계산할 수 없으면 `NOT_EVALUATED`로 둔다.
6. 기존 E9 seed 20개에서 paired 절제를 실행한다. 기존 JSON은 수정하지 않는다.
7. hard4 baseline은 기존 fold hash를 재생성해 일치시킨 뒤 별도 immutable sidecar에 저장한다.
8. E9 요약은 MAE, p90, return, paired $X$/raw superiority, oracle-excess의 전체 관문을 평가한다.
9. 현재 score/selection 로직이 바뀌지 않으면 seed 2,400,000 및 9,100,000의 기존 rotate4 확증을
   재사용한다. 과학적 로직이 바뀔 때만 fresh 20-seed confirm을 새 protocol로 실행한다.

## Execute

실행 순서는 테스트 작성, 최소 core/helper 구현, CLI 통합 테스트, runtime protocol freeze, old-seed
절제, 분석·보고다. Brier 선택은 같은 $D$에서 결정론적인지 replay test로 확인한다. 모든 새 결과는
atomic immutable JSON이며 source/protocol/result SHA-256과 `complete` flag를 갖는다.

## Review

완료 조건은 다음과 같다.

- arm별 총 생성 unit 6000과 고유 평가행 1500/3000/1500/3000/6000이 확인된다.
- 모든 AIPW score의 evaluation 행이 해당 D/S/N과 겹치지 않는다.
- Algorithm 1의 30-split, 최대 연결성분 median, tied set-valued 반환이 유지된다.
- hard4 baseline은 기존 fold hash가 일치할 때만 인정된다.
- paired data seed 표에서 total signed error, representation target error, conditional estimation error,
  membership과 $\rho$를 함께 보고한다.
- 20-seed 판정은 empirical benchmark gate이며 모집단 확률 certificate로 부르지 않는다.
- rotate4는 operational cross-fitted extension으로만 부르고 원고의 one-shot theorem을 전용하지 않는다.

## Retrospect와 중단 조건

fold hash 불일치, score row leakage, evaluation row 중복, protocol/source hash 불일치가 하나라도 있으면
실행을 중단한다. 2×2가 DS와 NE를 구분하지 못하면 원인을 `UNRESOLVED`로 남긴다. representation,
screen, nuisance, evaluation 각각의 매개 백분율은 계산하지 않는다. 기존 결과를 본 뒤 DGP, screen
$t=3.5$, 101점 $r$ 격자, nuisance, aggregation radius, 성능 기준을 바꾸지 않는다.

