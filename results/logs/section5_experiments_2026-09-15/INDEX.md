# Section 5 실험 로그 보관 (2026-09-15)

세션 임시 위치(scratchpad, 백그라운드 작업 로그, `/tmp` 번들)와 DeltaAI에 있던 실험 로그를 한곳에 모았다.
실험 결과 파일 자체는 이미 `results/` 아래에 있고, 이 폴더는 그 결과가 어떻게 만들어졌는지를 보여 주는 로그다.
DeltaAI의 결과 파일 42개는 모두 같은 해시의 로컬 사본이 있음을 확인했다(`deltaai/remote_results_sha256.txt`).

## 실험 경과

| 시각 (CDT) | 실험 | 결과 | 로그 |
|---|---|---|---|
| 09-13–09-14 | E11-v2 정확한 정리 확증 v2, E12 SCM-1 표본 크기 곡선 v2 | 둘 다 PASS | `transcript_experiment_runs.md` |
| 09-14 | SCM-6-v3 확증, n=12,000, 30 seeds (규약 de470c752ecb) | 관문 13개 중 12개, oracle excess 0.0532 미달 | `local_runs/scm6_v3_confirmation_resume_run.log`, `local_runs/2026-09-14_2317_*` |
| 09-14 19:30–21:16 | job 3152513, SCM-7-v3 qualification (GH200) | FAILED: warm seed 3,014초가 runtime 관문 1,200초 초과, GPU 1:46:07 | `deltaai/logs/scm7v3_main_3152513.out`, `deltaai_watch/2026-09-14_2252_*` |
| 09-14 19:14–22:59 | 수정 검사: 병렬 선택(spawn), 인코더 장치, PCA 캐시 | 모두 통과 후 재봉인(2ae2898a) | `local_checks/`, `check_scripts/` |
| 09-14 23:03–09-15 03:28 | job 3153894, SCM-7-v3 qualification 2 + 확증 20 | 확증 FAIL, 관문 13개 중 8개, GPU 4:24:43 | `deltaai/logs/scm7v3_main_3153894.out`, `deltaai_watch/2026-09-15_0332_*` |
| 09-15 01:39 | SCM-6-v3 큰 표본, n=24,000, 10 seeds | 13/13 PASS | `local_runs/scm6_v3_larger_n_run.log`, `local_runs/2026-09-15_0139_*` |
| 09-15 | 모집단 균형 근 곡선, E11-v2 m-curve | 계산 완료 | `local_runs/population_certificate_curve_run.log`, `local_runs/e11_v2_mcurve_run.log` |
| 09-15 | 이미지 전용 문턱 보정(v3 자료를 개발 자료로) | 문턱 4.2e-3 | `transcript_experiment_runs.md` |
| 09-15 06:58–10:01 | job 3155281, SCM-7-v4 확증, 15 seeds (규약 0304726ccf577009) | PASS, 관문 13개 모두, GPU 3:02:56 | `deltaai/logs/scm7v4_3155281.out`, `deltaai_watch/2026-09-15_0934_*` |
| 09-15 07:05–07:14 | Theorem 1 블록 고정 평가, 봉인 E12 적합(SCM-1, n=12,000) | 재현 차이 0 | `local_runs/2026-09-15_0714_*` |
| 09-15 | Theorem 1 블록 고정 평가, 봉인 family 적합(SCM-1–5, n=6,000) | 재현 차이 0 | `transcript_experiment_runs.md` |

GPU 누적 사용: 106.1 + 264.7 + 182.9 = 553.8분(10시간 예산 중 46.2분 남음).

## 보관 파일

| 파일 | 바이트 | sha256 앞 16자 | 내용 | 원래 위치 |
|---|---:|---|---|---|
| `bundles/scm7_v3_bundle_MANIFEST.sha256` | 10,801 | `882b72e86191f37c` | SCM-7-v3 셋째 봉인 번들로 보낸 117개 파일의 해시 | `/tmp/scm7_v3_bundle/MANIFEST.sha256` |
| `bundles/scm7_v3_bundle_README.txt` | 268 | `3fb9ad1d6b80f61e` | 번들 안내 | `/tmp/scm7_v3_bundle/README.txt` |
| `bundles/scm7_v4_bundle_MANIFEST.sha256` | 11,675 | `d0a653aec58c1df0` | SCM-7-v4 번들로 보낸 117개 파일의 해시 | `/tmp/scm7_v4_bundle/MANIFEST.sha256` |
| `bundles/scm7_v4_bundle_README.txt` | 268 | `3fb9ad1d6b80f61e` | 번들 안내 | `/tmp/scm7_v4_bundle/README.txt` |
| `check_scripts/check_encoder_device.py` | 1,168 | `2fa3a7e7fb382e16` | 위 인코더 장치 검사 스크립트 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/scratchpad/check_encoder_device.py` |
| `check_scripts/check_parallel_selection.py` | 1,649 | `5e53806214d6f305` | 위 병렬 선택 검사 스크립트 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/scratchpad/check_parallel_selection.py` |
| `check_scripts/check_pca_cache.py` | 2,278 | `cb7d74928094df9d` | 위 PCA 캐시 검사 스크립트 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/scratchpad/check_pca_cache.py` |
| `deltaai/logs/scm7v3_main_3152513.err` | 0 | `e3b0c44298fc1c14` | job 3152513 stderr(비어 있음) | `deltaai:/work/hdd/bgtp/yjung6/scm7_v3/logs/` |
| `deltaai/logs/scm7v3_main_3152513.out` | 1,188 | `d70b1ebf25c8bb7c` | job 3152513 stdout | `deltaai:/work/hdd/bgtp/yjung6/scm7_v3/logs/` |
| `deltaai/logs/scm7v3_main_3153894.err` | 0 | `e3b0c44298fc1c14` | job 3153894 stderr(비어 있음) | `deltaai:/work/hdd/bgtp/yjung6/scm7_v3/logs/` |
| `deltaai/logs/scm7v3_main_3153894.out` | 6,658 | `bfe6c796bb3d00f5` | job 3153894 stdout, seed별 줄 | `deltaai:/work/hdd/bgtp/yjung6/scm7_v3/logs/` |
| `deltaai/logs/scm7v4_3155281.err` | 0 | `e3b0c44298fc1c14` | job 3155281 stderr(비어 있음) | `deltaai:/work/hdd/bgtp/yjung6/scm7_v3/logs/` |
| `deltaai/logs/scm7v4_3155281.out` | 1,493 | `38ace3a8584ac60f` | job 3155281 stdout, seed별 줄 | `deltaai:/work/hdd/bgtp/yjung6/scm7_v3/logs/` |
| `deltaai/remote_results_missing_locally.txt` | 0 | `e3b0c44298fc1c14` | 로컬에 같은 해시가 없는 원격 결과 목록(빈 파일 = 모두 있음) | `비교 결과` |
| `deltaai/remote_results_sha256.txt` | 5,284 | `14880af2ec177b2d` | 원격 결과 파일 42개의 해시 | `deltaai: sha256sum results/` |
| `deltaai/sacct_and_balance_2026-09-15.txt` | 1,274 | `8c753f7f1d4279e4` | 세 job의 자원, 상태, 시간과 계정 잔액 | `deltaai: sacct, accounts` |
| `deltaai_watch/2026-09-14_2252_job3152513_checkpoint_watch.log` | 1,338 | `3e3af8bdf2ccce8d` | job 3152513(SCM-7-v3 qualification, runtime 관문 실패) 감시 로그 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/bome9fk1r.output` |
| `deltaai_watch/2026-09-14_2335_job3153894_checkpoint_watch.log` | 1,395 | `ed242b9e33cf6eb5` | job 3153894 중간 점검 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/bb0wlhoh1.output` |
| `deltaai_watch/2026-09-15_0332_job3153894_fetch_score_scm7_v3_confirmation.log` | 3,314 | `c1a7c5e7e4c86bb8` | job 3153894 종료, SCM-7-v3 확증 채점(FAIL 8/13)과 그림 스크립트 오류 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/b7vtv07x8.output` |
| `deltaai_watch/2026-09-15_0641_deltaai_workdir_listing_and_sacct.log` | 1,462 | `e9ba98ec357216e6` | v4 제출 전 DeltaAI 작업 폴더, 가상환경, sacct | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/b9vvvi8a6.output` |
| `deltaai_watch/2026-09-15_0934_job3155281_watch_ssh_dropped.log` | 974 | `5b72bd0bbca5b602` | job 3155281(SCM-7-v4) 감시, 07:50 이후 SSH 끊김 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/bffbidjsg.output` |
| `local_checks/2026-09-14_1914_parallel_selection_check_fork_failed.log` | 7,282 | `ff97ec9fc1d3eb83` | 병렬 선택 검사 1차: macOS fork에서 BrokenProcessPool | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/bh7xx3180.output` |
| `local_checks/2026-09-14_1917_parallel_selection_check_retry_failed.log` | 538 | `c009e9c64f8a0ece` | 병렬 선택 검사 2차 실패 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/bsc5ipdht.output` |
| `local_checks/2026-09-14_1928_parallel_selection_check_spawn_shared_memory_ok.log` | 244 | `fc2eebbe5a612586` | spawn과 공유 메모리로 고친 뒤 통과(8 workers 144초) | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/bngizy0t0.output` |
| `local_checks/2026-09-14_2256_encoder_device_check_first_attempt.log` | 562 | `ab917bbae3175c9e` | 인코더 장치 검사 1차: 장치 읽는 줄의 next() 결함 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/b2gsjppru.output` |
| `local_checks/2026-09-14_2256_pca_cache_check_first_attempt_failed.log` | 708 | `03c3d84667caf7e1` | PCA 캐시 검사 1차 실패 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/bx5sipwb4.output` |
| `local_checks/2026-09-14_2258_encoder_device_check_ok.log` | 401 | `e6dbb127273b9be6` | 인코더 장치 검사 통과(mps) | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/b5axhnhfp.output` |
| `local_checks/2026-09-14_2259_pca_cache_check_identical_ok.log` | 329 | `59afaf9931485ef5` | PCA 캐시 경로가 재계산과 동일(30 split 31초) | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/bdr4ccnhl.output` |
| `local_checks/check_encoder_device.log` | 350 | `44db125a9ab008a4` | SCM-7 인코더가 GPU 계열 장치에서 도는지 검사 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/scratchpad/check_encoder_device.log` |
| `local_checks/check_parallel.log` | 166 | `64e0376db6043d97` | SCM-7-v3 병렬 r 탐색 검사 로그 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/scratchpad/check_parallel.log` |
| `local_checks/check_pca_cache.log` | 283 | `b34bd10713b08e10` | SCM-7-v3 블록 PCA 캐시가 재계산과 같은지 검사 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/scratchpad/check_pca_cache.log` |
| `local_runs/2026-09-14_2317_scm6_v3_confirmation_scoring.log` | 1,127 | `2c9bd36bd2fd5444` | SCM-6-v3 확증 완료 감시와 채점 출력(12/13, oracle excess 미달) → results/scm6_v3/scores_confirmation_v1.json | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/bxul0icky.output` |
| `local_runs/2026-09-15_0139_scm6_v3_larger_n_wait_and_scoring.log` | 1,487 | `a8d80b1c46e47f61` | SCM-6-v3 큰 표본 arm 대기와 채점 출력(13/13) → results/scm6_v3/scores_larger_n_v1.json | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/b3drnshfb.output` |
| `local_runs/2026-09-15_0714_t1_fixed_block_scm1_replay.log` | 3,325 | `6f278843b60e7bdb` | Theorem 1 블록 고정 평가, 봉인 E12 n=12,000 재실행(차이 0) → results/t1_fixed_block_scm1_v1.json | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/tasks/byl003esk.output` |
| `local_runs/e11_v2_mcurve_run.log` | 1,019 | `8cafaa7cdebc78e3` | E11-v2 m-curve, m=1..14, 2,000 반복 → results/e11_v2_mcurve_v1.json | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/scratchpad/e11_mcurve.log` |
| `local_runs/population_certificate_curve_run.log` | 1,582 | `8678603535e7bc98` | 30개 split의 모집단 균형 근과 조정값 → results/population_certificate_curve_v1.json | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/scratchpad/pop_curve.log` |
| `local_runs/scm6_v3_confirmation_resume_run.log` | 1,896 | `6d9979aa810b5026` | SCM-6-v3 확증, n=12,000, 30 seeds, 재개 실행 로그 → results/scm6_v3/shards | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/scratchpad/scm6_v3_confirm_resume.log` |
| `local_runs/scm6_v3_larger_n_run.log` | 1,158 | `749469cb68c59896` | SCM-6-v3 큰 표본 arm, n=24,000, 10 seeds 실행 로그 | `/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/scratchpad/scm6_v3_largen.log` |
| `transcript_experiment_runs.md` | 839,986 | `b0d43392d828982f` | 실험을 실행, 채점, 검사, 그림으로 만든 셸 명령 277건과 그 출력. 전면 실행(E11-v2, E12, 문턱 보정, 채점, T1 family)의 로그는 여기에만 있다. | `session transcript` |

## 보관하지 않은 임시 파일과 이유

| 임시 파일 | 이유 |
|---|---|
| `scratchpad/thm1_layout_preview/` | 그림 배치만 보려고 만든 **가짜 수치**다. 실험 결과가 아니므로 저장하지 않는다. |
| `scratchpad/accept_backup/` | 색 표시 수용 전 원고 사본이다. 원본은 git HEAD에 있다. |
| `scratchpad/5tex_*.tex`, `section5_opening_thm1_draft.tex`, `apply_5tex.py` | 원고 초안과 반영 도구다. 수락 전 파란 초안은 `memo/2026-09-14-section5-storyline.md` 7절에 옮겼다. |
| `scratchpad/msbuild/`, `scratchpad/texcheck/`, `__pycache__/` | LaTeX 빌드 산출물과 바이트코드 캐시다. |
| `tasks/bqxs1vhtf.output` | 원고 4.tex를 출력한 것이다. 실험 로그가 아니다. |
| `tasks/bnx3m4bol.output` | 빈 파일이다. |
| `tasks/b1blorf0e.output` | 이 보관 작업의 목록 출력이다. |
| `/tmp/scm7_v3_bundle/`, `/tmp/scm7_v4_bundle/` 본체 | 코드와 MNIST 사본(각 13MB)이다. 해시 목록만 보관하고, 코드는 저장소와 shard의 source snapshot으로 추적한다. |
