# Real-World Data Audit Plan

Date: 2026-08-23 CDT

## Necessity Gate

The next modeling decision depends on whether the proposed public datasets actually contain a defensible treatment, outcome, pretreatment proxy surface, and benchmark target. Downloading and auditing the raw sources is the smallest intervention that resolves this uncertainty without fitting any model.

## Manager Intent Lock

### Output Contract

- **Deliverable:** source snapshots for WSCdata, LaLonde, and RHC, a machine-readable provenance manifest, and one data-audit memo.
- **Consumer:** Yonghan, who will decide whether to approve Step 2 modeling.
- **Form:** files under `materials/real_world_data/` and one Markdown memo under `memo/`.
- **Acceptance criteria:** every stored source file has a verified SHA-256 digest; schemas, sample counts, missingness, treatment and outcome candidates, WSC treatment-by-preference cells, proxy candidates, and timing risks are recorded; the WSC observational construction is stated precisely.
- **Evidence:** independent checksum recomputation, programmatic CSV or Stata inspection, and final path-scoped Git status.
- **Boundaries:** no model fitting, causal estimation, Monte Carlo, README or ARTIFACTS edit, dependency installation, Git staging, commit, or push. Existing dirty files are preserved.

## Executer Plan

1. Copy only source files required for the three audits from exact public URLs into dataset-specific directories.
2. Record source commit or page URL, access date, license or permission statement, local path, byte size, and SHA-256 digest in `manifest.json`.
3. Inspect each dataset with the existing Python runtime. Use CSV parsing and installed `pandas` Stata readers. Do not install packages.
4. Write the audit memo from the inspected values. Distinguish observed facts, proposed constructions, and unverified causal assumptions.
5. Recompute all hashes from the final local files and check the manifest against them.

## Failure Conditions

- A source URL is inaccessible or changes during the audit.
- Licensing or reuse permission cannot be located.
- The documented treatment, outcome, or timing cannot be reconciled with the stored schema.
- A source file cannot be parsed using the existing runtime.

Any failure is recorded explicitly. No silent source substitution is allowed.

## QA Strategy

- Assert expected row and column counts and binary treatment values.
- Verify WSC four-cell counts sum to the full sample.
- Compare item-table row counts with the WSC main table and flag the absence of an explicit participant identifier.
- Verify every manifest digest by recomputing SHA-256 after all copies.
- Review the memo from the reader, data-engineering, and causal-design perspectives.

## Manager Plan Review

Decision: `pass`.

The plan matches the approved Step 1, uses the smallest sufficient artifact set, and stops before modeling.
