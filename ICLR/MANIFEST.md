# ICLR 2027 source manifest

- Retrieval time: `2026-09-17T13:29:19-0500` (`2026-09-17T18:29:19Z`)
- Retrieval method: direct HTTPS download with redirects enabled and HTTP failures treated as errors
- Integrity algorithm: SHA-256
- Scope: public, official author-facing ICLR/OpenReview sources and the official ICLR 2027 LaTeX archive

## Downloaded official sources

| Local file | Official source URL | SHA-256 |
| --- | --- | --- |
| `official/call-for-papers.html` | `https://iclr.cc/Conferences/2027/CallForPapers` | `205cb8e804fa7728c5d9516063bc1cb91ca5608f5b0b08466abcbce47795602b` |
| `official/author-guidelines.html` | `https://iclr.cc/Conferences/2027/AuthorGuidelines` | `707990f15a47e0c80db25fa6b7bac368d566599bfe49fb426d0e2f632423bcc9` |
| `official/dates.html` | `https://iclr.cc/Conferences/2027/Dates` | `3e1cc26cb1da5f0f55a8dbb2de5b33a8cf0eaf2301c4420194a33bffdf56d2ed` |
| `official/ai-policy-for-authors.html` | `https://iclr.cc/Conferences/2027/AIPolicyForAuthors` | `cbdb333641cc0fea83ed7bd7c6fd2591d4daf7e7620fbfb03de7d006d5e49468` |
| `official/code-of-ethics.html` | `https://iclr.cc/public/CodeOfEthics` | `67a14d009eb952249d7ada572216bce8c741d63af9bf0d9e419dbdf764e02b26` |
| `official/code-of-conduct.html` | `https://iclr.cc/public/CodeOfConduct` | `02dbff86b15e2933b86d056ada0c9faae038a90cd129de7e1e31079e2dd757a9` |
| `official/reviewer-guidelines.html` | `https://iclr.cc/Conferences/2027/ReviewerGuidelines` | `8c26fd4c2e7a6d3d42094275d526a74e7435c722f80bb1162bf5fe334c8f28c7` |
| `official/submission-policies-explainer.html` | `https://blog.iclr.cc/2026/09/02/submission-policies-for-iclr-2027/` | `b1c26c170079d68efe0588a8f1c2cdabf8a7dc98579a7d1c7c6de8620795f8f7` |
| `official/paper-assistant-announcement.html` | `https://blog.iclr.cc/2026/09/10/making-googles-paper-assistant-tool-pat-available-to-iclr-submitters/` | `7c7f151e29ac431a799e1bac7b82a1545699d21b3d183001b948ea2e380d3f63` |
| `official/openreview-submission-invitation.json` | `https://api2.openreview.net/invitations?id=ICLR.cc%2F2027%2FConference%2F-%2FSubmission` | `23c8d0e700987c10d3b3b8616730b381cfd38f952ab02aa328ef630e25e83dfb` |
| `template/iclr-2027-style-files.zip` | `https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip` | `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e` |

The style archive returned `Content-Length: 39348`, `Last-Modified: Tue, 28 Jul 2026 18:53:19 GMT`, and `Content-Type: application/zip` when checked. Raw HTML snapshots can contain request-specific page data such as Content Security Policy nonces; their hashes identify the exact preserved snapshots, not a permanent upstream page identity.

## Official style archive inventory

The archive contained one directory entry and exactly seven files, all under the safe relative prefix `iclr2027/`. No absolute path or parent traversal entry was present.

| Extracted file | SHA-256 |
| --- | --- |
| `template/iclr2027/iclr2027_conference.bib` | `cdd86e7d4c31854dcf2145871657c944588a6d44c3b72e160ff4baa8df1a52fb` |
| `template/iclr2027/math_commands.tex` | `90473c4d0542070db244cea73ef962d6cddc5b2a746757e6a40ddf5fdfb90ba9` |
| `template/iclr2027/fancyhdr.sty` | `b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea` |
| `template/iclr2027/iclr2027_conference.bst` | `2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5` |
| `template/iclr2027/iclr2027_conference.tex` | `03e556d6e5593e498fd39f262ec2d184ebfe8693cbf47fb7cea3402d8d5166ac` |
| `template/iclr2027/natbib.sty` | `88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d` |
| `template/iclr2027/iclr2027_conference.sty` | `797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6` |

The official archive does not include a compiled `iclr2027_conference.pdf`; the preserved `.tex` sample can be compiled locally for verification without modifying the official source files.

## OpenReview form snapshot

The JSON contains one invitation with ID `ICLR.cc/2027/Conference/-/Submission`. Its schema exposes the current submission fields, including the author set, reciprocal-reviewing fields, AI-assistance disclosure, paper-visibility acknowledgement, PDF and supplementary upload constraints, and optional Paper Assistant request. The local checklist summarizes these fields but does not fill them or make decisions for the authors.

## Integrity check

From this directory, run:

```bash
shasum -a 256 -c SHA256SUMS
```

## Verification record

Verification on 2026-09-17 produced the following results:

- all 18 entries in `SHA256SUMS` passed;
- the archive contained exactly seven files and no absolute or parent-traversal path;
- the OpenReview JSON parsed successfully and contained the expected submission, AI-assistance, reciprocal-reviewing, and supplementary-material fields;
- every HTML snapshot contained its expected official page title or policy anchor;
- the untouched official sample compiled with `latexmk` and TeX Live 2026 to a seven-page, US-letter PDF with zero compilation errors;
- the upstream sample emitted underfull-box messages and an `end occurred inside a group` warning, which were not repaired because the extracted official files are preserved verbatim;
- the temporary build directory and generated PDF were removed after verification.
