# UDD duplicate audit

6350 distinct image slots, 33 sources. Exact = identical decoded pixels (md5); near = dhash Hamming ≤ 2.

Threshold note: documents are mostly-white, low-entropy images, so perceptual hashes saturate much faster than on photos — at the usual photo threshold (≤6) this corpus reports ~1.3k cross-source 'pairs' dominated by false positives (e.g. diagram ≈ receipt). At ≤2 the survivors are genuine re-uses; treat anything between as candidates needing eyeballing.

- **Exact duplicate groups:** 0 (0 cross-source)
- **Near-duplicate pairs:** 857 (188 cross-source)

## Cross-source near-duplicate pair counts

| source A | source B | near-dup pairs |
|---|---|---|
| doclaynet | docmatix | 34 |
| docmatix | hallusionbench | 28 |
| docmatix | omnidocbench | 21 |
| docmatix | mtvqa | 12 |
| hallusionbench | tatqa | 12 |
| doclaynet | omnidocbench | 10 |
| docmatix | plotqa | 10 |
| pubtabnet | tatqa | 7 |
| chartqa | doclaynet | 5 |
| ocrbench_v2 | screenqa | 5 |
| charxiv | doclaynet | 3 |
| doclaynet | hallusionbench | 3 |
| doclaynet | mathvista | 3 |
| hallusionbench | pubtabnet | 3 |
| mathvista | plotqa | 3 |
| ai2d | docmatix | 2 |
| chartqa | hallusionbench | 2 |
| charxiv | mathvista | 2 |
| charxiv | pubtabnet | 2 |
| doclaynet | plotqa | 2 |
| mathvista | pubtabnet | 2 |
| ai2d | doclaynet | 1 |
| ai2d | omnidocbench | 1 |
| chartqa | docmatix | 1 |
| charxiv | tatqa | 1 |
| charxiv | docmatix | 1 |
| charxiv | omnidocbench | 1 |
| cord | pubtabnet | 1 |
| doclaynet | infovqa | 1 |
| doclaynet | rvl_cdip | 1 |
| doclaynet | publaynet | 1 |
| docmatix | latexocr | 1 |
| hallusionbench | mtvqa | 1 |
| hallusionbench | rvl_cdip | 1 |
| infovqa | omnidocbench | 1 |
| mathvista | tatqa | 1 |
| ocrvqa | pubtabnet | 1 |
| pubtabnet | rvl_cdip | 1 |

Sample pairs:

- `ai2d:ai2d_0048`  ≈  `docmatix:docmatix_0027`
- `ai2d:ai2d_0073`  ≈  `doclaynet:doclaynet_0159`
- `ai2d:ai2d_0073`  ≈  `docmatix:docmatix_0011`
- `ai2d:ai2d_0073`  ≈  `omnidocbench:omnidocbench_0060`
- `chartqa:chartqa_0100`  ≈  `doclaynet:doclaynet_0159`
- `chartqa:chartqa_0129`  ≈  `doclaynet:doclaynet_0159`
- `chartqa:chartqa_0151`  ≈  `hallusionbench:hallusionbench_0005`
- `chartqa:chartqa_0152`  ≈  `doclaynet:doclaynet_0122`
- `chartqa:chartqa_0152`  ≈  `doclaynet:doclaynet_0159`
- `chartqa:chartqa_0152`  ≈  `docmatix:docmatix_0193`
- `chartqa:chartqa_0170`  ≈  `doclaynet:doclaynet_0122`
- `chartqa:chartqa_0171`  ≈  `hallusionbench:hallusionbench_0005`
- `charxiv:charxiv_0133`  ≈  `doclaynet:doclaynet_0092`
- `charxiv:charxiv_0133`  ≈  `doclaynet:doclaynet_0171`
- `charxiv:charxiv_0133`  ≈  `mathvista:mathvista_0136`

## Within-source near-duplicates (template/render reuse — expected for synthetic and chart sets)

| source | near-dup pairs |
|---|---|
| tatqa | 295 |
| docmatix | 111 |
| visualmrc | 70 |
| publaynet | 48 |
| chartqa | 46 |
| hallusionbench | 38 |
| plotqa | 19 |
| screenqa | 12 |
| pubtabnet | 9 |
| doclaynet | 7 |
| omnidocbench | 5 |
| mathvista | 3 |
| docvqa | 2 |
| ocrbench_v2 | 2 |
| rvl_cdip | 2 |

