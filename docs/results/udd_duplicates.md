# UDD duplicate audit

3250 distinct image slots, 33 sources. Exact = identical decoded pixels (md5); near = dhash Hamming ≤ 2.

Threshold note: documents are mostly-white, low-entropy images, so perceptual hashes saturate much faster than on photos — at the usual photo threshold (≤6) this corpus reports ~1.3k cross-source 'pairs' dominated by false positives (e.g. diagram ≈ receipt). At ≤2 the survivors are genuine re-uses; treat anything between as candidates needing eyeballing.

- **Exact duplicate groups:** 1 (1 cross-source)
- **Near-duplicate pairs:** 182 (32 cross-source)

## Cross-source EXACT duplicates (same stored image in two sources)

- `chartqa:chartqa_0092`  ↔  `mathvista:mathvista_0095`

## Cross-source near-duplicate pair counts

| source A | source B | near-dup pairs |
|---|---|---|
| docmatix | mtvqa | 6 |
| chartqa | mathvista | 4 |
| doclaynet | docmatix | 4 |
| docmatix | plotqa | 4 |
| doclaynet | hallusionbench | 3 |
| ai2d | docmatix | 2 |
| doclaynet | omnidocbench | 2 |
| docmatix | omnidocbench | 2 |
| pubtabnet | tatqa | 2 |
| ai2d | omnidocbench | 1 |
| doclaynet | infovqa | 1 |
| ocrbench_v2 | screenqa | 1 |

Sample pairs:

- `ai2d:ai2d_0048`  ≈  `docmatix:docmatix_0027`
- `ai2d:ai2d_0073`  ≈  `docmatix:docmatix_0011`
- `ai2d:ai2d_0073`  ≈  `omnidocbench:omnidocbench_0060`
- `chartqa:chartqa_0005`  ≈  `mathvista:mathvista_0073`
- `chartqa:chartqa_0035`  ≈  `mathvista:mathvista_0073`
- `chartqa:chartqa_0045`  ≈  `mathvista:mathvista_0073`
- `chartqa:chartqa_0061`  ≈  `mathvista:mathvista_0073`
- `doclaynet:doclaynet_0020`  ≈  `omnidocbench:omnidocbench_0040`
- `doclaynet:doclaynet_0037`  ≈  `infovqa:infovqa_0012`
- `doclaynet:doclaynet_0045`  ≈  `omnidocbench:omnidocbench_0047`
- `doclaynet:doclaynet_0052`  ≈  `docmatix:docmatix_0004`
- `doclaynet:doclaynet_0052`  ≈  `docmatix:docmatix_0006`
- `doclaynet:doclaynet_0052`  ≈  `docmatix:docmatix_0009`
- `doclaynet:doclaynet_0052`  ≈  `docmatix:docmatix_0041`
- `doclaynet:doclaynet_0065`  ≈  `hallusionbench:hallusionbench_0005`

## Within-source near-duplicates (template/render reuse — expected for synthetic and chart sets)

| source | near-dup pairs |
|---|---|
| tatqa | 64 |
| docmatix | 29 |
| hallusionbench | 25 |
| screenqa | 8 |
| visualmrc | 8 |
| chartqa | 3 |
| mathvista | 3 |
| ocrbench_v2 | 2 |
| omnidocbench | 2 |
| plotqa | 2 |
| pubtabnet | 2 |
| docvqa | 1 |
| publaynet | 1 |

