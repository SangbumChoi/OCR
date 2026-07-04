# UDD duplicate audit

9389 distinct image slots, 33 sources. Exact = identical decoded pixels (md5); near = dhash Hamming ≤ 2.

Threshold note: documents are mostly-white, low-entropy images, so perceptual hashes saturate much faster than on photos — at the usual photo threshold (≤6) this corpus reports ~1.3k cross-source 'pairs' dominated by false positives (e.g. diagram ≈ receipt). At ≤2 the survivors are genuine re-uses; treat anything between as candidates needing eyeballing.

- **Exact duplicate groups:** 0 (0 cross-source)
- **Near-duplicate pairs:** 1628 (307 cross-source)

## Cross-source near-duplicate pair counts

| source A | source B | near-dup pairs |
|---|---|---|
| pubtabnet | tatqa | 84 |
| doclaynet | docmatix | 29 |
| hallusionbench | tatqa | 18 |
| charxiv | tatqa | 17 |
| docmatix | omnidocbench | 15 |
| sroie | tatqa | 14 |
| docmatix | hallusionbench | 13 |
| doclaynet | omnidocbench | 9 |
| docmatix | mtvqa | 8 |
| ocrbench_v2 | screenqa | 8 |
| chartqa | doclaynet | 7 |
| doclaynet | hallusionbench | 7 |
| docmatix | plotqa | 6 |
| mathvista | plotqa | 6 |
| charxiv | doclaynet | 5 |
| hallusionbench | pubtabnet | 5 |
| charxiv | hallusionbench | 4 |
| charxiv | pubtabnet | 4 |
| doclaynet | mathvista | 4 |
| mathvista | tatqa | 4 |
| ai2d | docmatix | 3 |
| chartqa | docmatix | 3 |
| charxiv | mathvista | 3 |
| docmatix | latexocr | 3 |
| chartqa | hallusionbench | 2 |
| doclaynet | plotqa | 2 |
| docmatix | docvqa | 2 |
| hallusionbench | mtvqa | 2 |
| hallusionbench | mathvista | 2 |
| mathvista | pubtabnet | 2 |
| ai2d | doclaynet | 1 |
| ai2d | omnidocbench | 1 |
| charxiv | docmatix | 1 |
| charxiv | omnidocbench | 1 |
| cord | pubtabnet | 1 |
| doclaynet | infovqa | 1 |
| doclaynet | rvl_cdip | 1 |
| doclaynet | latexocr | 1 |
| doclaynet | publaynet | 1 |
| hallusionbench | rvl_cdip | 1 |
| infovqa | omnidocbench | 1 |
| ocrvqa | pubtabnet | 1 |
| pope | textvqa | 1 |
| pubtabnet | rvl_cdip | 1 |
| pubtabnet | sroie | 1 |
| screenqa | textvqa | 1 |

Sample pairs:

- `ai2d:ai2d_0048`  ≈  `docmatix:docmatix_0027`
- `ai2d:ai2d_0073`  ≈  `doclaynet:doclaynet_0159`
- `ai2d:ai2d_0073`  ≈  `docmatix:docmatix_0011`
- `ai2d:ai2d_0073`  ≈  `omnidocbench:omnidocbench_0060`
- `ai2d:ai2d_0268`  ≈  `docmatix:docmatix_0146`
- `chartqa:chartqa_0100`  ≈  `doclaynet:doclaynet_0159`
- `chartqa:chartqa_0129`  ≈  `doclaynet:doclaynet_0159`
- `chartqa:chartqa_0131`  ≈  `doclaynet:doclaynet_0238`
- `chartqa:chartqa_0151`  ≈  `hallusionbench:hallusionbench_0005`
- `chartqa:chartqa_0152`  ≈  `doclaynet:doclaynet_0122`
- `chartqa:chartqa_0152`  ≈  `doclaynet:doclaynet_0159`
- `chartqa:chartqa_0152`  ≈  `docmatix:docmatix_0193`
- `chartqa:chartqa_0164`  ≈  `docmatix:docmatix_0269`
- `chartqa:chartqa_0170`  ≈  `doclaynet:doclaynet_0122`
- `chartqa:chartqa_0171`  ≈  `hallusionbench:hallusionbench_0005`

## Within-source near-duplicates (template/render reuse — expected for synthetic and chart sets)

| source | near-dup pairs |
|---|---|
| tatqa | 754 |
| visualmrc | 191 |
| docmatix | 94 |
| publaynet | 90 |
| chartqa | 49 |
| plotqa | 42 |
| hallusionbench | 41 |
| pubtabnet | 14 |
| ocrbench_v2 | 12 |
| doclaynet | 9 |
| screenqa | 9 |
| mathvista | 4 |
| omnidocbench | 4 |
| docvqa | 2 |
| rvl_cdip | 2 |
| charxiv | 1 |
| im2latex | 1 |
| mtvqa | 1 |
| sroie | 1 |

