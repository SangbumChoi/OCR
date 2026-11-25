# UDD duplicate audit

39837 distinct image slots, 32 sources. Exact = identical decoded pixels (md5); near = dhash Hamming ≤ 2.

Threshold note: documents are mostly-white, low-entropy images, so perceptual hashes saturate much faster than on photos — at the usual photo threshold (≤6) this corpus reports ~1.3k cross-source 'pairs' dominated by false positives (e.g. diagram ≈ receipt). At ≤2 the survivors are genuine re-uses; treat anything between as candidates needing eyeballing.

- **Exact duplicate groups:** 0 (0 cross-source)
- **Near-duplicate pairs:** 22311 (4119 cross-source)

## Cross-source near-duplicate pair counts

| source A | source B | near-dup pairs |
|---|---|---|
| pubtabnet | tatqa | 731 |
| doclaynet | docmatix | 457 |
| charxiv | tatqa | 374 |
| docmatix | omnidocbench | 188 |
| docmatix | rvl_cdip | 115 |
| doclaynet | omnidocbench | 107 |
| docmatix | latexocr | 105 |
| docmatix | mtvqa | 92 |
| hallusionbench | tatqa | 84 |
| docmatix | visualmrc | 81 |
| chartqa | ocrbench | 73 |
| mathvista | tatqa | 71 |
| docmatix | docvqa | 64 |
| docmatix | plotqa | 61 |
| doclaynet | mtvqa | 50 |
| ai2d | tatqa | 49 |
| sroie | tatqa | 42 |
| doclaynet | docvqa | 40 |
| mathvista | plotqa | 35 |
| doclaynet | plotqa | 34 |
| ocrbench_v2 | screenqa | 34 |
| doclaynet | publaynet | 33 |
| docmatix | synthdog_en | 32 |
| mtvqa | omnidocbench | 31 |
| doclaynet | visualmrc | 30 |
| docmatix | ocrvqa | 30 |
| docvqa | omnidocbench | 30 |
| doclaynet | pubtabnet | 26 |
| charxiv | pubtabnet | 25 |
| doclaynet | latexocr | 25 |
| dvqa | mathvista | 25 |
| hallusionbench | pubtabnet | 25 |
| chartqa | doclaynet | 23 |
| ai2d | pubtabnet | 21 |
| doclaynet | mathvista | 21 |
| doclaynet | rvl_cdip | 21 |
| docmatix | ocrbench_v2 | 21 |
| docmatix | synthdog_ko | 21 |
| docmatix | hallusionbench | 21 |
| doclaynet | hallusionbench | 19 |
| ocrbench | ocrbench_v2 | 19 |
| charxiv | doclaynet | 18 |
| omnidocbench | visualmrc | 17 |
| chartqa | docmatix | 16 |
| docmatix | pubtabnet | 16 |
| latexocr | rvl_cdip | 16 |
| ai2d | docmatix | 14 |
| docvqa | mtvqa | 14 |
| funsd | ocrbench_v2 | 14 |
| funsd | ocrbench | 14 |
| docvqa | ocrbench | 13 |
| mtvqa | tatqa | 13 |
| mathvista | pubtabnet | 12 |
| mtvqa | visualmrc | 12 |
| mathvista | omnidocbench | 11 |
| mtvqa | pubtabnet | 11 |
| ocrbench_v2 | pubtabnet | 11 |
| charxiv | mtvqa | 10 |
| pubtabnet | textvqa | 10 |
| pubtabnet | synthdog_en | 10 |
| ai2d | doclaynet | 9 |
| chartqa | docvqa | 9 |
| docvqa | visualmrc | 9 |
| omnidocbench | pubtabnet | 9 |
| cord | pubtabnet | 8 |
| doclaynet | ocrvqa | 8 |
| doclaynet | synthdog_en | 8 |
| docmatix | dvqa | 8 |
| infovqa | ocrbench | 8 |
| latexocr | plotqa | 8 |
| ocrbench_v2 | synthdog_ko | 8 |
| ai2d | mtvqa | 7 |
| chartqa | visualmrc | 7 |
| charxiv | docmatix | 7 |
| charxiv | omnidocbench | 7 |
| docvqa | pubtabnet | 7 |
| hallusionbench | mtvqa | 7 |
| ocrvqa | pubtabnet | 7 |
| omnidocbench | rvl_cdip | 7 |
| chartqa | omnidocbench | 6 |
| charxiv | mathvista | 6 |
| hallusionbench | latexocr | 6 |
| mtvqa | rvl_cdip | 6 |
| mtvqa | ocrbench_v2 | 6 |
| ai2d | omnidocbench | 5 |
| charxiv | hallusionbench | 5 |
| doclaynet | synthdog_ko | 5 |
| docvqa | rvl_cdip | 5 |
| latexocr | mtvqa | 5 |
| latexocr | ocrbench_v2 | 5 |
| latexocr | visualmrc | 5 |
| pubtabnet | sroie | 5 |
| charxiv | infovqa | 4 |
| charxiv | screenqa | 4 |
| cord | synthdog_en | 4 |
| doclaynet | dvqa | 4 |
| doclaynet | ocrbench_v2 | 4 |
| hallusionbench | synthdog_en | 4 |
| infovqa | omnidocbench | 4 |
| ocrbench_v2 | textvqa | 4 |
| ocrvqa | plotqa | 4 |
| plotqa | synthdog_en | 4 |
| plotqa | visualmrc | 4 |
| pubtabnet | synthdog_ko | 4 |
| synthdog_en | synthdog_ko | 4 |
| ai2d | infovqa | 3 |
| ai2d | charxiv | 3 |
| ai2d | mathvista | 3 |
| chartqa | mtvqa | 3 |
| chartqa | rvl_cdip | 3 |
| chartqa | charxiv | 3 |
| chartqa | dvqa | 3 |
| charxiv | publaynet | 3 |
| cord | synthdog_ko | 3 |
| docmatix | infovqa | 3 |
| docmatix | mathvista | 3 |
| docvqa | ocrbench_v2 | 3 |
| dvqa | pubtabnet | 3 |
| hallusionbench | ocrvqa | 3 |
| hallusionbench | visualmrc | 3 |
| hallusionbench | mathvista | 3 |
| latexocr | ocrvqa | 3 |
| latexocr | synthdog_en | 3 |
| latexocr | synthdog_ko | 3 |
| mathvista | sroie | 3 |
| mathvista | mtvqa | 3 |
| mtvqa | synthdog_en | 3 |
| mtvqa | screenqa | 3 |
| ocrbench | stvqa | 3 |
| ocrbench | textvqa | 3 |
| ocrbench | ocrvqa | 3 |
| ocrbench_v2 | omnidocbench | 3 |
| omnidocbench | screenqa | 3 |
| omnidocbench | synthdog_ko | 3 |
| screenqa | textvqa | 3 |
| tatqa | textvqa | 3 |
| ai2d | docvqa | 2 |
| ai2d | textvqa | 2 |
| ai2d | hallusionbench | 2 |
| chartqa | hallusionbench | 2 |
| chartqa | pubtabnet | 2 |
| chartqa | mathvista | 2 |
| cord | textvqa | 2 |
| docmatix | publaynet | 2 |
| docvqa | stvqa | 2 |
| docvqa | synthdog_ko | 2 |
| docvqa | textvqa | 2 |
| docvqa | mathvista | 2 |
| dvqa | infovqa | 2 |
| dvqa | omnidocbench | 2 |
| hallusionbench | omnidocbench | 2 |
| hallusionbench | publaynet | 2 |
| latexocr | omnidocbench | 2 |
| mathvista | synthdog_en | 2 |
| mtvqa | ocrbench | 2 |
| mtvqa | textvqa | 2 |
| mtvqa | synthdog_ko | 2 |
| mtvqa | ocrvqa | 2 |
| ocrbench_v2 | stvqa | 2 |
| ocrbench_v2 | plotqa | 2 |
| ocrbench_v2 | synthdog_en | 2 |
| ocrvqa | synthdog_ko | 2 |
| omnidocbench | publaynet | 2 |
| omnidocbench | textvqa | 2 |
| plotqa | rvl_cdip | 2 |
| publaynet | rvl_cdip | 2 |
| pubtabnet | visualmrc | 2 |
| stvqa | synthdog_ko | 2 |
| ai2d | visualmrc | 1 |
| ai2d | synthdog_en | 1 |
| chartqa | plotqa | 1 |
| chartqa | publaynet | 1 |
| chartqa | screenqa | 1 |
| charxiv | docvqa | 1 |
| charxiv | visualmrc | 1 |
| charxiv | sroie | 1 |
| cord | plotqa | 1 |
| cord | docmatix | 1 |
| cord | mtvqa | 1 |
| cord | stvqa | 1 |
| cord | ocrbench_v2 | 1 |
| doclaynet | infovqa | 1 |
| doclaynet | screenqa | 1 |
| doclaynet | tatqa | 1 |
| docmatix | sroie | 1 |
| docmatix | funsd | 1 |
| docmatix | ocrbench | 1 |
| docvqa | screenqa | 1 |
| dvqa | mtvqa | 1 |
| dvqa | visualmrc | 1 |
| dvqa | publaynet | 1 |
| dvqa | screenqa | 1 |
| funsd | rvl_cdip | 1 |
| hallusionbench | im2latex | 1 |
| infovqa | mtvqa | 1 |
| infovqa | screenqa | 1 |
| infovqa | textvqa | 1 |
| infovqa | ocrvqa | 1 |
| mathvista | ocrbench | 1 |
| mtvqa | publaynet | 1 |
| ocrbench | omnidocbench | 1 |
| ocrbench_v2 | ocrvqa | 1 |
| ocrbench_v2 | visualmrc | 1 |
| ocrvqa | omnidocbench | 1 |
| ocrvqa | rvl_cdip | 1 |
| ocrvqa | synthdog_en | 1 |
| ocrvqa | visualmrc | 1 |
| ocrvqa | textvqa | 1 |
| omnidocbench | stvqa | 1 |
| publaynet | pubtabnet | 1 |
| publaynet | screenqa | 1 |
| publaynet | sroie | 1 |
| pubtabnet | stvqa | 1 |
| pubtabnet | screenqa | 1 |
| rvl_cdip | visualmrc | 1 |
| screenqa | synthdog_en | 1 |
| stvqa | synthdog_en | 1 |
| stvqa | textvqa | 1 |
| synthdog_en | textvqa | 1 |
| synthdog_en | visualmrc | 1 |
| synthdog_ko | visualmrc | 1 |
| synthdog_ko | textvqa | 1 |

Sample pairs:

- `ai2d:ai2d_0048`  ≈  `docmatix:docmatix_0027`
- `ai2d:ai2d_0048`  ≈  `infovqa:infovqa_0410`
- `ai2d:ai2d_0073`  ≈  `doclaynet:doclaynet_0159`
- `ai2d:ai2d_0073`  ≈  `doclaynet:doclaynet_0526`
- `ai2d:ai2d_0073`  ≈  `doclaynet:doclaynet_1280`
- `ai2d:ai2d_0073`  ≈  `doclaynet:doclaynet_1492`
- `ai2d:ai2d_0073`  ≈  `docmatix:docmatix_0011`
- `ai2d:ai2d_0073`  ≈  `docmatix:docmatix_0315`
- `ai2d:ai2d_0073`  ≈  `docmatix:docmatix_0774`
- `ai2d:ai2d_0073`  ≈  `docmatix:docmatix_1234`
- `ai2d:ai2d_0073`  ≈  `docmatix:docmatix_1394`
- `ai2d:ai2d_0073`  ≈  `docvqa:docvqa_1244`
- `ai2d:ai2d_0073`  ≈  `mtvqa:mtvqa_0228`
- `ai2d:ai2d_0073`  ≈  `mtvqa:mtvqa_1155`
- `ai2d:ai2d_0073`  ≈  `omnidocbench:omnidocbench_0060`

## Within-source near-duplicates (template/render reuse — expected for synthetic and chart sets)

| source | near-dup pairs |
|---|---|
| tatqa | 11671 |
| plotqa | 1390 |
| visualmrc | 1255 |
| publaynet | 1174 |
| docmatix | 1006 |
| chartqa | 606 |
| rvl_cdip | 290 |
| pubtabnet | 222 |
| doclaynet | 197 |
| omnidocbench | 81 |
| screenqa | 57 |
| hallusionbench | 42 |
| dvqa | 39 |
| sroie | 30 |
| ocrbench_v2 | 24 |
| mathvista | 18 |
| docvqa | 17 |
| im2latex | 14 |
| charxiv | 13 |
| latexocr | 13 |
| synthdog_ko | 10 |
| mtvqa | 8 |
| ai2d | 4 |
| ocrbench | 4 |
| synthdog_en | 3 |
| stvqa | 2 |
| infovqa | 1 |
| textvqa | 1 |

