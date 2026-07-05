# UDD duplicate audit

28299 distinct image slots, 32 sources. Exact = identical decoded pixels (md5); near = dhash Hamming ≤ 2.

Threshold note: documents are mostly-white, low-entropy images, so perceptual hashes saturate much faster than on photos — at the usual photo threshold (≤6) this corpus reports ~1.3k cross-source 'pairs' dominated by false positives (e.g. diagram ≈ receipt). At ≤2 the survivors are genuine re-uses; treat anything between as candidates needing eyeballing.

- **Exact duplicate groups:** 0 (0 cross-source)
- **Near-duplicate pairs:** 11221 (2217 cross-source)

## Cross-source near-duplicate pair counts

| source A | source B | near-dup pairs |
|---|---|---|
| pubtabnet | tatqa | 436 |
| charxiv | tatqa | 268 |
| doclaynet | docmatix | 216 |
| docmatix | omnidocbench | 99 |
| docmatix | rvl_cdip | 72 |
| chartqa | ocrbench | 64 |
| docmatix | latexocr | 64 |
| hallusionbench | tatqa | 57 |
| mathvista | tatqa | 56 |
| doclaynet | omnidocbench | 52 |
| ai2d | tatqa | 36 |
| sroie | tatqa | 33 |
| docmatix | docvqa | 28 |
| ocrbench_v2 | screenqa | 26 |
| doclaynet | docvqa | 25 |
| docmatix | plotqa | 21 |
| docmatix | ocrvqa | 20 |
| docmatix | mtvqa | 20 |
| ai2d | pubtabnet | 19 |
| charxiv | pubtabnet | 19 |
| doclaynet | mtvqa | 17 |
| doclaynet | latexocr | 17 |
| hallusionbench | pubtabnet | 17 |
| mathvista | plotqa | 17 |
| doclaynet | pubtabnet | 16 |
| doclaynet | plotqa | 16 |
| docmatix | hallusionbench | 16 |
| chartqa | doclaynet | 15 |
| doclaynet | publaynet | 14 |
| doclaynet | mathvista | 14 |
| funsd | ocrbench | 14 |
| docvqa | omnidocbench | 13 |
| dvqa | mathvista | 13 |
| charxiv | doclaynet | 12 |
| doclaynet | hallusionbench | 12 |
| latexocr | rvl_cdip | 12 |
| ai2d | docmatix | 10 |
| chartqa | docmatix | 10 |
| docvqa | ocrbench | 10 |
| charxiv | mtvqa | 9 |
| doclaynet | rvl_cdip | 9 |
| mathvista | pubtabnet | 9 |
| mtvqa | omnidocbench | 9 |
| mtvqa | tatqa | 9 |
| docmatix | pubtabnet | 8 |
| infovqa | ocrbench | 8 |
| docvqa | mtvqa | 7 |
| mtvqa | pubtabnet | 7 |
| ai2d | doclaynet | 6 |
| charxiv | docmatix | 6 |
| charxiv | mathvista | 6 |
| charxiv | omnidocbench | 6 |
| doclaynet | ocrvqa | 6 |
| hallusionbench | latexocr | 6 |
| ai2d | mtvqa | 5 |
| charxiv | hallusionbench | 5 |
| cord | pubtabnet | 5 |
| docvqa | pubtabnet | 5 |
| mathvista | omnidocbench | 5 |
| ocrbench | ocrbench_v2 | 5 |
| ocrvqa | pubtabnet | 5 |
| omnidocbench | pubtabnet | 5 |
| ai2d | omnidocbench | 4 |
| chartqa | docvqa | 4 |
| charxiv | infovqa | 4 |
| charxiv | screenqa | 4 |
| hallusionbench | mtvqa | 4 |
| infovqa | omnidocbench | 4 |
| latexocr | plotqa | 4 |
| ocrbench_v2 | synthdog_ko | 4 |
| ai2d | infovqa | 3 |
| ai2d | charxiv | 3 |
| ai2d | mathvista | 3 |
| chartqa | omnidocbench | 3 |
| chartqa | charxiv | 3 |
| docmatix | infovqa | 3 |
| docmatix | mathvista | 3 |
| docvqa | rvl_cdip | 3 |
| dvqa | pubtabnet | 3 |
| hallusionbench | ocrvqa | 3 |
| hallusionbench | mathvista | 3 |
| latexocr | ocrvqa | 3 |
| mathvista | mtvqa | 3 |
| omnidocbench | rvl_cdip | 3 |
| pubtabnet | synthdog_en | 3 |
| ai2d | hallusionbench | 2 |
| chartqa | rvl_cdip | 2 |
| chartqa | hallusionbench | 2 |
| chartqa | mtvqa | 2 |
| chartqa | pubtabnet | 2 |
| cord | synthdog_en | 2 |
| doclaynet | dvqa | 2 |
| docmatix | dvqa | 2 |
| docmatix | visualmrc | 2 |
| docvqa | ocrbench_v2 | 2 |
| docvqa | stvqa | 2 |
| docvqa | synthdog_ko | 2 |
| funsd | ocrbench_v2 | 2 |
| hallusionbench | omnidocbench | 2 |
| latexocr | omnidocbench | 2 |
| mtvqa | screenqa | 2 |
| ocrbench | stvqa | 2 |
| ocrbench | ocrvqa | 2 |
| ocrbench_v2 | pubtabnet | 2 |
| ocrvqa | plotqa | 2 |
| omnidocbench | screenqa | 2 |
| omnidocbench | publaynet | 2 |
| ai2d | textvqa | 1 |
| ai2d | docvqa | 1 |
| chartqa | screenqa | 1 |
| chartqa | dvqa | 1 |
| charxiv | sroie | 1 |
| cord | plotqa | 1 |
| cord | mtvqa | 1 |
| doclaynet | infovqa | 1 |
| doclaynet | screenqa | 1 |
| doclaynet | ocrbench_v2 | 1 |
| docmatix | ocrbench_v2 | 1 |
| docmatix | synthdog_en | 1 |
| docmatix | funsd | 1 |
| docvqa | textvqa | 1 |
| funsd | rvl_cdip | 1 |
| hallusionbench | im2latex | 1 |
| infovqa | mtvqa | 1 |
| infovqa | screenqa | 1 |
| infovqa | ocrvqa | 1 |
| latexocr | ocrbench_v2 | 1 |
| latexocr | visualmrc | 1 |
| mathvista | ocrbench | 1 |
| mathvista | synthdog_en | 1 |
| mathvista | sroie | 1 |
| mtvqa | synthdog_en | 1 |
| mtvqa | ocrbench | 1 |
| mtvqa | rvl_cdip | 1 |
| mtvqa | ocrbench_v2 | 1 |
| mtvqa | textvqa | 1 |
| mtvqa | synthdog_ko | 1 |
| mtvqa | visualmrc | 1 |
| ocrbench | textvqa | 1 |
| ocrbench | omnidocbench | 1 |
| ocrbench_v2 | stvqa | 1 |
| ocrbench_v2 | textvqa | 1 |
| ocrvqa | omnidocbench | 1 |
| ocrvqa | rvl_cdip | 1 |
| ocrvqa | textvqa | 1 |
| ocrvqa | synthdog_ko | 1 |
| publaynet | pubtabnet | 1 |
| publaynet | rvl_cdip | 1 |
| pubtabnet | sroie | 1 |
| pubtabnet | stvqa | 1 |
| pubtabnet | textvqa | 1 |
| screenqa | synthdog_en | 1 |
| screenqa | textvqa | 1 |
| stvqa | synthdog_ko | 1 |
| synthdog_en | synthdog_ko | 1 |
| synthdog_en | textvqa | 1 |

Sample pairs:

- `ai2d:ai2d_0048`  ≈  `docmatix:docmatix_0027`
- `ai2d:ai2d_0048`  ≈  `infovqa:infovqa_0410`
- `ai2d:ai2d_0073`  ≈  `doclaynet:doclaynet_0159`
- `ai2d:ai2d_0073`  ≈  `doclaynet:doclaynet_0526`
- `ai2d:ai2d_0073`  ≈  `docmatix:docmatix_0011`
- `ai2d:ai2d_0073`  ≈  `docmatix:docmatix_0315`
- `ai2d:ai2d_0073`  ≈  `docmatix:docmatix_0774`
- `ai2d:ai2d_0073`  ≈  `mtvqa:mtvqa_0228`
- `ai2d:ai2d_0073`  ≈  `omnidocbench:omnidocbench_0060`
- `ai2d:ai2d_0073`  ≈  `omnidocbench:omnidocbench_0585`
- `ai2d:ai2d_0268`  ≈  `docmatix:docmatix_0146`
- `ai2d:ai2d_0305`  ≈  `charxiv:charxiv_0140`
- `ai2d:ai2d_0305`  ≈  `mathvista:mathvista_0109`
- `ai2d:ai2d_0305`  ≈  `pubtabnet:pubtabnet_0040`
- `ai2d:ai2d_0305`  ≈  `pubtabnet:pubtabnet_0136`

## Within-source near-duplicates (template/render reuse — expected for synthetic and chart sets)

| source | near-dup pairs |
|---|---|
| tatqa | 5583 |
| visualmrc | 1070 |
| publaynet | 613 |
| plotqa | 463 |
| docmatix | 449 |
| chartqa | 214 |
| rvl_cdip | 180 |
| pubtabnet | 123 |
| doclaynet | 97 |
| hallusionbench | 42 |
| screenqa | 32 |
| omnidocbench | 23 |
| mathvista | 18 |
| dvqa | 16 |
| charxiv | 13 |
| ocrbench_v2 | 13 |
| sroie | 12 |
| docvqa | 11 |
| latexocr | 9 |
| im2latex | 5 |
| ai2d | 4 |
| ocrbench | 4 |
| mtvqa | 3 |
| synthdog_ko | 3 |
| stvqa | 2 |
| infovqa | 1 |
| synthdog_en | 1 |

