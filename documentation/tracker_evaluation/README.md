# Fixed-observation association benchmark

Identity association on the fixed universe of non-interpolated saved observations. Gold identity 0 is excluded from association scoring and retained for rejection scoring.

**Dataset:** 21 manually curated development videos; no held-out split and no parameter refitting.

**Development-set warning:** In-sample/development-set comparison. Algorithm design may have been informed by these videos, and the provenance of ILP parameter tuning is uncertain.

**Observation provenance:** Saved source observations were preselected and their boxes were RTS-smoothed after manual identity reconstruction; results are conditional association scores, not end-to-end MOT scores.

## Aggregate results

| Tracker | Coverage | Pair precision (micro) | Pair recall (micro) | Pair F1 (micro / macro) | Contaminated tracks | Fragmentation excess | Exact videos | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `boxmot_oc_sort` | 0.981 | 0.985 | 0.662 | 0.792 / 0.857 | 9/165 (5.5%) | 845 | 1/21 | 9.0 s |
| `boxmot_bot_sort_gmc_no_reid` | 0.996 | 0.920 | 0.752 | 0.828 / 0.858 | 16/142 (11.3%) | 244 | 4/21 | 497.7 s |
| `production_preprocessor_ilp` | 0.999 | 0.957 | 0.918 | 0.937 / 0.970 | 9/95 (9.5%) | 42 | 11/21 | 402.3 s |

## Metric definitions

- Pairwise precision/recall/F1 compare whether every pair of kept observations is assigned to the same identity. Missing outputs become unique singleton predictions.
- Micro pair metrics pool pair counts and therefore weight long identities quadratically. Macro metrics average the per-video scores.
- A contaminated predicted track contains observations from more than one gold identity. This is the catastrophic false-merge error for the application.
- Fragmentation excess is the sum, over gold identities, of the number of predicted pieces beyond one.
- An exact video has full kept-observation coverage and a partition identical to gold, up to permutation of track labels.
- Rejection metrics treat an omitted observation as rejected and gold identity 0 as the manually marked discard class. Tracker numeric label 0 has no special meaning.

## Findings

- The production pipeline has the strongest overall association F1, much less fragmentation, and the most exact videos.
- It does not satisfy the application’s nominal zero-false-merge requirement on this reconstruction: 9 of 95 emitted tracks are identity-contaminated, spanning 7 videos.
- OC-SORT is conservative: it has the best pairwise precision but fragments the 88 gold identities into hundreds of pieces.
- BoT-SORT reduces fragmentation relative to OC-SORT but still trails the production pipeline on recall, F1, and exact-video rate.
- These results document the intended offline-association trade-off—far fewer splits at the cost of more false merges—but do not support a claim of uniformly superior or near-perfect tracking.

## Reproduction

From the repository root, install `requirements.txt` and run:

```powershell
python -m video_processing.inference.optimization.compare_trackers --no-resume
```

The pinned BoxMOT distribution is 13.0.17; its package-level version string reports 13.0.16. A checkpoint is written under `tmp/` after every completed video-method pair.

## Tracker configurations

- **boxmot_oc_sort:** BoxMOT OC-SORT defaults frozen explicitly: min_conf=0.1, det_thresh=0.2, max_age=30, min_hits=3, IoU threshold=0.3, delta_t=3, inertia=0.2, BYTE disabled, Q_xy=0.01, Q_s=0.0001.
- **boxmot_bot_sort_gmc_no_reid:** BoxMOT BoT-SORT with ECC GMC and ReID disabled; thresholds 0.5/0.1/0.6, buffer=30, match=0.8, fuse-score disabled.
- **production_preprocessor_ilp:** Current TrackPreProcessor followed by current ILPTracker defaults; border-derived foreground masking, whole-fragment appearance prototypes, four-dimensional Kalman motion gating, conservative motion/area/aspect link vetoes, and masked VidStab transforms with 20 px observation masks; no parameter refitting.

## Limitations

- This is not a detector benchmark and does not support MOTA or HOTA claims. Every method receives the same saved observations.
- The manually curated files are the development set. The production tracker configuration is frozen for this run, but the provenance of its numerical tuning is uncertain.
- The saved non-interpolated boxes were emitted after gold-track reconstruction and RTS smoothing. The benchmark therefore measures association conditional on preselected, gold-derived smoothed observations.
- The historical preprocessor fragments are not preserved. The production pipeline starts from singleton saved observations and reruns preprocessing before ILP association.
- BoT-SORT is run without ReID because its standard appearance model is pedestrian-specific. It retains its own ECC camera-motion compensation.
- Runtime is wall-clock method execution on this machine. OC-SORT needs no video decode; BoT-SORT includes video decode and ECC GMC; production includes masked stabilization and association but excludes initial crop embedding extraction. Runtime values are therefore operational diagnostics, not a speed ranking.

See `benchmark_results.json` and `per_video.csv` for exact machine-readable values.
