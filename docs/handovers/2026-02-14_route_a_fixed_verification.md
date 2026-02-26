# Handover — 2026-02-14 — Route A-Fixed Experiments Complete

## What was accomplished
- Verified all 3 preprocessing fixes (RobustScaler, Partial Recon Loss, Target Encoding)
- Generated fixed embeddings for all 4 splits (128-dim)
- Created and ran 3 new pipeline scripts (Steps 1.4A, 1.4B, 1.5A, 1.5B)
- Updated `WORKING_PLAN.md` Phase 5.6 with full results
- Updated `decision_log.md` with experiment results and conclusions
- Updated Final Conclusion and Extended Routes Summary

## Current state
- **Phase**: 5.6 — Route A-Fixed (WORKING_PLAN.md) — **ALL STEPS COMPLETE**
- **Remaining**: Step 1.7 (Owner Code Review)

### Key Results

| Step | Model | Test AUC | vs Baseline |
|------|-------|----------|-------------|
| 1.4A | Student (emb) | **0.7021** | +0.0109 vs A |
| 1.4B | Teacher (raw+emb) | **0.7362** | −0.0029 vs B |
| 1.4B | Student (raw+emb) | 0.7328 | −0.0003 vs C |
| 1.5B | Ensemble (raw+emb) | 0.7294 | D_R: −0.0097 vs F |

### Key Finding
**Preprocessing fixes work (+0.0164 Teacher AUC), but pseudo-labeling is the bottleneck — not embedding quality.**

### Files modified this session
- `docs/WORKING_PLAN.md` — Phase 5.6 results, Extended Routes Summary, Final Conclusion
- `docs/decision_log.md` — Full experiment results and analysis
- `src/models/route_a_fixed_hybrid.py` (NEW) — Step 1.4A
- `src/models/route_c_fixed_hybrid.py` (NEW) — Step 1.4B
- `src/models/route_f_fixed_ensemble.py` (NEW) — Steps 1.5A+1.5B

## Test results

| Fix | Verification |
|-----|-------------|
| RobustScaler | Variance 0.000046 → 0.032456 (704.3x) ✅ |
| Partial Loss | 116 cols masked, 23.8% cells ✅ |
| Target Encoding | 58+18 OHE → 2 continuous cols ✅ |
| Fixed embeddings | 4 splits × 128-dim generated ✅ |
| All 4 experiments | Completed with results saved ✅ |

## Next steps (for the next agent/conversation)
1. Owner Code Review (Step 1.7) of Route A-Fixed results
2. Phase 7: Final Report & Submission
3. Consider future work: end-to-end fine-tuning, improved pseudo-labeling

## Open questions / blockers
- None — all experiments complete, documentation updated

## Key files to read first
- `AI_GUIDE.md`
- `docs/WORKING_PLAN.md` (Phase 5.6 — full results)
- `docs/decision_log.md` (last entry — experiment conclusions)
- `outputs/results/route_a_fixed_metrics.json`
- `outputs/results/route_c_fixed_metrics.json`
- `outputs/results/route_f_fixed_results.json`
