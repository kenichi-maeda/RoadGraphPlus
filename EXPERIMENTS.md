# Training Log

## Stage 1 ✅
- **Checkpoint:** checkpoints_stage1/last.ckpt
- **Config:** lr=1e-3, epochs=25, lambda_e=0 (no Edge)
- **Result:** Junction/Offset trained successfully

## Stage 2 V1 ❌ FAILED
- **Checkpoint:** checkpoints_stage2_v1_failed/last.ckpt
- **Config:** lr=1e-3, epochs=50, warmup=10
- **Result:** Val Edge F1=0.098 (10%), severe overfitting
- **Problem:** LR too high, Junction degraded (recall 1.0→0.5)

## Stage 2 V2 🔄 (Current)
- **Checkpoint:** checkpoints_stage2_v2/
- **Changes:** lr=1e-4, weight_decay=1e-4, epochs=75, warmup=15, gradient_clip=1.0, early_stop
- **Target:** Val Edge F1 > 0.2, maintain Junction recall > 0.7

| Param | V1 | V2 |
|-------|----|----|
| lr | 1e-3 | **1e-4** |
| weight_decay | 1e-5 | **1e-4** |
| gradient_clip | None | **1.0** |
| early_stop | No | **Yes** |