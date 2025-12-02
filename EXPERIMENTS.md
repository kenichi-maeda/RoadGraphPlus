# Training Log

## Stage 1 ✅
- Checkpoint: `checkpoints_stage1/last.ckpt`
- Config: lr=1e-3, epochs=25, no edge training
- Result: Junction Loss=0.00285, Offset Loss=0.00528

## Stage 2 V1 ❌ 
- Checkpoint: `checkpoints_stage2_v1_failed/last.ckpt`
- Config: lr=1e-3, epochs=50
- Result: Val Edge F1=0.098, overfitting
- Issue: LR太高

## Stage 2 V2 ⚠️
- Checkpoint: `checkpoints_stage2_v2/last.ckpt`
- Config: lr=1e-4, weight_decay=1e-4, epochs=75
- Result: Train Edge F1=0.568, Val Edge F1=0.093
- Issue: 降低lr没用，问题一样

| Metric | V1 | V2 |
|--------|----|----|
| Val Edge F1 | 0.098 | 0.093 |
| Val Edge F1 (GT nodes) | 0.567 | 0.567 |

**结论**: edge模块本身能学，但预测的节点质量太差

---

## SpaceNet Moscow Data ✅
- 189张图，已转换并上传到`data_spacenet/`
- 准备做跨数据集测试

---

## Current Problems
1. Edge预测在val上崩了 (9.3% F1)
2. Eval遇到bug - test dataloader为空
3. Oscar网络不稳定

---

## Next Steps
1. 修eval用validation set
2. 跑data和Moscow的结果
3. 找Ken拿最好的参数 or ckpt
4. 改进方向:
   - Junction质量过滤 (min_gt_prob=0.2太松)
   - 跨数据集泛化研究
   - 对比学习改进edge prediction