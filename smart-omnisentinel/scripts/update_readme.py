path = r"C:\Users\OMHARI\Development\Projects\ML_Projects\smart-omnisentinel-final\smart-omnisentinel\README.md"
with open(path, "r", encoding="utf-8") as f:
    c = f.read()

target = "---\n\n## Project Structure"
replacement = """---

## SOTA Verification & Evaluation Suite

SmartOmniSentinel includes fully automated regression and verification suites covering all Phase 1 & Phase 2 deliverables:

```bash
# Run Phase 1 Verification Suite (Hardware exports, proximity, ST-GCN, Bi-GRU, ablation, cross-dataset)
python -m unittest tests/test_phase1_full.py -v

# Run Phase 2 Verification Suite (Audio 1D-CNN, scene-adaptive thresholds, TAL head, attention GIFs, weapon escalation)
python -m unittest tests/test_phase2_full.py -v
```

### Technical Deliverables & Publications
- **Full IEEE Technical Report:** [`report/final_report.md`](report/final_report.md)
- **4-Stage Ablation Benchmarks:** [`experiments/ablation_table.md`](experiments/ablation_table.md)
- **Audio Multimodal Benchmark:** [`experiments/audio_ablation.md`](experiments/audio_ablation.md)
- **Zero-Shot Cross-Dataset Benchmark:** [`experiments/cross_dataset.md`](experiments/cross_dataset.md)
- **Explainability Visualizations (4 Animated Diagnostic Cases):** [`docs/figures/`](docs/figures/)

---

## Project Structure"""

if target in c:
    c = c.replace(target, replacement, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(c)
    print("README.md updated successfully!")
else:
    print("Target not found in README.md")

