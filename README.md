# The Geometry of Multi-Task Grokking: Transverse Instability, Superposition, and Weight Decay Phase Structure

We extend geometric analysis of grokking to multi-task modular arithmetic, training shared-trunk Transformers on dual-task (mod-add + mod-mul) and tri-task (mod-add + mod-mul + mod-sq) objectives across a systematic weight decay sweep (up to 90 runs). We combine trajectory PCA, commutator defect analysis, Hessian eigenspectra, and causal gradient perturbations to characterize the geometry of multi-task generalization. We find that tasks grok in a fixed order with near-orthogonal heads; trajectories remain confined to an integrable low-dimensional manifold; weight decay acts as a phase parameter; solutions are holographically incompressible; and removing <10% of orthogonal gradient components eliminates grokking, revealing transverse fragility and redundant center manifolds in overparameterized models.

Code and figures for the paper:

> **The Geometry of Multi-Task Grokking: Transverse Instability, Superposition, and Weight Decay Phase Structure**
> Yongzhong Xu (2026)

## Key Findings

We train shared-trunk Transformers on dual-task (mod-add + mod-mul) and tri-task (mod-add + mod-mul + mod-sq) modular arithmetic across a systematic weight decay sweep (up to 90 runs), and discover five consistent phenomena:

1. **Staggered grokking order** &mdash; Multiplication generalizes first, followed by squaring and then addition, with consistent delays across seeds.

2. **Universal integrability** &mdash; Optimization trajectories remain confined to a low-dimensional execution manifold with near-perfect integrability (&rho; &asymp; 1.000). Commutator defect onset reliably precedes generalization (42/42 conditions).

3. **Weight decay phase structure** &mdash; Grokking timescale, curvature depth, reconstruction threshold, and defect lead covary systematically with weight decay, revealing distinct dynamical regimes. Zero decay has curvature but no grokking.

4. **Holographic incompressibility** &mdash; Solutions occupy only 4&ndash;8 PCA directions yet are distributed across full-rank weights and destroyed by minimal perturbation. Neither SVD, pruning, nor &plusmn;5% scaling preserves performance.

5. **Transverse fragility and redundancy** &mdash; Removing <10% of orthogonal gradient components eliminates grokking. Dual-task models partially recover at extreme deletion (50%), suggesting redundant center manifolds; tri-task models do not.

<p align="center">
  <img src="modadd_modmul/plots/figMT_K_integrability_s42.png" width="45%" alt="Integrability">
  <img src="tri_task/plots/figTT_W2_hero_defect_predicts_grok.png" width="45%" alt="Defect predicts grokking">
</p>
<p align="center"><em>Left: Near-perfect integrability (&rho; &asymp; 1.000) in dual-task training. Right: Commutator defect onset precedes grokking across tri-task conditions.</em></p>

## Model

- **Architecture**: 2-layer Transformer encoder, pre-norm, d_model=128, 4 heads, d_ff=256, GELU, no dropout
- **Dual-task**: ~303k params, shared trunk + 2 heads for (x+y) mod 97 and (x&middot;y) mod 97
- **Tri-task**: ~315k params, shared trunk + 3 heads (adds x&sup2;+y&sup2; mod 97)
- **Optimizer**: AdamW, lr=1e-3, &beta;&#8322;=0.98, batch 512, grad clip 1.0
- **Weight decay sweep**: &lambda; &isin; {0.0, 0.1, 0.2, 0.3, 0.5, 1.0} &times; 3 seeds (42, 137, 2024)

## Repository Structure

```
multitask-grokking/
├── modadd_modmul/          # Dual-task experiments (mod-add + mod-mul)
│   ├── train_multitask.py          # Training script (all WD × seeds)
│   ├── pca_analysis.py             # Trajectory PCA and manifold analysis
│   ├── commutator_analysis.py      # Commutator defect and integrability
│   ├── hessian_analysis.py         # Hessian bottom eigenvalue estimation
│   ├── hessian_aggregate.py        # Multi-seed Hessian summary
│   ├── hessian_wd*.py              # Per-condition Hessian runs
│   ├── pca_threshold_scan.py       # Reconstruction threshold k* scan
│   ├── gradient_projection_ablation.py  # Orthogonal gradient deletion
│   ├── eigenvector_ablation*.py    # Eigenvector ablation variants
│   ├── postgrok_compression.py     # SVD/pruning/scaling compression tests
│   ├── snapshot_decomposition.py   # Temporal decomposition analysis
│   ├── defect_wd_comparison.py     # Cross-WD defect comparison
│   ├── fit_wd_groktime.py          # WD vs grok time analysis
│   └── plots/                      # 125 figures (figMT_*, figTHR_*, etc.)
│
├── tri_task/               # Tri-task experiments (+ mod-sq)
│   ├── train_tritask.py            # Training script
│   ├── train_wd02_03.py            # Additional WD conditions
│   ├── pca_analysis.py             # Trajectory PCA
│   ├── commutator_analysis.py      # Commutator defect
│   ├── hessian_analysis.py         # Hessian eigenvalues
│   ├── generalization_dynamics.py  # Defect-grokking lead time analysis
│   ├── pca_threshold_scan.py       # Reconstruction threshold k*
│   ├── ortho_fine_scan.py          # Fine-grained orthogonal deletion
│   └── plots/                      # 79 figures (figTT_*, figTRI_*, etc.)
│
└── .gitignore              # Excludes .pt checkpoints (~20GB)
```

## Running the Experiments

### Prerequisites

```bash
pip install torch numpy matplotlib scipy tqdm
```

### 1. Train models

```bash
# Dual-task: trains all 18 conditions (6 WD × 3 seeds)
python modadd_modmul/train_multitask.py

# Tri-task: trains all 18 conditions
python tri_task/train_tritask.py
python tri_task/train_wd02_03.py  # WD=0.2, 0.3 conditions
```

Checkpoints are saved to `results/` subdirectories (~20GB total, excluded from git).

### 2. Analysis pipeline

Run in order after training completes:

```bash
# ── Dual-task analyses ──
python modadd_modmul/pca_analysis.py              # Trajectory PCA, eigenspectra
python modadd_modmul/commutator_analysis.py        # Commutator defect, integrability
python modadd_modmul/hessian_analysis.py           # Hessian bottom eigenvalue
python modadd_modmul/pca_threshold_scan.py         # Reconstruction threshold k*
python modadd_modmul/postgrok_compression.py       # Compression tests (SVD, pruning, scaling)
python modadd_modmul/gradient_projection_ablation.py  # Orthogonal gradient deletion

# ── Tri-task analyses ──
python tri_task/pca_analysis.py
python tri_task/commutator_analysis.py
python tri_task/hessian_analysis.py
python tri_task/pca_threshold_scan.py
python tri_task/generalization_dynamics.py         # Defect lead time analysis
python tri_task/ortho_fine_scan.py                 # Fine-grained deletion scan
```

All scripts produce figures in their respective `plots/` directories.

## Key Figures

| Figure | File | Description |
|--------|------|-------------|
| Training curves | `figMT_A_accuracy_s42.png` | Dual-task accuracy showing staggered grokking |
| Integrability | `figMT_K_integrability_s42.png` | &rho; &asymp; 1.000 across training |
| Defect traces | `figMT_J_defect_s42.png` | Commutator defect with orthogonal projection |
| Hessian phase | `figMT_H18_combined_dual.png` | Grok time vs curvature depth across WD |
| Compression | `figCOMP_A_compression.png` | SVD/pruning/scaling all fail |
| Threshold k* | `figTHR_B_threshold_vs_wd.png` | Reconstruction threshold vs weight decay |
| Defect predicts grokking | `figTT_W2_hero_defect_predicts_grok.png` | Defect onset leads grokking (tri-task) |
| Orthogonal deletion | `figORTHO_fine_dose_response.png` | Dose-response with 10% fragility cliff |
| Lead time | `figTT_X_defect_lead_time.png` | Defect lead time across WD |

## Companion Papers

- [Low-Dimensional Execution Manifolds in Transformer Learning Dynamics](https://arxiv.org/abs/2602.10496) &mdash; Single-task geometric analysis (arXiv:2602.10496)
- [Low-Dimensional and Transversely Curved Optimization Dynamics in Grokking](https://arxiv.org/abs/2602.16746) &mdash; Companion paper on integrability and curvature (arXiv:2602.16746)
- [Early-Warning Signals of Grokking via Loss-Landscape Geometry](https://arxiv.org/abs/2602.16967) &mdash; Extension to Dyck languages and SCAN benchmark (arXiv:2602.16967)

## Citation

```bibtex
@article{xu2026multitask,
  title={The Geometry of Multi-Task Grokking: Transverse Instability, Superposition, and Weight Decay Phase Structure},
  author={Xu, Yongzhong},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
