# Parameter Reference

Typical biological parameter values for PhysiCell simulations. All rates are in 1/min, volumes in μm³, speeds in μm/min, distances in μm.

## Cell Volumes

| Cell Type | Total Volume (μm³) | Nuclear Volume (μm³) | Fluid Fraction | Radius (μm) |
|-----------|--------------------:|---------------------:|---------------:|-------------:|
| Generic epithelial | 2494 | 540 | 0.75 | 8.41 |
| Tumor cell | 2494 | 540 | 0.75 | 8.41 |
| Fibroblast | 2000-4000 | 400-800 | 0.75 | 7.8-9.8 |
| Macrophage | 1767 | 500 | 0.75 | 7.5 |
| CD8+ T cell | 478 | 120 | 0.75 | 4.84 |
| NK cell | 600 | 150 | 0.75 | 5.2 |
| Neutrophil | 300 | 80 | 0.75 | 4.15 |
| Endothelial | 1500 | 350 | 0.75 | 7.1 |

## Motility Parameters

| Cell Type | Speed (μm/min) | Persistence (min) | Migration Bias |
|-----------|----------------:|-------------------:|---------------:|
| Tumor (low motility) | 0.1-0.5 | 5-15 | 0.0-0.3 |
| Tumor (invasive) | 0.5-2.0 | 5-30 | 0.3-0.8 |
| Fibroblast | 0.5-1.0 | 10-30 | 0.0-0.5 |
| Macrophage | 1.0-4.0 | 5-10 | 0.5-0.9 |
| CD8+ T cell | 4.0-10.0 | 2-5 | 0.5-0.9 |
| NK cell | 5.0-15.0 | 2-5 | 0.5-0.9 |
| Neutrophil | 5.0-20.0 | 1-3 | 0.7-0.95 |
| Non-motile (epithelial) | 0.0 | — | — |

## Death Rates

| Cell Type | Apoptosis Rate (1/min) | Necrosis Rate (1/min) | Notes |
|-----------|-----------------------:|----------------------:|-------|
| Normal epithelial | 5.31667e-5 | 0.0 | ~1% per day |
| Slow-growing tumor | 5.31667e-5 | 0.00277 | Low apoptosis, hypoxia-driven necrosis |
| Aggressive tumor | 1e-4 to 5e-4 | 0.00277 | Higher turnover |
| Immune cell (activated) | 1.67e-4 | 0.0 | ~10% per day |
| Immune cell (exhausted) | 5e-4 to 1e-3 | 0.0 | Rapid turnover |

Key conversions:
- 1% per day ≈ 6.94e-6 per min
- 5% per day ≈ 3.57e-5 per min
- 10% per day ≈ 7.27e-5 per min
- Necrosis rate 0.00277 ≈ ~100% probability within hours under anoxia

## Substrate Parameters

### Oxygen
| Parameter | Value | Units |
|-----------|------:|-------|
| Diffusion coefficient | 100000 | μm²/min |
| Decay rate | 0.1 | 1/min |
| Initial condition | 38.0 | mmHg |
| Dirichlet boundary | 38.0 (enabled) | mmHg |
| Typical uptake rate | 10.0 | 1/min |
| Hypoxia threshold | 5-15 | mmHg |
| Severe hypoxia | <5 | mmHg |
| Normoxia | 30-40 | mmHg |

### Glucose
| Parameter | Value | Units |
|-----------|------:|-------|
| Diffusion coefficient | 30000 | μm²/min |
| Decay rate | 0.0025 | 1/min |
| Initial condition | 16.9 | mM |
| Dirichlet boundary | 16.9 (enabled) | mM |
| Typical uptake rate | 0.5-2.0 | 1/min |

### Chemokines (generic)
| Parameter | Value | Units |
|-----------|------:|-------|
| Diffusion coefficient | 50000 | μm²/min |
| Decay rate | 0.01 | 1/min |
| Initial condition | 0.0 | dimensionless |
| Dirichlet boundary | not enabled | — |
| Typical secretion rate | 0.01-1.0 | 1/min |

### Drugs (generic small molecule)
| Parameter | Value | Units |
|-----------|------:|-------|
| Diffusion coefficient | 100000 | μm²/min |
| Decay rate | 0.01-0.1 | 1/min |
| Initial condition | 0.0 | μM |
| Dirichlet boundary | varies | μM |
| Typical uptake rate | 0.01-0.1 | 1/min |

## Hill Function Parameters

Hill function parameters (half_max, hill_power) for cell rules must be derived from literature validation using `validate_rules_batch()`. Do NOT use hardcoded default values — these parameters are model-specific and must be justified by experimental evidence.

General guidance:
- **hill_power 2-4**: Graded, dose-dependent response
- **hill_power 6-10**: Sharp, switch-like threshold response
- **half_max**: Should correspond to the EC50/IC50 from experimental dose-response data

## Cell Cycle Models

| Model | Default Transition Rate | Typical Doubling Time | Best for |
|-------|------------------------:|----------------------:|----------|
| Ki67_basic | ~0.00072 (1/min) | ~24 hours | Simple proliferating populations |
| Ki67_advanced | varies by phase | varies | Detailed Ki67 dynamics |
| live | ~0.00072 (1/min) | ~24 hours | Simple live cell model |
| cycling_quiescent | ~0.00072 / 0.0 | varies | Cells with quiescent states |
| flow_cytometry | varies by phase | varies | Matching flow cytometry data |
| flow_cytometry_separated | varies by phase | varies | Detailed cell cycle phases |

Key doubling time conversions:
- 12 hours: rate ≈ 0.00096/min
- 24 hours: rate ≈ 0.00048/min (or ~0.00072 for Ki67 model)
- 48 hours: rate ≈ 0.00024/min
- 72 hours: rate ≈ 0.00016/min

## Domain Size Guidelines

| Scenario | Domain Size (μm) | dx (μm) | Notes |
|----------|:-----------------:|:-------:|-------|
| Small spheroid | 500 × 500 × 20 | 20 | Quick test, <1000 cells |
| Medium tumor | 1000 × 1000 × 20 | 20 | Standard 2D simulation |
| Large tumor | 2000 × 2000 × 20 | 20 | Large 2D, slower |
| Tissue section | 3000 × 1000 × 20 | 20 | Rectangular tissue |
| 3D spheroid | 500 × 500 × 500 | 20 | True 3D (very slow) |

For 2D simulations, set `domain_z=20` and `dx=20` (single voxel in z).

## Simulation Time Guidelines

| Scenario | Time (min) | Notes |
|----------|:----------:|-------|
| Quick test | 720 (12 hr) | Verify setup |
| Short simulation | 1440 (1 day) | Early dynamics |
| Standard run | 4320 (3 days) | Most biological scenarios |
| Extended run | 7200 (5 days) | Long-term dynamics |
| Week simulation | 10080 (7 days) | Slow processes |
