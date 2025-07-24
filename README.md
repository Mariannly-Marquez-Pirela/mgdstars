# MGDStars

A Python package for **Minimal Geometrical Deformation (MGD)** compact star calculations in modified gravity theories.

## 🌟 Features

- **Multiple seed models**: TolmanIV, HeintzmannIIa, WymanIIa
- **Robust numerical solver** with adaptive initial guess strategies
- **Comprehensive physical property evaluation**: density, pressure, anisotropy profiles
- **Model performance comparison** and convergence diagnostics
- **Professional visualization tools** for stellar structure analysis

## 🚀 Quick Start

### Installation

```bash
pip install mgdstars
```

### Basic Usage

```python
from mgdstars import MGDStars
import numpy as np

# Create MGD star solver
star = MGDStars(alpha=0.1, seed_model='TolmanIV')

# Solve for a neutron star configuration
result = star.solve_star_configuration(M_val=1.4, R_val=12.0)

if result['success']:
    print(f"Solution found!")
    print(f"A0 = {result['A0']:.4f}")
    print(f"B0 = {result['B0']:.6f}")
    print(f"C0 = {result['C0']:.4f}")
    
    # Evaluate physical properties
    r_range = np.linspace(0.1, 11.9, 100)
    properties = star.evaluate_at_radius(r_range)
    
    print(f"Central density: {properties['rho'][0]:.4f}")
    print(f"Central pressure: {properties['pr'][0]:.4f}")
```

### Advanced: Model Comparison

```python
# Compare different seed models
models = ['TolmanIV', 'HeintzmannIIa', 'WymanIIa']
results = {}

for model in models:
    star = MGDStars(alpha=0.01, seed_model=model)  # Note: smaller alpha for some models
    result = star.solve_star_configuration(M_val=1.4, R_val=12.0)
    results[model] = result
    
    if result['success']:
        print(f"{model}: SUCCESS (residual: {result['residual']:.2e})")
    else:
        print(f"{model}: FAILED")
```

## 📊 Seed Models

| Model | Metric Structure | Recommended α Range | Best For |
|-------|------------------|-------------------|----------|
| **TolmanIV** | `e^ν = B₀²(r²/A₀² + 1)` | 0.005 - 0.15 | General use, robust convergence |
| **HeintzmannIIa** | `e^ν = A₀²(B₀r² + 1)³` | 0.001 - 0.02 | Alternative parameterization |
| **WymanIIa** | `e^ν = (A₀ - B₀r²)²` | 0.001 - 0.008 | Specialized applications |

## 🔬 Examples

Check the `examples/` directory for comprehensive notebooks:
- `mgd_seed_models_comparison.ipynb`: Systematic comparison of all seed models
- `neutron_star_mgd_analysis_tolmaniv.ipynb`: Detailed TolmanIV analysis

## 📖 API Reference

### MGDStars Class

#### Constructor
```python
MGDStars(alpha=0.1, k2_factor=8*np.pi, seed_model='TolmanIV')
```

#### Key Methods
- `solve_star_configuration(M_val, R_val, **kwargs)`: Solve for stellar configuration
- `evaluate_at_radius(r_vals)`: Calculate physical properties at given radii
- `diagnose_convergence_failure(M_val, R_val)`: Debug convergence issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 📧 Contact

**Mariannly Marquez**  
📧 mariannly.marquez@yachaytech.edu.ec  
🏫 Yachay Tech University

## 🙏 Citation

If you use MGDStars in your research, please cite:

```bibtex
@software{mgdstars2024,
  author = {Marquez, Mariannly},
  title = {MGDStars: Minimal Geometrical Deformation compact star calculations},
  year = {2024},
  url = {https://github.com/mariannly/mgdstars}
}
```