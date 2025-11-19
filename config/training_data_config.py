# ============================================================
# Parameter ranges for parametric PINN
# ============================================================
# r, sigma, T will be sampled in these ranges during training
r_min, r_max = 0.0, 0.10           # risk-free rate range
sigma_min, sigma_max = 0.10, 0.60  # volatility range
T_min, T_max = 0.5, 3.0            # maturity range in years

# Spatial domain for X = M / S
x_min = 1.0
x_max = 10.0