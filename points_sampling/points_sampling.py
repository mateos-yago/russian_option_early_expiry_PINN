import torch
from config.training_data_config import *

# ============================================================
# Sampling utilities
# ============================================================
def sample_params(n_samples, device):
    """Sample (r, sigma, T) in the predefined ranges."""
    r = r_min + (r_max - r_min) * torch.rand(n_samples, 1, device=device)
    sigma = sigma_min + (sigma_max - sigma_min) * torch.rand(n_samples, 1, device=device)
    T = T_min + (T_max - T_min) * torch.rand(n_samples, 1, device=device)
    return r, sigma, T

def sample_pde_points(n_samples, boundary_net, device):
    """
    Continuation-region points: t in [0,T], x in [x_min, b(t)].
    Used to enforce the PDE and premium positivity.
    """
    r, sigma, T = sample_params(n_samples, device)
    u_t = torch.rand(n_samples, 1, device=device)
    t = u_t * T  # t in [0, T] for each sample

    with torch.no_grad():
        b = boundary_net(t, r, sigma, T)

    u_x = torch.rand(n_samples, 1, device=device)
    x = x_min + u_x * (b - x_min)  # x in [x_min, b(t)]

    return t, x, r, sigma, T

def sample_free_boundary_points(n_samples, boundary_net, device):
    """
    Free-boundary points: x = b(t).
    Used for value matching and smooth pasting.
    """
    r, sigma, T = sample_params(n_samples, device)
    u_t = torch.rand(n_samples, 1, device=device)
    t = u_t * T
    x_b = boundary_net(t, r, sigma, T)
    return t, x_b, r, sigma, T

def sample_reflecting_boundary_points(n_samples, device):
    """
    Reflecting boundary points: x = x_min = 1.
    Used for V_x(t,1) = 0.
    """
    r, sigma, T = sample_params(n_samples, device)
    u_t = torch.rand(n_samples, 1, device=device)
    t = u_t * T
    x = torch.full_like(t, x_min)
    return t, x, r, sigma, T

def sample_terminal_points(n_samples, device):
    """
    Terminal condition: t = T, x in [x_min, x_max].
    Used for V(T,x) = x.
    """
    r, sigma, T = sample_params(n_samples, device)
    t = T.clone()  # t = T elementwise
    x = x_min + (x_max - x_min) * torch.rand(n_samples, 1, device=device)
    return t, x, r, sigma, T

def sample_far_boundary_points(n_samples, device):
    """
    Far boundary: x = x_max, t in [0,T].
    Used for V(t, x_max) = x_max.
    """
    r, sigma, T = sample_params(n_samples, device)
    u_t = torch.rand(n_samples, 1, device=device)
    t = u_t * T
    x = torch.full_like(t, x_max)
    return t, x, r, sigma, T

def sample_stopping_region_points(n_samples, boundary_net, device):
    """
    Stopping region: x in [b(t), x_max].
    Used for V(t,x) = x when x >= b(t).
    """
    r, sigma, T = sample_params(n_samples, device)
    u_t = torch.rand(n_samples, 1, device=device)
    t = u_t * T

    with torch.no_grad():
        b = boundary_net(t, r, sigma, T)

    u_x = torch.rand(n_samples, 1, device=device)
    x = b + u_x * (x_max - b)  # x in [b(t), x_max]

    return t, x, r, sigma, T