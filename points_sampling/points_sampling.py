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

def sample_pde_points(n_samples, b_net, device):
    # Sample params r, sigma, T as before
    r, sigma, T = sample_params(n_samples, device=device)
    u_t = torch.rand(n_samples, 1, device=device)
    t = u_t * T

    with torch.no_grad():
        b = b_net(t, r, sigma, T)

    # Two groups: near-boundary and interior
    n_near = n_samples // 2
    n_int  = n_samples - n_near

    # 1) Near boundary: x ~ [b - eps, b]
    eps = 0.2  # in x units; tune to your domain
    t_near = t[:n_near]
    b_near = b[:n_near]
    r_near, sigma_near, T_near = r[:n_near], sigma[:n_near], T[:n_near]

    lo = torch.clamp(b_near - eps, min=x_min)
    u_near = torch.rand(n_near, 1, device=device)
    x_near = lo + u_near * (b_near - lo)

    # 2) Interior: x ~ [x_min, b - eps] (where possible)
    t_int = t[n_near:]
    b_int = b[n_near:]
    r_int, sigma_int, T_int = r[n_near:], sigma[n_near:], T[n_near:]

    lo_int = x_min * torch.ones_like(b_int)
    hi_int = torch.clamp(b_int - eps, min=x_min + 1e-3)
    u_int = torch.rand(n_int, 1, device=device)
    x_int = lo_int + u_int * (hi_int - lo_int)

    # Concatenate
    t_all = torch.cat([t_near, t_int], dim=0)
    x_all = torch.cat([x_near, x_int], dim=0)
    r_all = torch.cat([r_near, r_int], dim=0)
    sigma_all = torch.cat([sigma_near, sigma_int], dim=0)
    T_all = torch.cat([T_near, T_int], dim=0)

    return t_all, x_all, r_all, sigma_all, T_all


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