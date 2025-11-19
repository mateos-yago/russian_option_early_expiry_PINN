from config.training_hyperparams_config import *
from derivatives.derivatives import derivatives_V
from points_sampling import points_sampling
import torch

# ============================================================
# PINN loss
# ============================================================
def pinn_loss(value_net, boundary_net, device):
    # ---------- 1) PDE in continuation region ----------
    t_pde, x_pde, r_pde, sigma_pde, T_pde = points_sampling.sample_pde_points(n_pde_samples_per_batch, boundary_net, device)
    V_pde, V_t_pde, V_x_pde, V_xx_pde = derivatives_V(t_pde, x_pde, r_pde, sigma_pde, T_pde, value_net)

    # PDE: V_t - r x V_x + 0.5 sigma^2 x^2 V_xx = 0
    pde_residual = V_t_pde - r_pde * x_pde * V_x_pde + 0.5 * sigma_pde**2 * x_pde**2 * V_xx_pde
    loss_pde = torch.mean(pde_residual**2)

    # ---------- 2) Free boundary: value matching + smooth pasting ----------
    t_fb, x_fb, r_fb, sigma_fb, T_fb = points_sampling.sample_free_boundary_points(n_boundary_samples, boundary_net, device)
    V_fb, _, V_x_fb, _ = derivatives_V(t_fb, x_fb, r_fb, sigma_fb, T_fb, value_net)

    # Value matching: V(t, b(t)) = b(t)
    loss_vm = torch.mean((V_fb - x_fb)**2)

    # Smooth pasting: V_x(t, b(t)) = 1
    loss_sp = torch.mean((V_x_fb - 1.0)**2)

    # ---------- 3) Reflecting boundary at x = 1 ----------
    t_ref, x_ref, r_ref, sigma_ref, T_ref = points_sampling.sample_reflecting_boundary_points(n_boundary_samples, device)
    _, _, V_x_ref, _ = derivatives_V(t_ref, x_ref, r_ref, sigma_ref, T_ref, value_net)

    # V_x(t,1) = 0
    loss_reflect = torch.mean(V_x_ref**2)

    # ---------- 4) Terminal condition at t = T ----------
    t_term, x_term, r_term, sigma_term, T_term = points_sampling.sample_terminal_points(n_terminal_samples, device)
    V_term, _, _, _ = derivatives_V(t_term, x_term, r_term, sigma_term, T_term, value_net)

    # V(T,x) = x
    loss_term = torch.mean((V_term - x_term)**2)

    # ---------- 5) Far boundary at x = x_max ----------
    t_far, x_far, r_far, sigma_far, T_far = points_sampling.sample_far_boundary_points(n_far_samples, device)
    V_far, _, _, _ = derivatives_V(t_far, x_far, r_far, sigma_far, T_far, value_net)

    # V(t, x_max) = x_max
    loss_far = torch.mean((V_far - x_far)**2)

    # ---------- 6) Stopping region: x >= b(t) ----------
    t_stop, x_stop, r_stop, sigma_stop, T_stop = points_sampling.sample_stopping_region_points(n_boundary_samples, boundary_net, device)
    V_stop, _, _, _ = derivatives_V(t_stop, x_stop, r_stop, sigma_stop, T_stop, value_net)

    # V(t,x) = x in stopping region
    loss_stop = torch.mean((V_stop - x_stop)**2)

    # ---------- 7) Early-exercise premium positivity in continuation ----------
    # Enforce V(t,x) >= x in continuation region by penalizing (x - V)+
    premium_violation = torch.relu(x_pde - V_pde)
    loss_premium_pos = torch.mean(premium_violation**2)

    # ---------- Combine with weights ----------

    loss_total = (
        w_pde     * loss_pde +
        w_vm      * loss_vm +
        w_sp      * loss_sp +
        w_reflect * loss_reflect +
        w_term    * loss_term +
        w_far     * loss_far +
        w_stop    * loss_stop +
        w_premium * loss_premium_pos
    )

    return {
        "loss_total": loss_total,
        "loss_pde": loss_pde,
        "loss_vm": loss_vm,
        "loss_sp": loss_sp,
        "loss_reflect": loss_reflect,
        "loss_term": loss_term,
        "loss_far": loss_far,
        "loss_stop": loss_stop,
        "loss_premium_pos": loss_premium_pos,
    }