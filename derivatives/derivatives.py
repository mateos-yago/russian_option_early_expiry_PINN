import torch

# ============================================================
# Derivatives via autograd
# ============================================================
def derivatives_V(t, x, r, sigma, T, value_net):
    """
    Compute V, V_t, V_x, V_xx for given (t, x, r, sigma, T).
    """
    t.requires_grad_(True)
    x.requires_grad_(True)

    V = value_net(t, x, r, sigma, T)  # (N,1)

    V_t = torch.autograd.grad(
        V, t,
        grad_outputs=torch.ones_like(V),
        retain_graph=True,
        create_graph=True
    )[0]

    V_x = torch.autograd.grad(
        V, x,
        grad_outputs=torch.ones_like(V),
        retain_graph=True,
        create_graph=True
    )[0]

    V_xx = torch.autograd.grad(
        V_x, x,
        grad_outputs=torch.ones_like(V_x),
        retain_graph=True,
        create_graph=True
    )[0]

    return V, V_t, V_x, V_xx