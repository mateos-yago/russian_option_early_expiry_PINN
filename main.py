import torch
from nets.nets import ValueNet, BoundaryNet
from config.training_hyperparams_config import *
from torch import optim
from loss_function.loss_function import pinn_loss

def main():

    # ============================================================
    # Device
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate networks and optimizer
    V_net = ValueNet().to(device)
    b_net = BoundaryNet().to(device)

    params = list(V_net.parameters()) + list(b_net.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss_dict = pinn_loss(V_net, b_net, device=device)
        loss = loss_dict["loss_total"]
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"Total: {loss_dict['loss_total']:.3e} | "
                f"PDE: {loss_dict['loss_pde']:.3e} | "
                f"VM: {loss_dict['loss_vm']:.3e} | "
                f"SP: {loss_dict['loss_sp']:.3e} | "
                f"Ref: {loss_dict['loss_reflect']:.3e} | "
                f"Term: {loss_dict['loss_term']:.3e} | "
                f"Far: {loss_dict['loss_far']:.3e} | "
                f"Stop: {loss_dict['loss_stop']:.3e} | "
                f"Prem+: {loss_dict['loss_premium_pos']:.3e}"
            )

if __name__ == "__main__":
    main()