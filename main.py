import torch
import torch.optim as optim
import os

from nets.nets import BoundaryNet, ValueNet
from config.training_hyperparams_config import *
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

    # ============================================================
    # Stopping / monitoring hyperparameters (tune as needed)
    # ============================================================
    target_total_loss = 1.0e-4   # stop if total loss below this
    target_pde_loss   = 1.0e-4   # and PDE loss below this

    patience    = 5000           # epochs without sufficient improvement
    min_delta   = 1.0e-5         # required improvement in loss to reset patience
    print_every = 500            # how often to print progress

    # ============================================================
    # Early stopping state
    # ============================================================
    best_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    os.makedirs("checkpoints", exist_ok=True)
    best_ckpt_path = "checkpoints/ckpt_best_russian.pt"

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        loss_dict = pinn_loss(V_net, b_net, device=device)
        loss = loss_dict["loss_total"]
        loss.backward()
        optimizer.step()

        # Convert to Python floats for comparisons/logging
        total_loss_val = float(loss.item() if hasattr(loss, "item") else loss)
        pde_loss_val = float(
            loss_dict["loss_pde"].item()
            if hasattr(loss_dict["loss_pde"], "item")
            else loss_dict["loss_pde"]
        )

        # ========================================================
        # Logging
        # ========================================================
        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:6d} | "
                f"Total: {total_loss_val:.3e} | "
                f"PDE: {pde_loss_val:.3e} | "
                f"VM: {loss_dict['loss_vm']:.3e} | "
                f"SP: {loss_dict['loss_sp']:.3e} | "
                f"Ref: {loss_dict['loss_reflect']:.3e} | "
                f"Term: {loss_dict['loss_term']:.3e} | "
                f"Far: {loss_dict['loss_far']:.3e} | "
                f"Stop: {loss_dict['loss_stop']:.3e} | "
                f"Prem+: {loss_dict['loss_premium_pos']:.3e}"
            )

        # ========================================================
        # Check for new best model
        # ========================================================
        if total_loss_val + min_delta < best_loss:
            best_loss = total_loss_val
            best_epoch = epoch
            epochs_no_improve = 0  # reset patience

            ckpt = {
                "epoch": epoch,
                "value_net_state": V_net.state_dict(),
                "boundary_net_state": b_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": best_loss,
                "losses": {
                    k: float(v.item() if hasattr(v, "item") else v)
                    for k, v in loss_dict.items()
                },
            }
            torch.save(ckpt, best_ckpt_path)
        else:
            epochs_no_improve += 1

        # ========================================================
        # Target-loss based stopping
        # ========================================================
        if (total_loss_val <= target_total_loss) and (pde_loss_val <= target_pde_loss):
            print(
                f"Stopping: target losses reached at epoch {epoch} "
                f"(Total={total_loss_val:.4e}, PDE={pde_loss_val:.4e})"
            )
            break

        # ========================================================
        # Patience-based early stopping
        # ========================================================
        if epochs_no_improve >= patience:
            print(
                f"Stopping: no improvement greater than {min_delta:.2e} "
                f"for {patience} epochs. Best loss {best_loss:.4e} at epoch {best_epoch}."
            )
            break

    # ============================================================
    # Final save: load best model and save clean state dicts
    # ============================================================
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        V_net.load_state_dict(best_ckpt["value_net_state"])
        b_net.load_state_dict(best_ckpt["boundary_net_state"])
        optimizer.load_state_dict(best_ckpt["optimizer_state"])
        print(
            f"Loaded best model from epoch {best_ckpt['epoch']} "
            f"with loss {best_ckpt['loss']:.4e}"
        )
    else:
        print("Warning: no best checkpoint found; saving last-epoch parameters.")

    torch.save(V_net.state_dict(), "checkpoints/value_net_final_russian.pt")
    torch.save(b_net.state_dict(), "checkpoints/boundary_net_final_russian.pt")
    torch.save(optimizer.state_dict(), "checkpoints/optimizer_final_russian.pt")


if __name__ == "__main__":
    main()
