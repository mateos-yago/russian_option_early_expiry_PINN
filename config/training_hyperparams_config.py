# ============================================================
# Training hyperparameters
# ============================================================
n_pde_samples_per_batch = 2048
n_boundary_samples = 1024
n_terminal_samples = 1024
n_far_samples = 512

max_epochs = 30000
learning_rate = 5.0e-4

# ============================================================
# Stopping / monitoring hyperparameters
# ============================================================
target_total_loss = 1.0e-4   # stop if total loss below this
target_pde_loss   = 1.0e-4   # and PDE loss below this

patience    = 10000           # epochs without sufficient improvement
min_delta   = 1.0e-5         # required improvement in loss to reset patience


# ============================================================
# Loss function weights
# ============================================================
w_pde = 30.0
w_vm = 20.0
w_sp = 20.0
w_reflect = 2.0
w_term = 10.0
w_far = 2.0
w_stop = 10.0
w_premium = 25.0