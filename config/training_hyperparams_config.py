# ============================================================
# Training hyperparameters
# ============================================================
n_pde_samples_per_batch = 2048
n_boundary_samples = 512
n_terminal_samples = 1024
n_far_samples = 512

n_epochs = 30000
learning_rate = 1e-3

w_pde = 1.0
w_vm = 20.0
w_sp = 20.0
w_reflect = 2.0
w_term = 10.0
w_far = 2.0
w_stop = 10.0
w_premium = 2.0