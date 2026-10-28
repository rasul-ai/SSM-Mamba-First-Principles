def selective_scan(x, delta, A, B, C, D, mode="parallel"):
    with _no_autocast(x):
        x, delta, B, C = x.float(), delta.float(), B.float(), C.float()
        # Discretize (Figure 3). Zero-order hold gives A_bar = exp(delta * A). For B the official
        # implementation uses the simpler first-order form B_bar = delta * B, and so do we.
        A_bar = torch.exp(delta.unsqueeze(-1) * A)  # (batch, L, d, n)
        Bx_bar = (delta * x).unsqueeze(-1) * B.unsqueeze(2)  # (batch, L, d, n)
        h = (parallel_scan if mode == "parallel" else sequential_scan)(A_bar, Bx_bar)
        y = torch.einsum("bldn,bln->bld", h, C)  # y_t = C_t h_t
        return y + x * D  #     + D x_t
