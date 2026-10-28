def step(self, u, cache):  # u: (batch, d_model), a single token
    conv_window, h = cache
    x, z = self.in_proj(u).chunk(2, dim=-1)
    conv_window = torch.cat([conv_window[..., 1:], x.unsqueeze(-1)], dim=-1)
    x = F.silu((conv_window * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias)
    delta, B, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
    delta = F.softplus(self.dt_proj(delta)).unsqueeze(-1)  # (batch, d, 1)
    A = -torch.exp(self.A_log.float())
    h = torch.exp(delta * A) * h + delta * x.unsqueeze(-1) * B.unsqueeze(1)  # h_t = A_bar h_{t-1} + B_bar x_t
    y = torch.einsum("bdn,bn->bd", h, C) + self.D * x  # y_t = C_t h_t + D x_t
    return self.out_proj(y * F.silu(z)), (conv_window, h)
