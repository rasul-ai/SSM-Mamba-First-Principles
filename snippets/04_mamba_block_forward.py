def forward(self, u):  # u: (batch, L, d_model)
    L = u.size(1)
    x, z = self.in_proj(u).chunk(2, dim=-1)  # 01-02
    x = self.conv1d(x.transpose(1, 2))[..., :L].transpose(1, 2)  # 03 (the first L outputs are the causal ones)
    x = F.silu(x)  # 04
    delta, B, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
    delta = F.softplus(self.dt_proj(delta))  # delta_t > 0
    A = -torch.exp(self.A_log.float())
    mode = self.scan if self.scan != "auto" else ("sequential" if u.device.type == "cpu" else "parallel")
    y = selective_scan(x, delta, A, B, C, self.D, mode)  # 05 Select and scan
    y = y * F.silu(z)  # 06 Gate: the z path's SiLU is the volume knob
    return self.out_proj(y)  # 07
