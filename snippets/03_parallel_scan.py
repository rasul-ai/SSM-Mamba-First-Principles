def parallel_scan(a, b):
    L = b.size(1)
    if L == 1:
        return b
    if L % 2:  # odd length: append an identity step (a=1, b=0) so that every token has a partner
        a = torch.cat([a, torch.ones_like(a[:, :1])], dim=1)
        b = torch.cat([b, torch.zeros_like(b[:, :1])], dim=1)
    (a_even, a_odd), (b_even, b_odd) = a.unflatten(1, (-1, 2)).unbind(2), b.unflatten(1, (-1, 2)).unbind(2)
    h_odd = parallel_scan(a_odd * a_even, a_odd * b_even + b_odd)  # the state after each pair
    h_before = torch.cat([torch.zeros_like(h_odd[:, :1]), h_odd[:, :-1]], dim=1)
    h_even = a_even * h_before + b_even  # the state after the first token of each pair
    return torch.stack([h_even, h_odd], dim=2).flatten(1, 2)[:, :L]
