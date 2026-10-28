def sequential_scan(a, b):
    h, hs = torch.zeros_like(b[:, 0]), []
    for a_t, b_t in zip(a.unbind(1), b.unbind(1)):  # unbind, not a[:, t]: its backward is one stack, not L copies
        h = a_t * h + b_t
        hs.append(h)
    return torch.stack(hs, dim=1)
