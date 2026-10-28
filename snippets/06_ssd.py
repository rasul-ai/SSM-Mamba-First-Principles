def ssd(x, log_decay, B, C, chunk):
    b, L, nh, p = x.shape
    pad = -L % chunk
    if pad:  # pad the end; later positions never affect earlier ones
        x, log_decay, B, C = (F.pad(t, (0, 0) * (t.dim() - 2) + (0, pad)) for t in (x, log_decay, B, C))
    c = x.size(1) // chunk
    x = x.reshape(b, c, chunk, nh, p)
    B, C = B.reshape(b, c, chunk, -1), C.reshape(b, c, chunk, -1)
    a = log_decay.reshape(b, c, chunk, nh).permute(0, 3, 1, 2)  # (batch, heads, chunks, chunk)
    a_cum = torch.cumsum(a, dim=-1)

    # 1. within each chunk: masked attention, scores = C . B, mask = decay matrix
    scores = torch.einsum("bcln,bcsn->bcls", C, B)
    y_diag = torch.einsum("bcls,bhcls,bcshp->bclhp", scores, torch.exp(segsum(a)), x)
    # 2. the state each chunk would leave behind if it had started from zero
    states = torch.einsum("bcln,bhcl,bclhp->bchpn", B, torch.exp(a_cum[..., -1:] - a_cum), x)
    # 3. pass states from chunk to chunk: the recurrence, one step per chunk
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    chunk_decay = torch.exp(segsum(F.pad(a_cum[..., -1], (1, 0))))
    states = torch.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)[:, :-1]  # the state entering each chunk
    # 4. what each position reads out of the state that entered its chunk
    y_off = torch.einsum("bcln,bchpn,bhcl->bclhp", C, states, torch.exp(a_cum))
    return (y_diag + y_off).reshape(b, c * chunk, nh, p)[:, :L]
