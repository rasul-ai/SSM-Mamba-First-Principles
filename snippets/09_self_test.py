# 2. Reading a sequence all at once == reading it one token at a time with a fixed-size state.
for arch in ("mamba1", "mamba2"):
    cfg = MambaConfig(vocab_size=13, arch=arch, d_model=32, n_layers=2, headdim=16, chunk_size=8, scan="parallel")
    model = MambaLM(cfg).eval()
    idx = torch.randint(0, 13, (2, 37))  # 37 is deliberately not a multiple of the chunk size
    with torch.no_grad():
        full = model(idx)
        cache, steps = model.init_cache(2), []
        for t in range(idx.size(1)):
            logits, cache = model.step(idx[:, t], cache)
            steps.append(logits)
        stepwise = torch.stack(steps, dim=1)
        if arch == "mamba1":
            for layer in model.layers:
                layer.mixer.scan = "sequential"
            assert torch.allclose(model(idx), full, atol=1e-5), "mamba1: sequential and parallel scans differ"
    err = (full - stepwise).abs().max().item()
    assert err < 1e-4, f"{arch}: forward and step-by-step differ by {err:.2e}"
    print(f"ok  {arch}: whole-sequence forward == token-by-token recurrence (max difference {err:.1e})")
