class ResidualBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = MambaBlock(cfg) if cfg.arch == "mamba1" else Mamba2Block(cfg)

    def forward(self, x):
        return x + self.mixer(self.norm(x))

    def step(self, x, cache):
        y, cache = self.mixer.step(self.norm(x), cache)
        return x + y, cache


def forward(self, idx, targets=None):
    x = self.embedding(idx)
    for layer in self.layers:
        if self.grad_checkpoint and self.training:  # recompute in backward instead of storing
            x = checkpoint(layer, x, use_reentrant=False)
        else:
            x = layer(x)
    logits = self.lm_head(self.norm_f(x))
    if targets is None:
        return logits
    return logits, F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())

@torch.no_grad()
def generate(self, prompt_ids, max_new_tokens, temperature=1.0, top_k=0):
    device = self.embedding.weight.device
    cache = self.init_cache(1)
    for t in prompt_ids:  # read the prompt like the deck's notebook: one token at a time
        logits, cache = self.step(torch.tensor([t], device=device), cache)
    for _ in range(max_new_tokens):
        if temperature <= 0:
            next_id = logits.argmax(-1)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)[:, 0]
        yield next_id.item()
        logits, cache = self.step(next_id, cache)
