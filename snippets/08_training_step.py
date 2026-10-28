for step in range(1, args.steps + 1):
    lr = lr_at(step - 1)
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = get_batch(train_data, args.batch_size, args.seq_len, device)
    with amp():
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()
