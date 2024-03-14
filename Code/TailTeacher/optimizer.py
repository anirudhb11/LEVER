import transformers

def prepare_optimizer_and_schedular(args, snet, t_total):
    """
    Set-up the optimizer and schedular

    * t_total has to be pre-calculated (lr will be zero after these many steps)
    """
    params = []
    # Adding encoder params
    enc_params = [param for param in snet.encoder.encoder.parameters()]
    params.append({'params': enc_params, 'lr': args.enc_lr, 'eps': 1e-06, 'weight_decay':args.enc_wd})
    
    optimizer = transformers.AdamW(params)
    print("Optimizer: ", optimizer)
    t_total = t_total * args.epochs
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    return optimizer, scheduler

