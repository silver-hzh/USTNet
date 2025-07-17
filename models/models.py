def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'USTNet':
        assert (opt.dataset_mode == 'unaligned')
        from .USTNet import USTNet
        model = USTNet()
    elif opt.model == 'USTNetPT':
        assert (opt.dataset_mode == 'unaligned')
        from .USTNet_pretrain import USTNetPT
        model = USTNetPT()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model