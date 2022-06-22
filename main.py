import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, SelfCriticalDataset
from train import run
import opts

import pdb
import transformers

def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data, a=0.01)


if __name__ == '__main__':
    opt = opts.parse_opt()
    
    ### fail safe
    if opt.dataset == 'hatcp':
        assert opt.impt_threshold == 0.55
    elif opt.dataset == 'gqacp':
        assert opt.impt_threshold == 0.3
    elif opt.dataset == 'clevrxai':
        assert opt.impt_threshold == 0.85
        
    ## Set random seeds for reproducibility
    if opt.seed == 0:
        seed = random.randint(1, 10000)
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        seed = opt.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True  # For reproducibility
    
    # load dictionary
    dictionary = Dictionary.load_from_file(f'{opt.data_dir}/dictionary.pkl')
    opt.ntokens = dictionary.ntoken
    print("dictionary ntoken", dictionary.ntoken)

    if opt.use_scr_loss:
        opt.apply_answer_weight = True

    ### creating datasets
    # train dataset
    if opt.split is not None:
        train_dset = SelfCriticalDataset(opt.split, opt.hint_type, dictionary, opt,
                                         discard_items_without_hints=not opt.do_not_discard_items_without_hints)
        train_loader = DataLoader(train_dset,
                                  opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    else:
        train_dset = None
        train_loader = None
    # val dataset
    eval_dset = SelfCriticalDataset(opt.split_test, opt.hint_type, dictionary, opt, 
                                       discard_items_without_hints = opt.discard_items_for_test)
    eval_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    if opt.use_two_testsets:
        assert(opt.split_test_2 is not None)
        eval_dset_2 = SelfCriticalDataset(opt.split_test_2, opt.hint_type, dictionary, opt, 
                                           discard_items_without_hints = opt.discard_items_for_test)
        eval_loader_2 = DataLoader(eval_dset_2, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    else:
        eval_loader_2 = None

    # update opts
    opt.full_v_dim = eval_dset.full_v_dim
    opt.num_ans_candidates = eval_dset.num_ans_candidates
    opt.num_objects = eval_dset.num_objects
    
    
    ## Create model
    if opt.model_type == 'updn':
        from models.updn import UpDn
        model = UpDn(opt)
    elif opt.model_type == 'updn_ramen':
        from models.updn import UpDn_ramen
        model = UpDn_ramen(opt)
    elif opt.model_type == 'lang_only':
        # load language-only updn model
        from models.lang_only import LangOnly
        model = LangOnly(opt)
    elif opt.model_type == 'lxmert':
        from models.lxmert import lxmert
        model = lxmert(opt)
    else:
        raise ValueError("unsupported model type")
    
    model = model.cuda()
    if 'lxmert' not in opt.model_type:
        model.apply(weights_init_kn)

    model = nn.DataParallel(model).cuda()
    model.train()
    
    run(model,
        train_loader,
        eval_loader,
        eval_loader_2,
        opt)
