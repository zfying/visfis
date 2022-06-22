import argparse
from distutils.util import strtobool



def parse_opt(s=None):
    parser = argparse.ArgumentParser()
    
    # logging & saving
    parser.add_argument('--saved_model_prefix', type=str, default='')
    parser.add_argument('--print_every_batch', action='store_true')
    parser.add_argument('--save_every_epoch', action='store_true')
    
    ## dataset
    parser.add_argument('--discard_items_for_test', type=bool, default=False)
    parser.add_argument('--portion_of_training', type=float, default=1.0)
    
    parser.add_argument('--train_subset', type=str) 
    parser.add_argument('--val_subset', type=str) 
    
    parser.add_argument('--use_two_testsets', action='store_true')
    parser.add_argument('--split_test_2', type=str, help='second test split')
    
    # spatial feats
    parser.add_argument('--spatial_type', type=str, default='none', choices=['none', 'simple', 'linear', 'mesh'])
    parser.add_argument('--spatial_length', default=0, type=int)
    
    # oracle exp
    parser.add_argument('--oracle_type', type=str, default="none", choices=['none', 'simple', 'wordvec'])
    parser.add_argument('--oracle_embed_size', type=int, default=512)
    parser.add_argument('--oracle_threshold', type=float, default=0.85)
    
    parser.add_argument('--use_input_mask', action='store_true') 
    parser.add_argument('--mask_type', type=str, default='impt', choices=['impt', 'non_impt']) 
    
    # model
    parser.add_argument('--model_type', type=str, default='updn_ramen', choices=['updn', 'rn', 'ban', 'updn_ramen', 'lang_only', 'updn_ramen_new', 'lxmert'])
    parser.add_argument('--num_objects', type=int)
    parser.add_argument('--lxmert_hid_dim', type=int, default=768)
    
    ## Feature Importance
    parser.add_argument('--model_importance', type=str, default='none', choices=['gradcam', 'expected_gradient', 'LOO', 'KOI', 'SHAP', 'avg_effect', 'updn_att', 'none'])
    parser.add_argument('--gradcam_type', type=str, default='sum', choices=['sum', 'l1', 'l2']) # for expected gradient / gradcam
    parser.add_argument('--num_sample_omission', type=int) # for all omission based methods
    parser.add_argument('--num_sample_eg', type=int) # for expected gradient
    parser.add_argument('--eval_FI', action='store_true') # model.eval for computing FI
    parser.add_argument('--OBJ11', action='store_true') # ugly hack to make visfis work
    parser.add_argument('--FI_predicted_class', dest='FI_predicted_class', 
                    type=lambda x: bool(strtobool(x))) # FI on predicted class vs. gt class; required
    parser.add_argument('--vqa_loss_type', type=str, default="softmax", choices=['softmax', 'sigmoid'])
    
    parser.add_argument('--impt_threshold', type=float)
    
    # control for random for VisFIS
    parser.add_argument('--random_suff', action='store_true') 
    parser.add_argument('--random_unc', action='store_true') 
    parser.add_argument('--random_inv_FI', action='store_true') 
    parser.add_argument('--random_align', action='store_true') 
    
    ## additional objectives
    # hint
    parser.add_argument('--use_relaxed_hint_loss', action='store_true')
    parser.add_argument('--use_hint_loss', action='store_true')
    parser.add_argument('--hint_loss_weight', type=float)
    # invariance-FI=0
    parser.add_argument('--use_zero_loss', action='store_true')
    parser.add_argument('--zero_loss_weight', type=float)
    parser.add_argument('--normalize_FI', type=bool, default=True) # only for zero_loss
    # direct alignment
    parser.add_argument('--use_direct_alignment', action='store_true') ## also use use align_loss_type
    parser.add_argument('--alignment_loss_weight', type=float)
    # data augment methods
    parser.add_argument('--aug_type', type=str, default='none', choices=['none','suff-human','suff-random','invariance', 'uncertainty-NonImptOnly', 'suff-uncertainty', 'suff-uncertainty-Chang2021', 'uncertainty-uniform', "saliency-guided"])
    parser.add_argument('--align_loss_type', type=str, choices=['l1', 'l2', 'kl', 'cossim'])
    parser.add_argument('--aug_loss_weight', type=float)  # for suff only
    
    parser.add_argument('--replace_func', type=str, default='negative_ones', choices=['0s', 'negative_ones', 'random_sample', 'gaussian-singla', 'gaussian-ours', 'shuffle'])
    
    # train / val
    parser.add_argument('--lr_type', type=str, default='none', choices=['none','ramen','neg_analysis'],
                        help='decide what lr scheduler to use') 
    parser.add_argument('--pretrain_finetune', action='store_true') 
    parser.add_argument('--pretrain_epochs', type=int, default=50) 
    parser.add_argument('--finetune_lr', type=float, default=2e-5)  # 2e-5 for HINT, 5e-5 for SCR
    
    # calc metrics
    parser.add_argument('--calc_dp_level_metrics', action='store_true') 
    parser.add_argument('--percentages', type=list, default=[0.1,0.25,0.5]) 
    parser.add_argument('--RRR_only', action='store_true')  # for metrics
    parser.add_argument('--ACC_only', action='store_true')  # for metrics
    
    # Data input settings
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default='vqacp2', choices=['xaicp', 'gqacp', 'hatcp'])
    parser.add_argument('--do_not_discard_items_without_hints', action='store_true')
    parser.add_argument('--split', type=str, help='training split')
    parser.add_argument('--hint_type', type=str)
    parser.add_argument('--split_test', type=str, help='test split')

    # parser.add_argument('--rnn_size', type=int, default=1280,
    #                     help='size of the rnn in number of hidden nodes in question gru')
    parser.add_argument('--num_hid', type=int, default=1280,
                        help='size of the rnn in number of hidden nodes in question gru')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GCN layers')
    parser.add_argument('--rnn_type', type=str, default='gru',
                        help='rnn, gru, or lstm')
    parser.add_argument('--v_dim', type=int, default=2048,
                        help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--logit_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--activation', type=str, default='ReLU',
                        help='number of layers in the RNN')
    parser.add_argument('--norm', type=str, default='weight',
                        help='number of layers in the RNN')
    parser.add_argument('--initializer', type=str, default='kaiming_normal',
                        help='number of layers in the RNN')
    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=40,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=384,
                        help='minibatch size')
    parser.add_argument('--grad_clip', type=int, default=0.25,
                        help='clip gradients at this value') # neg analysis 0.25; ramen 50; lxmert 5
    parser.add_argument('--dropC', type=float, default=0.5,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropG', type=float, default=0.2,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropL', type=float, default=0.1,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropW', type=float, default=0.4,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='strength of dropout in the Language Model RNN')

    # Optimization: for the Language Model

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='what update to use? rmsprop|sgd|sgdmom|adagrad|Adam')
    # parser.add_argument('--learning_rate', type=float, default=0.001,
    #                     help='learning rate')
    parser.add_argument('--learning_rate', type=float, help='learning rate')

    parser.add_argument('--optim_alpha', type=float, default=0.9,
                        help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                        help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay')
    parser.add_argument('--seed', type=int, default=1000,
                        help='seed')
    parser.add_argument('--ntokens', type=int, default=777,
                        help='ntokens')
    parser.add_argument('--load_checkpoint_path')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='directory to store checkpointed models')


    # Other params that probably don't need to be changed
    parser.add_argument('--load_model_states', type=str, default=0,
                        help='which model to load')

    parser.add_argument('--evaluate_every', type=int, default=300,
                        help='which model to load')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--predict_checkpoint', type=str)
    parser.add_argument('--change_scores_every_epoch', action='store_true')
    parser.add_argument('--test_has_regularization_split', action='store_true')
    parser.add_argument('--apply_answer_weight', action='store_true')
    parser.add_argument('--ignore_counting_questions', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[5, 10])
    parser.add_argument('--lr_gamma', type=float, default=1)

    # SCR loss parameters
    parser.add_argument('--use_scr_loss', action='store_true')

    parser.add_argument('--num_sub', type=int, default=5,
                        help='size of the proposal object set')

    parser.add_argument('--bucket', type=int, default=4,
                        help='bucket of predicted answers')

    parser.add_argument('--scr_hint_loss_weight', type=float, default=0,
                        help='Influence strength loss weights')

    parser.add_argument('--scr_compare_loss_weight', type=float, default=0,
                        help='self-critical loss weights')

    parser.add_argument('--reg_loss_weight', type=float, default=0.0,
                        help='regularization loss weights, set to zero in our paper ')

    # Parameters for the main VQA loss
    parser.add_argument('--vqa_loss_weight', type=float, default=1)

    
    if s is not None:
        args = parser.parse_args("")
    else:
        args = parser.parse_args()

    return args