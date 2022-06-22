import os
import sys

import json
import pickle 
from six.moves import cPickle

import re
import numpy as np

from tqdm import tqdm

from dataset import Dictionary
from preprocessing.create_dictionary import create_dictionary, create_glove_embedding_init
from preprocessing.compute_softscore import compute_target, filter_answers, create_ans2label

if __name__ == '__main__':
    # set arguments
    data_root = str(sys.argv[1])
    dataset = 'GQACP'
    
    # 1: create dictionary
    d = create_dictionary(dataset, data_root)
    d.dump_to_file(f'{data_root}/dictionary.pkl')
    d = Dictionary.load_from_file(f'{data_root}/dictionary.pkl')
    
    emb_dim = 300
    glove_file = f'{data_root}/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(f'{data_root}/glove6b_init_%dd.npy' % emb_dim, weights)
    
    # 2: compute target 
    features_path = os.path.join(data_root, 'processed')
    train_answer_file = os.path.join(data_root, 'questions', 'train_annotations.json')
    # read annotations
    train_answers = json.load(open(train_answer_file))
    if 'annotations' in train_answers:
        train_answers = train_answers['annotations']
    _path = os.path.join(data_root, 'questions', 'dev_annotations.json')
    dev_answers = json.load(open(_path))
    if 'annotations' in dev_answers:
        dev_answers = dev_answers['annotations']
    _path = os.path.join(data_root, 'questions', 'test-id_annotations.json')
    test_id_answers = json.load(open(_path))
    if 'annotations' in test_id_answers:
        test_id_answers = test_id_answers['annotations']
    _path = os.path.join(data_root, 'questions', 'test-ood_annotations.json')
    test_ood_answers = json.load(open(_path))
    if 'annotations' in test_ood_answers:
        test_ood_answers = test_ood_answers['annotations']
    answers = train_answers + dev_answers + test_id_answers + test_ood_answers
    # read questions
    train_question_file = os.path.join(data_root, 'questions', 'train-100k_questions.json')
    train_questions = json.load(open(train_question_file))
    if 'questions' in train_questions:
        train_questions = train_questions['questions']
    _path = os.path.join(data_root, 'questions', 'dev_questions.json')
    dev_questions = json.load(open(_path))
    if 'questions' in dev_questions:
        dev_questions = dev_questions['questions']
    _path = os.path.join(data_root, 'questions', 'test-id_questions.json')
    test_id_questions = json.load(open(_path))
    if 'questions' in test_id_questions:
        test_id_questions = test_id_questions['questions']
    _path = os.path.join(data_root, 'questions', 'test-ood_questions.json')
    test_ood_questions = json.load(open(_path))
    if 'questions' in test_ood_questions:
        test_ood_questions = test_ood_questions['questions']
    # compute target 
    occurrence = filter_answers(answers, min_occurence=0)
    ans2label = create_ans2label(occurrence, 'trainval', cache_root=features_path)
    # fixed_score = 1
    fixed_score=1.0
    splits_list = ['train', 'dev', 'test-id', 'test-ood',
                  'train-100k', 'dev-100k', 'test-id-100k', 'test-ood-100k']
    for split in splits_list:
        _path = os.path.join(data_root, 'questions', split+'_annotations.json')
        _answers = json.load(open(_path))['annotations']
        compute_target(_answers, ans2label, split, cache_root=features_path, fixed_score=fixed_score)
    
    # 3: create ans-cossim
    num_answers = 1842
    ans_cossim = np.ones((num_answers, num_answers))
    _path = os.path.join(data_root, 'ans_cossim.pkl')
    pickle.dump(ans_cossim, open(_path,'wb'))
    