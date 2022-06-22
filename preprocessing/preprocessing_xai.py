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

import utils

def convert_from_clevr_to_vqa_format(data_root, dataset, split):
    print(f"Converting {split} questions to VQA2 format...")
    clevr_file = json.load(open(os.path.join(data_root, 'questions', f'{dataset}_{split}_questions.json')))
    qns = clevr_file['questions']
    vqa_qns = []
    vqa_annotations = []
    for qn in qns:
        vqa_qn = {
            'question': qn['question'],
            'question_id': qn['question_index'],
            'image_id': qn['image_index']
        }
        vqa_ann = {
            'image_id': qn['image_index'],
            'question_id': qn['question_index']
        }

        # if 'test' not in split: 
        # qtype = qn['program'][-1]['function']
        qtype = qn['program'][-1]['type'] # 'type' instead of 'function'
        answer = qn['answer']
        answers = [{'answer': qn['answer']}]
        vqa_ann['answer_type'] = qtype
        vqa_ann['question_type'] = qtype
        vqa_ann['multiple_choice_answer'] = answer
        vqa_ann['answers'] = answers
        
        vqa_qns.append(vqa_qn)
        vqa_annotations.append(vqa_ann)

    vqa_qns = {'questions': vqa_qns}
    vqa_annotations = {'annotations': vqa_annotations}

    with open(os.path.join(data_root, 'questions', f'{split}_questions.json'), 'w') as f:
        json.dump(vqa_qns, f)
    # if 'test' not in split:
    with open(os.path.join(data_root, 'questions', f'{split}_annotations.json'), 'w') as f:
        json.dump(vqa_annotations, f)

def convert_xai_ans_details(data_root, split):
    answer_file = os.path.join(data_root, 'questions', f'{split}_annotations.json')
    answers = json.load(open(answer_file))
    
    for a in answers['annotations']:
        assert type(a["answers"][0]["answer"]) == type(a['multiple_choice_answer'])
        ans_type = type(a["answers"][0]["answer"])
        if ans_type != str:
            if ans_type == int:
                a["answers"][0]["answer"] = str(a["answers"][0]["answer"])
                a['multiple_choice_answer'] = str(a['multiple_choice_answer'])
            elif ans_type == bool:
                if a["answers"][0]["answer"] == False:
                    a["answers"][0]["answer"] = "no"
                    a['multiple_choice_answer'] = "no"
                elif a["answers"][0]["answer"] == True:
                    a["answers"][0]["answer"] = "yes"
                    a['multiple_choice_answer'] = "yes"
                else:
                    raise ValueError("unknown bool")
            else: 
                raise ValueError("unknown ans type")
    with open(answer_file, 'w') as f:
        json.dump(answers, f)
    return answers


if __name__ == '__main__':
    # set arguments
    data_root = str(sys.argv[1])
    dataset = 'xaicp'
    
    # 1: convert qns to vqa format
    convert_from_clevr_to_vqa_format(data_root, dataset, 'train')
    convert_from_clevr_to_vqa_format(data_root, dataset, 'dev')
    convert_from_clevr_to_vqa_format(data_root, dataset, 'test-id')
    convert_from_clevr_to_vqa_format(data_root, dataset, 'test-ood')
    
    # 2: change ans data type
    train_ans = convert_xai_ans_details(data_root, 'train')
    dev_ans = convert_xai_ans_details(data_root, 'dev')
    test_id_ans = convert_xai_ans_details(data_root, 'test-id')
    test_ood_ans = convert_xai_ans_details(data_root, 'test-ood')
    
    # 3: process ids_map
    ids_map_path = os.path.join(data_root,'xaicp_ids_map.json')
    ids_map = json.load(open(ids_map_path))

    new_ids_map = {}
    new_image_ix_to_id = {}
    new_image_id_to_ix = {}

    for k, v in ids_map['image_ix_to_id'].items():
        new_image_ix_to_id[int(k)] = v
    for k, v in ids_map['image_id_to_ix'].items():
        new_image_id_to_ix[int(k)] = v

    new_ids_map['image_ix_to_id'] = new_image_ix_to_id
    new_ids_map['image_id_to_ix'] = new_image_id_to_ix

    image_id2ix = new_ids_map['image_id_to_ix']

    _path = os.path.join(data_root, 'xaicp_imgid2img.pkl')
    pickle.dump(image_id2ix, open(_path, 'wb'))
    
    # 4: create dictionary
    d = create_dictionary(dataset, data_root)
    d.dump_to_file(f'{data_root}/dictionary.pkl')

    d = Dictionary.load_from_file(f'{data_root}/dictionary.pkl')
    emb_dim = 300
    glove_file = f'{data_root}/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(f'{data_root}/glove6b_init_%dd.npy' % emb_dim, weights)

    # 5: compute target
    # read annotations
    features_path = os.path.join(data_root, 'processed')
    train_answer_file = os.path.join(data_root, 'questions', 'train_annotations.json')
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
    # read questions
    train_question_file = os.path.join(data_root, 'questions', 'train_questions.json')
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

    answers = train_answers + dev_answers + test_id_answers + test_ood_answers
    occurrence = filter_answers(answers, min_occurence=0)
    ans2label = create_ans2label(occurrence, 'trainval', cache_root=features_path)

    # compute target
    # for clevr fixed_score = 1
    fixed_score=1.0
    compute_target(train_answers, ans2label, 'train', cache_root=features_path, fixed_score=fixed_score)
    compute_target(dev_answers, ans2label, 'dev', cache_root=features_path, fixed_score=fixed_score)
    compute_target(test_id_answers, ans2label, 'test-id', cache_root=features_path, fixed_score=fixed_score)
    compute_target(test_ood_answers, ans2label, 'test-ood', cache_root=features_path, fixed_score=fixed_score)

    # 6: create ans cossim = 1
    ans_cossim = np.ones((28,28))
    _path = os.path.join(data_root, 'ans_cossim.pkl')
    pickle.dump(ans_cossim, open(_path,'wb'))