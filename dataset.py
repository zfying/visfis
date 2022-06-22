import os
import json
import pickle as cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle
from torch.utils.data.sampler import Sampler

import pdb


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-', ' ').replace('.', '').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                if '-' in w:
                    print(w)
                tokens.append(self.add_word(w))
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(len(self.word2idx))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)




class SelfCriticalDataset(Dataset):

    def __init__(self, split,
                 hint_type,
                 dictionary,
                 opt,
                 discard_items_without_hints=False):
        super(SelfCriticalDataset, self).__init__()
        self.split = split
        self.hint_type = hint_type
        self.dictionary = dictionary  # questions' dictionary
        self.opt = opt
        self.data_dir = opt.data_dir
        self.discard_items_without_hints = discard_items_without_hints
        if hint_type is None and self.discard_items_without_hints:
            raise Exception("Cannot discard items without hints because hint_type is not specified")
        
        ## load data 
        # load hint 
        if self.hint_type is not None:
            if self.opt.dataset in ['xaicp', 'gqacp', 'hatcp']:
                hint_fname = f'hints/{self.opt.dataset}_{self.hint_type}.pkl'
            else:
                hint_fname = f'hints/{self.split}_{self.hint_type}.pkl'
            self.hint = cPickle.load(open(os.path.join(self.data_dir, hint_fname), 'rb'))
            print(f"loaded hints from {hint_fname}")
            
        # support controlled hint exp
        if self.opt.random_suff or self.opt.random_unc or self.opt.random_inv_FI or self.opt.random_align:
            hint_fname = f'hints/{self.opt.dataset}_hints_random.pkl'
            self.hint_random = cPickle.load(open(os.path.join(self.data_dir, hint_fname), 'rb'))
            print("loaded random hint")
        
        # get questions
        self.questions = self.get_questions() 
        # get annotations
        self.annotations = self.get_annotations()
        print(f"loaded questions/annotations")
        # get qid_to_target
        self.qid_to_target = self.get_qid_to_target()
        print('loaded qid_to_targets')
        # get ans2label / label2ans
        ans2label_path = os.path.join(self.data_dir, 'processed', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(self.data_dir, 'processed', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        print('loaded ans2label and label2ans')
        self.num_ans_candidates = len(self.ans2label)
        print(f'num_ans_candidates is {self.num_ans_candidates}')
        
        # load features and spatials from hdf5 file
        self.image_id2ix = {}
        self.hf = {}
        self.features = {}
        self.spatials = {}
        self.get_features() 
        
        # calc v feature length
        if self.opt.model_type == 'lxmert':
            assert self.opt.spatial_type == 'simple'
        self.full_v_dim = self.len_pure_visual + \
                        utils.get_spatial_length(self.opt.spatial_type, self.opt.spatial_length) + \
                        utils.get_oracle_length(self.opt.oracle_type, self.opt.oracle_embed_size)
        # init nn.Embedding
        if self.opt.oracle_type == 'wordvec':
            self.oracle_embed = nn.Embedding(2, self.opt.oracle_embed_size, padding_idx=-1)  # Binary
        else:
            self.oracle_embed = None
            
            
        self.init_vqx()        
        self.tokenize()
        self.tensorize()
        
        print(f"split {self.split} len {self.datalen}")
        # clean up
        del self.questions, self.annotations
        del self.dictionary, self.ans2label, self.label2ans, self.qid_to_target, self.hint
    
    
    def get_qid_to_target(self):
        if self.opt.dataset in ["vqacp2", "hatcp"]:
            train_target = cPickle.load(open(os.path.join(self.data_dir, 'processed', f'train_target.pkl'), 'rb'))
            val_target = cPickle.load(open(os.path.join(self.data_dir, 'processed', f'val_target.pkl'), 'rb'))
            target = train_target + val_target
        else: # for gqa and xai, only read current split
            target = cPickle.load(open(os.path.join(self.data_dir, 'processed', f'{self.split}_target.pkl'), 'rb'))
        qid_to_target = {}
        for t in target:
            question_id = t['question_id']
            assert question_id not in qid_to_target
            qid_to_target[question_id] = t
        return qid_to_target

    def get_questions(self):
        if self.opt.dataset == 'vqacp2':
            if self.split == "train":
                f = os.path.join(self.data_dir, f'vqacp_v2_train_questions.json')
            else:
                f = os.path.join(self.data_dir, f'vqacp_v2_test_questions.json')
            
            return json.load(open(f))
        elif self.opt.dataset == 'vqa2':
            year = '2015' if self.split == 'test' else '2014'
            f = os.path.join(self.data_dir, f'v2_OpenEnded_mscoco_{self.split}{year}_questions.json')
            return json.load(open(f))['questions']
        elif self.opt.dataset in ['xaicp','gqacp', 'hatcp']:
            # support subset -> only for GQA-CP
            if self.split == 'train' and self.opt.train_subset is not None:
                f = os.path.join(self.data_dir, f'questions/{self.split}-{self.opt.train_subset}_questions.json')
            elif self.split != 'train' and self.opt.val_subset is not None:
                f = os.path.join(self.data_dir, f'questions/{self.split}-{self.opt.val_subset}_questions.json')
            else:
                f = os.path.join(self.data_dir, f'questions/{self.split}_questions.json')
            questions = json.load(open(f))['questions']
            
            # suport varying training size -> training size ablation
            if self.split == 'train' and self.opt.portion_of_training != 1.0:                          
                # randomized version
                import random
                ori_len = len(questions) 
                num_removal = int(len(questions) * (1-self.opt.portion_of_training))
                random.shuffle(questions)
                for i in range(num_removal):
                    questions.pop()
                print(f"new question size: {len(questions)}， portion is {len(questions) / ori_len}")
            return questions
            
        else:
            raise ValueError(f'cannot get questions for {self.opt.dataset}')
        
    def get_annotations(self):
        if self.opt.dataset == 'vqacp2':
            if self.split == "train":
                f = os.path.join(self.data_dir, f'vqacp_v2_{self.split}_annotations.json')
            else:
                f = os.path.join(self.data_dir, f'vqacp_v2_test_annotations.json')
            return json.load(open(f))
        elif self.opt.dataset == 'vqa2':
            year = '2015' if self.split == 'test' else '2014'
            f = os.path.join(self.data_dir, f'v2_mscoco_{self.split}{year}_annotations.json')
            return json.load(open(f))['annotations']
        elif self.opt.dataset in ['xaicp','gqacp', 'hatcp']:
            # support subset -> for GQA-CP only
            if self.split == 'train' and self.opt.train_subset is not None:
                f = os.path.join(self.data_dir, 'questions', 
                                 f'{self.split}-{self.opt.train_subset}_annotations.json')
            elif self.split != 'train' and self.opt.val_subset is not None:
                f = os.path.join(self.data_dir, 'questions', 
                                 f'{self.split}-{self.opt.val_subset}_annotations.json')
            else:
                f = os.path.join(self.data_dir, 'questions', f'{self.split}_annotations.json')
            return json.load(open(f))['annotations']
        else:
            raise ValueError(f'cannot get annotations for {self.opt.dataset}')
    
    def get_features(self):
        if self.opt.dataset in ['xaicp', 'gqacp']: # shared train/val features -> xai
            print(f'loading hdf5 for combined train/val')
            # read image_id2ix
            _path = os.path.join(self.data_dir, f'{self.opt.dataset}_imgid2img.pkl')
            self.image_id2ix = cPickle.load(open(_path, 'rb'))
            # read hdf5
            h5_path = os.path.join(self.data_dir, f'{self.opt.dataset}.hdf5')
            self.hf = h5py.File(h5_path, 'r')
            self.features = self.hf.get('image_features')
            self.spatials = self.hf.get('spatial_features')
            # get para
            self.len_pure_visual = self.features.shape[2]
            self.num_objects = self.features.shape[1]
            
            
        elif self.opt.dataset =="vqacp2" or self.opt.dataset =="hatcp": # cp -> need to load both train and val
            print(f'loading hdf5 for {self.split} split')
            # read image_id2ix
            self.image_id2ix = {}
            self.image_id2ix["train"] = cPickle.load(open(os.path.join(self.data_dir,
                                                          'train36_imgid2img.pkl'), 'rb'))
            self.image_id2ix["val"] = cPickle.load(open(os.path.join(self.data_dir,
                                                          'val36_imgid2img.pkl'), 'rb'))
            # read hdf5
            self.hf = {}
            self.features = {}
            self.spatials = {}
            h5_path = os.path.join(self.data_dir, 'train36.hdf5')
            self.hf["train"] = h5py.File(h5_path, 'r')
            self.features["train"] = self.hf["train"].get('image_features')
            self.spatials["train"] = self.hf["train"].get('spatial_features')

            h5_path = os.path.join(self.data_dir, 'val36.hdf5')
            self.hf["val"] = h5py.File(h5_path, 'r')
            self.features["val"] = self.hf["val"].get('image_features')
            self.spatials["val"] = self.hf["val"].get('spatial_features')
            # get para
            self.len_pure_visual = self.features["train"].shape[2]
            self.num_objects = self.features["train"].shape[1]
            
        else:
            raise ValueError("unsupported dataset in get_features()")

    def init_vqx(self):
        
        print("initializing vqx...")
        count = 0
        self.entries = {}
        
        # iter through questions
        for index, question in tqdm(enumerate(self.questions)):
            image_id = question['image_id']
            question_id = question['question_id']
            answer_ori = self.annotations[index]['multiple_choice_answer']

            if self.discard_items_without_hints and question_id not in self.hint.keys():
                # ignore discarded item
                continue
            elif self.hint_type is not None and question_id in self.hint.keys():
                hint = self.hint[question_id]
                hint_flag = 1
            else:
                hint = np.zeros((self.num_objects))
                hint_flag = 0
            
            # add hint as oracle to v_feature
            hint_scores = torch.from_numpy(hint)
            hint_scores = hint_scores.float().unsqueeze(1)
            
            # support controlled hint exp
            if self.opt.random_suff or self.opt.random_unc or self.opt.random_inv_FI or self.opt.random_align:
                hint_random_scores = self.hint_random[question_id]
                hint_random_scores = torch.from_numpy(hint_random_scores)
                hint_random_scores = hint_random_scores.float().unsqueeze(1)
                hint_scores = (hint_scores, hint_random_scores)
            
            if self.opt.dataset in ["vqacp2", "hatcp"]: # two splits
                if image_id in self.image_id2ix['train']:
                    cur_split = 'train'
                else:
                    cur_split = 'val'

            new_entry = {'image': self.image_id2ix[cur_split][image_id] if self.opt.dataset in ["vqacp2", "hatcp"] else self.image_id2ix[image_id],
                         'image_id': image_id,
                         'question_id': question_id,
                         'question': question['question'],
                         'answer': self.qid_to_target[question_id],
                         'hint': hint_scores,
                         'hint_flag': hint_flag,
                        'answer_ori': answer_ori}
            self.entries[count] = new_entry
            count += 1
        self.datalen = count
        print(f"split {self.split} init_vqx count {count}")
        return count

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for e_id in range(len(self.entries)):
            entry = self.entries[e_id]
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for e_id in range(len(self.entries)):
            entry = self.entries[e_id]
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if labels is None:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
            elif len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        imgid = entry['image_id']
        qid = entry['question_id']        
        
        hint_score = entry['hint']
        hint_flag = entry['hint_flag']
        
        # get features/spatials
        if self.opt.dataset in ["vqacp2", "hatcp"]: # two splits
            if imgid in self.image_id2ix['train']:
                cur_split = 'train'
            else:
                cur_split = 'val'
            image_ix = self.image_id2ix[cur_split][imgid]
            features = torch.from_numpy(np.array(self.features[cur_split][image_ix]))
            spatials = torch.from_numpy(np.array(self.spatials[cur_split][image_ix]))
        else: # one split
            image_ix = self.image_id2ix[imgid]
            features = torch.from_numpy(np.array(self.features[image_ix]))
            spatials = torch.from_numpy(np.array(self.spatials[image_ix]))

        # add spatials to v_feature
        curr_v_feature = utils.adding_spatials(self.opt, features, spatials,
                                                 self.opt.spatial_type,
                                                 self.opt.spatial_length,
                                                 self.num_objects)
        # add oracle
        curr_v_feature = utils.adding_oracles(curr_v_feature, hint_score,
                                           self.opt.oracle_type,
                                           self.oracle_embed)
        
        question_ori = entry['question']
        answer_ori = entry['answer_ori']
        
        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        
        
        if labels is not None:
            target.scatter_(0, labels, scores)

        return curr_v_feature, question, target, hint_score, qid, imgid, hint_flag, question_ori, answer_ori
    
    def __len__(self):
        return self.datalen

