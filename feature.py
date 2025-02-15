import os
import numpy as np
import jieba
from fuzzysearch import find_near_matches

from tqdm import tqdm
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from utils.utils import text_norm_before_cut, text_norm_after_cut

DEBUG = False

def bagw_tfidf_sim(cv, tt, context, question, options):
    choice_num = len(options)

    bag_c = cv.transform([context])
    bag_q = cv.transform([question])
    bag_o = cv.transform(options)

    tfidf_c = tt.transform(bag_c)
    tfidf_q = tt.transform(bag_q)
    tfidf_o = tt.transform(bag_o)

    bag_cq_sim = cosine_similarity(bag_c, bag_q)
    bag_co_sim = cosine_similarity(bag_c, bag_o)
    bag_qo_sim = cosine_similarity(bag_q, bag_o)
    tfidf_cq_sim = cosine_similarity(tfidf_c, tfidf_q)
    tfidf_co_sim = cosine_similarity(tfidf_c, tfidf_o)
    tfidf_qo_sim = cosine_similarity(tfidf_q, tfidf_o)

    if DEBUG:
        print(bag_cq_sim)
        print(bag_co_sim)
        print(bag_qo_sim)
        print(tfidf_cq_sim)
        print(tfidf_co_sim)
        print(tfidf_qo_sim)

    all_feat = []
    for i in range(choice_num):
        #feat = [bag_cq_sim[0,0], bag_co_sim[0,i], bag_qo_sim[0,i], tfidf_cq_sim[0,0], tfidf_co_sim[0,i], tfidf_qo_sim[0,i]]
        feat = [bag_co_sim[0,i], bag_qo_sim[0,i], tfidf_co_sim[0,i], tfidf_qo_sim[0,i]]
        all_feat.append(feat)

    return np.array(all_feat)

def word2vec(w2v_model, s):
    '''
    s: 'word word word'
    '''
    dim = w2v_model.get_dimension()
    base = np.zeros(dim)
    s_list = s.split()
    if len(s_list) == 0:
        return base

    for w in s_list:
        base += w2v_model.get_word_vector(w)
    base = base / len(s_list)
    return base

def word_embedding(w2v_model, context, question, options):
    #vec_c = w2v_model.get_sentence_vector(context)
    #vec_q = w2v_model.get_sentence_vector(question)
    #vec_os = [ w2v_model.get_sentence_vector(op) for op in options ]
    choice_num = len(options)

    vec_c = word2vec(w2v_model, context)
    vec_q = word2vec(w2v_model, question)
    vec_os = [ word2vec(w2v_model, op) for op in options ]
    vec_c = np.array([vec_c])
    vec_q = np.array([vec_q])
    vec_os = np.array(vec_os)

    cq_sim = cosine_similarity(vec_c, vec_q)
    co_sim = cosine_similarity(vec_c, vec_os)
    qo_sim = cosine_similarity(vec_q, vec_os)

    all_feat = []
    for i in range(choice_num):
        #feat = [cq_sim[0,0], co_sim[0,i], qo_sim[0,i]]
        feat = [co_sim[0,i], qo_sim[0,i]]
        all_feat.append(feat)

    return np.array(all_feat)

def wer(ref, hyp, mode = 'word', without_len = False):
    if ref == '':
        r = '慘'

    if mode == 'word':
        s1 = ref.split()
        s2 = hyp.split()
    else:
        s1 = list(ref)
        s2 = list(hyp)

    d = np.zeros([len(s1)+1,len(s2)+1])
    d[:,0] = np.arange(len(s1)+1)
    d[0,:] = np.arange(len(s2)+1)

    for j in range(1,len(s2)+1):
        for i in range(1,len(s1)+1):
            if s1[i-1] == s2[j-1]:
                d[i,j] = d[i-1,j-1]
            else:
                d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+1)

    return d[-1,-1]-len(s1)

def distane_feature(context, question, options):
    choice_num = len(options)

    all_feat = []

    for op in options:
        feat = []

        # word based
        w_edit = wer(context, op, mode = 'word')
        feat.append(w_edit)

        # TODO: word based lcs?

        # char based
        context_c = context.replace(' ', '')
        op_c = op.replace(' ', '')
        feat.append(wer(context_c, op_c, mode = 'char'))

        match = SequenceMatcher(None, context_c, op_c).find_longest_match(0, len(context_c),0,len(op_c))
        lcs = len(context_c[match.a:match.a+match.size])
        feat.append(lcs)

        all_feat.append(feat)

    return np.array(all_feat)

def get_position(context_no_space, text_with_space, valid_length, max_error, margin_error):
    context = context_no_space
    text = text_with_space.split()

    accumulate_position = 0
    count = 0

    for word in text:
        if len(word) < valid_length:
            continue
        error_distance = len(word) - margin_error
        if error_distance < 0:
            error_distance = 0
        if error_distance > max_error:
            error_distance = max_error
        all_position = find_near_matches(word, context, max_l_dist=error_distance)

        for position in all_position:
            count += 1
            accumulate_position += (position[0]+position[1])/2

    if count == 0:
        return -1

    return accumulate_position/count

def get_position_feat(context_no_space, question, options, valid_length, max_error, margin_error):
    pos_q = get_position(context_no_space, question, valid_length, max_error, margin_error)
    all_feat = []
    for op in options:
        pos_op = get_position(context_no_space, op, valid_length, max_error, margin_error)
        if pos_q == -1:
            if pos_op == -1:
                all_feat.append([1.])
            else:
                all_feat.append([0.5])
        else:
            if pos_op == -1:
                all_feat.append([1.])
            else:
                all_feat.append([abs(pos_op-pos_q)/len(context_no_space)])
    return np.array(all_feat)

def get_feature(data, fasttext_model):
    '''
    data :[dict{'context_nostop':...  ,
                'question_nostop':... ,
                'options_nostop':[list of options],
                'context_bopo':... ,
                'question_bopo':...,
                'options_bopo':[list of bopo options] }]
    '''
    c_cv = CountVectorizer(analyzer = 'char')
    w_cv = CountVectorizer(analyzer = 'word')

    c_tt = TfidfTransformer()
    w_tt = TfidfTransformer()

    c_cv_bopo = CountVectorizer(analyzer = 'char')
    w_cv_bopo = CountVectorizer(analyzer = 'word')

    c_tt_bopo = TfidfTransformer()
    w_tt_bopo = TfidfTransformer()

    w2v_model = fasttext_model

    # get bag-of-word, tfidf
    all = []
    all_bopo = []
    for sample in tqdm(data):
        context = sample['context_nostop']
        question = sample['question_nostop']
        options = sample['options_nostop']

        all += [context, question]
        all += options

        context_bopo = sample['context_bopo']
        question_bopo = sample['question_bopo']
        options_bopo = sample['options_bopo']

        all_bopo += [context_bopo, question_bopo]
        all_bopo += options_bopo

    x = c_cv.fit_transform(all)
    c_tt.fit(x)
    x = w_cv.fit_transform(all)
    w_tt.fit(x)

    x = c_cv_bopo.fit_transform(all_bopo)
    c_tt_bopo.fit(x)
    x = w_cv_bopo.fit_transform(all_bopo)
    w_tt_bopo.fit(x)

    X_train = []

    for i, sample in enumerate(tqdm(data)):
        context = sample['context_nostop']
        question = sample['question_nostop']
        options = sample['options_nostop']

        context_bopo = sample['context_bopo']
        question_bopo = sample['question_bopo']
        options_bopo = sample['options_bopo']

        w_f = bagw_tfidf_sim(w_cv, w_tt, context, question, options)
        c_f = bagw_tfidf_sim(c_cv, c_tt, context, question, options)

        w_f_bopo = bagw_tfidf_sim(w_cv_bopo, w_tt_bopo, context_bopo, question, options)
        c_f_bopo = bagw_tfidf_sim(c_cv_bopo, c_tt_bopo, context_bopo, question_bopo, options_bopo)

        w2v = word_embedding(w2v_model, context, question, options)

        d_f = distane_feature(context, question, options)
        d_f_bopo = distane_feature(context_bopo, question_bopo, options_bopo)

        pos = get_position_feat(sample['context'].replace(' ',''), question, options, 0, 1, 2)
        pos_bopo = get_position_feat(sample['context_bopo_stop'].replace(' ',''), question_bopo, options_bopo, 0, 1, 2)

        option_num = len(options)
        is_neg = np.zeros([option_num,1])
        for neg_word in ['不', '沒有', '否', '不是', '非']:
            if neg_word in question.replace(' ',''):
                is_neg = np.ones([option_num,1])
                break

        f = np.concatenate((w_f, c_f, w2v, d_f, is_neg, w_f_bopo, c_f_bopo, d_f_bopo, pos, pos_bopo),-1)
        #f = np.concatenate((w_f, c_f, w2v, d_f, is_neg, pos, pos_bopo))

        X_train.append(f)

    X_train = np.stack(X_train)

    return X_train

def get_zhuyin_seq(sent, zhuyin_dict):
    '''
    zhuyin: {char:[possible_zhuyin_seq1, possible_zhuyin_seq2...]}
    '''
    zhuyin_seq = ""
    for char in sent:
        if char == ' ':
            zhuyin_seq += ' '
            continue
        if char in zhuyin_dict:
            zhuyin_combination = zhuyin_dict[char][0][:-1] # no tone
            zhuyin_seq += zhuyin_combination
    return zhuyin_seq

def read_zhuyin_dict(path):
    with open(path, 'r') as f:
        all_lines = f.read().splitlines()
    zhuyin_dict = {}
    for line in all_lines:
        line = line.strip()
        char = line.split(' ', 1)[0]
        combination = line.split(' ', 1)[1].split('/')
        zhuyin_dict[char] = combination
    return zhuyin_dict

def similar_bopo_replace(text):
    text = text.replace('ㄓ', 'ㄗ')
    text = text.replace('ㄔ', 'ㄘ')
    text = text.replace('ㄕ', 'ㄙ')
    text = text.replace('ㄣ', 'ㄥ')
    text = text.replace('ㄧ', 'ㄩ')
    return text

def aug_with_zhuyin(data, replace=False):
    zhuyin_dict = read_zhuyin_dict('./ZhuYin.map')

    all_new_data = []
    for sample in data:
        sample['context_bopo'] = get_zhuyin_seq(sample['context_nostop'], zhuyin_dict)
        sample['question_bopo'] = get_zhuyin_seq(sample['question_nostop'], zhuyin_dict)
        sample['options_bopo'] = [get_zhuyin_seq(sent, zhuyin_dict) for sent in sample['options_nostop']]

        sample['context_bopo_stop'] = get_zhuyin_seq(sample['context'], zhuyin_dict)
        sample['question_bopo_stop'] = get_zhuyin_seq(sample['question'], zhuyin_dict)
        sample['options_bopo_stop'] = [get_zhuyin_seq(sent, zhuyin_dict) for sent in sample['options']]

        if replace:
            sample['context_bopo'] = similar_bopo_replace(sample['context_bopo'])
            sample['question_bopo'] = similar_bopo_replace(sample['question_bopo'])
            sample['options_bopo'] = [similar_bopo_replace(text) for text in sample['options_bopo']]
            sample['context_bopo_stop'] = similar_bopo_replace(sample['context_bopo_stop'])
            sample['question_bopo_stop'] = similar_bopo_replace(sample['question_bopo_stop'])
            sample['options_bopo_stop'] = [similar_bopo_replace(text) for text in sample['options_bopo_stop']]
        all_new_data.append(sample)

    return all_new_data

def cut(text):
    text = text_norm_before_cut(text)
    word_list = list(jieba.cut(text))
    no_stop_word_list = text_norm_after_cut(word_list)
    return ' '.join(word_list), ' '.join(no_stop_word_list)

def modified_fomrat(origin_data):
    all_new_data = []
    for data in origin_data:
        new_data = {}
        cut_context = cut(data['context'])
        new_data['context'] = cut_context[0]
        new_data['context_nostop'] = cut_context[1]

        cut_question = cut(data['question'])
        new_data['question'] = cut_question[0]
        new_data['question_nostop'] = cut_question[1]

        all_option = []
        all_option_no_stop = []
        for option in data['options']:
            cut_text = cut(option)
            all_option.append(cut_text[0])
            all_option_no_stop.append(cut_text[1])
        new_data['options'] = all_option
        new_data['options_nostop'] = all_option_no_stop

        all_new_data.append(new_data)

    return all_new_data


if __name__ == '__main__':
    import json
    import fastText
    import sys
    data = json.load(open('./data/dev.json', 'r'))
    data = modified_fomrat(data)
    data = aug_with_zhuyin(data)

    model = fastText.FastText.load_model('./word2vec/model.bin')
    a = get_feature(data, model)
    print(a[0])
    print(a.shape)


