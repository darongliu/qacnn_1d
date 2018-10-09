import os
import numpy as np

from tqdm import tqdm
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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
        feat = [bag_cq_sim[0], bag_co_sim[0,i], bag_qo_sim[0,i], tfidf_cq_sim[0], tfidf_co_sim[0,i], tfidf_qo_sim[0,i]]
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
        feat = [cq_sim[0], co_sim[0,i], qo_sim[0,i]]
        all_feat.append(feat)

    return np.array(all_feat)

def wer(ref, hyp, mode = 'word', without_len = True):
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

    return d[-1,-1]

def distane_feature(context, question, options):
    choice_num = len(options)

    word_edit = []
    char_edit = []

    char_lcs = []

    for op in options:

        # word based
        w_edit = wer(op, context, mode = 'word')
        word_edit.append(w_edit)

        # TODO: word based lcs?

        # char based
        context_c = context.replace(' ', '')
        op_c = op.replace(' ', '')
        char_edit.append(wer(op_c, context_c, mode = 'char'))

        match = SequenceMatcher(None, context_c, op_c).find_longest_match(0, len(context_c),0,len(op_C))
        lcs = len(context_c[match.a:match.a+match.size])
        char_lcs.append(lcs)

    ret = word_edit + char_edit + char_lcs
    ret = np.array(ret)
    return ret

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
        c_f_bopo = bagw_tfidf_sim(c_cv_bopo, c_tt_bopo, context_bopo, question, options)

        w2v = word_embedding(w2v_model, context, question, options)

        d_f = distane_feature(context, question, options)
        d_f_bopo = distane_feature(context_bopo, question_bopo, options_bopo)

        is_neg = 0

        for neg_word in ['不', '沒有', '否', '不是', '非']:
            if neg_word in question.split():
                is_neg = 1
                break
        is_neg = [is_neg]

        f = np.concatenate((w_f, c_f, w2v, d_f, is_neg, w_f_bopo, c_f_bopo, d_f_bopo))
        #f = np.concatenate((w_f, c_f, w2v, d_f, is_neg))

        X_train.append(f)

    X_train = np.stack(X_train)

    return X_train

if __name__ == '__main__':
    import json
    import _pickle as cPickle

    c_vocab = cPickle.load(open('./preprocess/wiki_c_vocab.pkl', 'rb'))
    w_vocab = cPickle.load(open('./preprocess/wiki_w_vocab.pkl', 'rb'))

    data = json.load(open('../data/elementary/processed/all.json.seg.scale', 'r'))

    X_train, Y_train = get_feature(data)

    print(X_train.shape)
