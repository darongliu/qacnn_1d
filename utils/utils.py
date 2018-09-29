import torch
import os
import sys
csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)
from number2chinese import replace_numbers

def put_to_cuda(tensor_list):
    return [x.cuda() for x in tensor_list]

def print_and_logging(f, log):
    print(log)
    f.write(log)
    f.write('\n')

def text_norm_before_cut(text):
    '''
    input is a string
    '''
    def remove_words(text, file_name=None, remove_wrd_list=None):
        if file_name is not None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            all_remove = open(os.path.join(dir_path, file_name), 'r').read().splitlines()
            all_remove = [word.strip() for word in all_remove]
            all_remove = sorted(all_remove, key=lambda x:len(x))
            for word in all_remove:
                text = text.replace(word, "")
        if remove_wrd_list is not None:
            for word in remove_wrd_list:
                text = text.replace(word, "")
        return text
    # change number to chinese
    text = replace_numbers(text)
    # remove punctutation
    text = remove_words(text, file_name='punctuation.txt')
    # remove eng and blank and \n
    text = remove_words(text, remove_wrd_list=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ \n"))

    return text

def text_norm_after_cut(word_list):
    '''
    input is a word list, which is segmented by jieba
    '''
    def remove_words(word_list, file_name=None, remove_wrd_list=None):
        if file_name is not None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            all_remove = open(os.path.join(dir_path, file_name), 'r').read().splitlines()
            all_remove = [word.strip() for word in all_remove]
            for word in word_list:
                if word in all_remove:
                    word_list.remove(word)
        if remove_wrd_list is not None:
            for word in word_list:
                if word in remove_wrd_list:
                    word_list.remove(word)
        return word_list
    #get rid of stop words
    word_list = remove_words(word_list, file_name='stopWords.txt')

    return word_list


if __name__ == '__main__':
    import jieba
    dir_path = os.path.dirname(os.path.realpath(__file__))
    test_text = open(os.path.join(dir_path, 'test.txt'),'r').read()
    test_text = text_norm_before_cut(test_text)
    print(test_text, '\n')
    test_word_list = list(jieba.cut(test_text))
    test_word_list = text_norm_after_cut(test_word_list)
    print("".join(test_word_list))

