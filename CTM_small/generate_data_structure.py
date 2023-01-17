import json
import argparse
import numpy as np
import pickle as pk
from utils.data_processed import punc_delete, get_cutter, sentence2index_matrix


def main(args):
    save_path = args.path + args.problem

    with open(args.path + 'w2id_thulac.pkl', 'rb') as f:
        word2id_dict = pk.load(f)
        f.close()

    file_list = ['Rtrain', 'Rvalid', 'Rtest']

    # Generate structure for CNN-based models
    max_length = 512
    for i in range(len(file_list)):
        fact_lists = []
        law_label_lists = []
        accu_label_lists = []
        term_lists = []
        with open(args.path + args.problem + '{0}.json'.format(file_list[i]), 'r', encoding='utf-8') as f:
            idx = 0
            for line in f.readlines():
                idx += 1
                line = json.loads(line)
                fact = line['fact_cut'].strip().split(' ')
                fact = punc_delete(fact)

                id_list = []
                word_num = 0
                for j in range(int(min(len(fact), max_length))):
                    if fact[j] in word2id_dict:
                        id_list.append(int(word2id_dict[fact[j]]))
                        word_num += 1
                    else:
                        id_list.append(int(word2id_dict['UNK']))
                while len(id_list) < max_length:
                    id_list.append(int(word2id_dict['BLANK']))

                if word_num <= 10:
                    print(", ".join(fact).encode(encoding="utf-8"))
                    print(idx, line['accu'])
                    continue

                id_numpy = np.array(id_list)

                fact_lists.append(id_numpy)
                law_label_lists.append(line['law'])
                accu_label_lists.append(line['accu'])
                term_lists.append(line['term'])
            f.close()
        data_dict = {'fact_list': fact_lists, 'law_label_lists': law_label_lists, 'accu_label_lists': accu_label_lists,
                     'term_lists': term_lists}
        pk.dump(data_dict, open(save_path + '{0}_processed_thulac.pkl'.format(file_list[i]), 'wb'))
        print('For CNN-based models: {0}_dataset is processed over'.format(file_list[i])+'\n')

    # Generate structure for LSTM-based models
    dict_path = args.path + 'Thuocl_seg.txt'
    stopword_path = args.path + 'stop_word.txt'
    cut = get_cutter(dict_path=dict_path, stopword_path=stopword_path, stop_words_filtered=False)
    doc_len = 15
    sent_len = 100

    for i in range(len(file_list)):
        fact_lists = []
        law_label_lists = []
        accu_label_lists = []
        term_lists = []
        with open(args.path + args.problem + '{0}.json'.format(file_list[i]), 'r', encoding='utf-8') as f:
            idx = 0
            for line in f.readlines():
                idx += 1
                line = json.loads(line)
                fact = line['fact_cut']
                sentence, word_num, sent_words = sentence2index_matrix(fact, word2id_dict, doc_len, sent_len, cut)

                if word_num <= 10:
                    print(", ".join(fact).encode(encoding="utf-8"))
                    print(idx, ", ".join([word for sent in sent_words for word in sent]).encode(encoding="utf-8"))
                    continue

                fact_lists.append(sentence)
                law_label_lists.append(line['law'])
                accu_label_lists.append(line['accu'])
                term_lists.append(line['term'])
            f.close()
        data_dict = {'fact_list': fact_lists, 'law_label_lists': law_label_lists, 'accu_label_lists': accu_label_lists,
                     'term_lists': term_lists}
        pk.dump(data_dict, open(save_path + '{0}_processed_thulac_Legal_basis.pkl'.format(file_list[i]), 'wb'))
        print('For LSTM-based models: {0}_dataset is processed over'.format(file_list[i]) + '\n')


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Structure")
    parser.add_argument('-p', dest='path', default='datax/')
    parser.add_argument('-d', dest='problem', default='small/')  # or 'big/'
    args = parser.parse_args()
    main(args)
