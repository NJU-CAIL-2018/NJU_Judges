import random
import json
import jieba

import legal_instrument.system_path as constant
import pickle
import numpy as np

# param
embedding_size = 128
num_sampled = 64
vocabulary_size = 10000
batch_size = 128


##
## change linux file to windows file
def change_new_line():
    file = open('C:\\Users\\njdx\\Desktop\\文书\\windows_law.txt', "w", encoding="UTF-8")
    with open('C:\\Users\\njdx\\Desktop\\文书\\law.txt', "r", encoding="UTF-8") as f:
        line = f.readline()
        while line:
            line = line.replace('\n', '\r\n')
            file.write(line)
            line = f.readline()


def read_accu():
    accu_dict = dict()
    with open(constant.ACCU_FILE, "r", encoding="UTF-8") as f:
        line = f.readline()
        while line:
            accu_dict[line.strip()] = len(accu_dict)
            line = f.readline()
    reverse_accu_dict = dict(zip(accu_dict.values(), accu_dict.keys()))

    return accu_dict, reverse_accu_dict


def get_dictionary_and_embedding():
    with open("./dump_data/dump_embedding.txt", "rb") as f:
        embedding = pickle.load(f)
    with open("./dump_data/dump_dict.txt", "rb") as f:
        word_dictionary = pickle.load(f)

    return word_dictionary, embedding, dict(zip(word_dictionary.values(), word_dictionary.keys()))


def change_fact_to_vector(fact, embedding, dictionary):
    result = np.zeros(embedding_size)
    count = 0
    for word in list(jieba.cut(fact, cut_all=False)):
        if word in dictionary:
            count = count + 1
            result += embedding[dictionary[word]]

    if count != 0:
        result = result / count
    return result


# label : 数据列，one-hot编码之后非零的列
# max : 总类数
def change_label_to_one_hot(label, max):
    return np.eye(max)[label]


# read file into memory
def read_data_in_accu_format(file_name, accu_size, embedding, dictionary, accu_dict, one_hot = True):
    data = []
    # control data size
    i = 0
    data_x = []
    data_y = []
    with open(file_name, "r", encoding="UTF-8") as f:
        line = f.readline()
        while line and i < 1000:
            i = i + 1
            obj = json.loads(line)
            l = obj['meta']['accusation']

            for index, accusation in enumerate(l):
                if accusation in accu_dict:
                    data_x.append(change_fact_to_vector(obj['fact'], embedding, dictionary))
                    if one_hot:
                        data_y.append(change_label_to_one_hot(accu_dict[accusation], accu_size))
                    else:
                        data_y.append(accu_dict[accusation])

            line = f.readline()

    return data_x, data_y


def generate_batch(batch_size, data_x, data_y, label_size):
    x = np.ndarray([batch_size, embedding_size])
    y = np.ndarray([batch_size, label_size], dtype=int)

    for i in range(batch_size):
        index = random.randint(0, len(data_x) - 1)
        x[i] = data_x[index]
        y[i] = data_y[index]

    return x, y
