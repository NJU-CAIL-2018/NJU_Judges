import random
import json
import jieba

import legal_instrument.system_path as constant
import pickle
import numpy as np
import datetime


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
            accu_dict[line.strip()] = len(accu_dict) + 1
            line = f.readline()
    reverse_accu_dict = dict(zip(accu_dict.values(), accu_dict.keys()))

    return accu_dict, reverse_accu_dict


def get_dictionary_and_embedding():
    with open("../dump_data/word_vector/dump_embedding.txt", "rb") as f:
        embedding = pickle.load(f)
    with open("../dump_data/word_vector/dump_dict.txt", "rb") as f:
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
# this function is slower than below
# def read_data_in_accu_format(file_name, accu_size, embedding, dictionary, accu_dict, one_hot = True):
#     data = []
#     # control data size
#     i = 0
#     data_x = np.zeros([1, embedding_size])
#     if one_hot :
#         data_y = np.zeros([1, len(accu_dict)])
#     else:
#         data_y = np.zeros([1, 1], dtype=int)
#
#     time = datetime.datetime.now()
#
#     with open(file_name, "r", encoding="UTF-8") as f:
#         line = f.readline()
#
#         while line and i < 10000:
#             i = i + 1
#             obj = json.loads(line)
#             l = obj['meta']['accusation']
#
#             for index, accusation in enumerate(l):
#                 if accusation in accu_dict:
#                     data_x = np.row_stack((data_x, change_fact_to_vector(obj['fact'], embedding, dictionary)))
#                     if one_hot:
#                         data_y = np.row_stack((data_y, change_label_to_one_hot(accu_dict[accusation], accu_size)))
#                     else:
#                         data_y = np.row_stack((data_y,accu_dict[accusation]))
#             if i % 1000 == 0:
#                 print("read ", i, "lines")
#             line = f.readline()
#
#     print(datetime.datetime.now() - time)
#
#     return data_x[1:], data_y[1:]
def read_data_in_imprisonment_format(file_name, embedding, dictionary):
    data = []
    # control data size
    i = 0
    data_x = []
    data_y = []

    time = datetime.datetime.now()
    with open(file_name, "r", encoding="UTF-8") as f:
        line = f.readline()

        while line:
            i = i + 1
            obj = json.loads(line)
            l = obj['meta']['term_of_imprisonment']['imprisonment']

            data_x.append(change_fact_to_vector(obj['fact'], embedding, dictionary))
            data_y.append(l)

            if i % 1000 == 0:
                print("read ", i, "lines")
            line = f.readline()

    result_x = np.ndarray([len(data_x), embedding_size])
    result_y = np.ndarray([len(data_y)], dtype='float')
    for i in range(len(data_x)):
        result_x[i] = data_x[i]
        result_y[i] = data_y[i]

    print(datetime.datetime.now() - time)

    return result_x, result_y


def read_data_in_accu_format(file_name, accu_size, embedding, dictionary, accu_dict, one_hot = True):
    data = []
    # control data size
    i = 0
    data_x = []
    data_y = []

    time = datetime.datetime.now()
    with open(file_name, "r", encoding="UTF-8") as f:
        line = f.readline()

        while line:
            i = i + 1
            obj = json.loads(line)
            l = obj['meta']['accusation']

            for index, accusation in enumerate(l):
                if accusation in accu_dict:
                    data_x.append(change_fact_to_vector(obj['fact'], embedding, dictionary))
                    data_y.append(accu_dict[accusation])

            if i % 1000 == 0:
                print("read ", i, "lines")
            line = f.readline()

    result_x = np.ndarray([len(data_x), embedding_size])
    if one_hot:
        result_y = np.ndarray([len(data_y), accu_size])
        for i in range(len(data_x)):
            result_x[i] = data_x[i]
            result_y[i] = change_label_to_one_hot(data_y[i], accu_size)

    else:
        result_y = np.ndarray([len(data_y)])
        for i in range(len(data_x)):
            result_x[i] = data_x[i]
            result_y[i] = data_y[i]

    print(datetime.datetime.now() - time)

    return result_x, result_y


def generate_batch(batch_size, data_x, data_y):
    x = np.ndarray([batch_size, embedding_size])
    if len(data_y.shape) > 1:
        y = np.ndarray([batch_size, len(data_y[0])], dtype=int)
    else:
        y = np.ndarray([batch_size], dtype=int)
    #print(len(data_y[0]))

    for i in range(batch_size):
        index = random.randint(0, len(data_x) - 1)
        x[i] = data_x[index]
        y[i] = data_y[index]

    return x, y
