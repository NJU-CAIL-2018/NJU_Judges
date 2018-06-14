import numpy as np
import json
import jieba
import legal_instrument.data_util.generate_batch as generator
import datetime
import legal_instrument.system_path as constant

# param
embedding_size = 128


# def change_fact_to_matrices(fact, embedding, row_size, dictionary):
#     result = np.zeros(shape=(row_size, embedding_size))
#     data_x = []
#     row = 0
#     for word in list(jieba.cut(fact, cut_all=False)):
#         if row == row_size:
#             row = 0
#             matrix = result.copy()
#             data_x.append(matrix)
#             result = np.zeros(shape=(row_size, embedding_size))
#         if word in dictionary and row < row_size:
#             result[row] = embedding[dictionary[word]]
#             row += 1
#     matrix = result.copy()
#     while row < row_size:
#         result[row] = np.zeros(embedding_size)
#         row += 1
#     data_x.append(matrix)
#     return data_x


# data_X shape = [-1, embedding_size * row_size]
def change_fact_to_matrices(fact, embedding, row_size, dictionary):
    result = np.zeros(shape=(row_size, embedding_size))
    data_x = []
    row = 0
    for word in list(jieba.cut(fact, cut_all=False)):
        if row == row_size:
            row = 0
            matrix = result.copy().reshape((1, row_size * embedding_size))
            data_x.append(matrix[0])
            result = np.zeros(shape=(row_size, embedding_size))
        if word in dictionary and row < row_size:
            result[row] = embedding[dictionary[word]]
            row += 1
    matrix = result.copy().reshape((1, row_size * embedding_size))
    while row < row_size:
        result[row] = np.zeros(embedding_size)
        row += 1
    data_x.append(matrix[0])
    return data_x


# data_X shape = [-1, embedding_size * (row_size + 2)]
def read_data_in_imprisonment_format_with_accu(file_name, embedding, row_size, dictionary, accu_dict):
    data = []
    # control data size
    i = 0
    data_x = []
    data_y = []

    time = datetime.datetime.now()
    with open(file_name, "r", encoding="UTF-8") as f:
        line = f.readline()

        while line and i < 10000:
            i = i + 1
            obj = json.loads(line)
            accusation = obj['meta']['accusation']
            imprisonment = obj['meta']['term_of_imprisonment']['imprisonment']
            death_penalty = obj['meta']['term_of_imprisonment']['death_penalty']
            life_imprisonment = obj['meta']['term_of_imprisonment']['life_imprisonment']
            cur_y = []

            accu = []
            for accusation in accusation:
                accu.append(accu_dict[accusation])

            if death_penalty:
                cur_y.append(1)
                cur_y.append(-1)
            elif life_imprisonment:
                cur_y.append(1)
                cur_y.append(1)
            else:
                cur_y.append(-1)
                cur_y.append(0)

            cur_y.append(imprisonment)

            data_list = change_fact_to_matrices(obj['fact'], embedding, row_size, dictionary)
            for data in data_list:
                data_x.append(np.concatenate((data, change_label_to_n_hot(accu, len(accu_dict)))))
                data_y.append(cur_y)

            if i % 1000 == 0:
                print("read ", i, "lines")
            line = f.readline()

    result_x = np.ndarray([len(data_x), embedding_size * row_size + 2 * embedding_size])
    result_y = np.ndarray([len(data_y), 3], dtype='float')
    for i in range(len(data_x)):
        result_x[i] = data_x[i]
        result_y[i][0] = data_y[i][0]
        result_y[i][1] = data_y[i][1]
        result_y[i][2] = data_y[i][2]

    print(datetime.datetime.now() - time)

    return result_x, result_y


def read_data_in_imprisonment_format(file_name, embedding, row_size, dictionary, accu_dict):
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
            accusation = obj['meta']['accusation']
            imprisonment = obj['meta']['term_of_imprisonment']['imprisonment']
            death_penalty = obj['meta']['term_of_imprisonment']['death_penalty']
            life_imprisonment = obj['meta']['term_of_imprisonment']['life_imprisonment']
            cur_y = []

            accu = []
            for accusation in accusation:
                accu.append(accu_dict[accusation])

            if death_penalty:
                cur_y.append(1)
                cur_y.append(-1)
            elif life_imprisonment:
                cur_y.append(1)
                cur_y.append(1)
            else:
                cur_y.append(-1)
                cur_y.append(0)

            cur_y.append(imprisonment)

            data_list = change_fact_to_matrices(obj['fact'], embedding, row_size, dictionary)
            for data in data_list:
                data_x.append(data)
                data_y.append(cur_y)

            if i % 1000 == 0:
                print("read ", i, "lines")
            line = f.readline()

    result_x = np.ndarray([len(data_x), embedding_size * row_size])
    result_y = np.ndarray([len(data_y), 3], dtype='float')
    for i in range(len(data_x)):
        result_x[i] = data_x[i]
        result_y[i][0] = data_y[i][0]
        result_y[i][1] = data_y[i][1]
        result_y[i][2] = data_y[i][2]

    print(datetime.datetime.now() - time)

    return result_x, result_y


# we can only assmue one_hot is true because we are dealing the multi-label
# data_X shape = [-1, embedding_size * row_size]
def read_data_in_accu_format(file_name, embedding, row_size, dictionary, accu_dict, one_hot=True):
    data = []
    # control data size
    i = 0
    data_x = []
    data_y = []

    time = datetime.datetime.now()
    with open(file_name, "r", encoding="UTF-8") as f:
        line = f.readline()

        while line and i < 10000:
            i = i + 1
            obj = json.loads(line)
            l = obj['meta']['accusation']

            accu_y = []
            for accusation in l:
                accu_y.append(accu_dict[accusation])

            data_list = change_fact_to_matrices(obj['fact'], embedding, row_size, dictionary)
            for data in data_list:
                data_x.append(data)
                data_y.append(accu_y)

            if i % 1000 == 0:
                print("read ", i, "lines")
            line = f.readline()

    result_x = np.ndarray([len(data_x), embedding_size * row_size])
    if one_hot:
        result_y = np.ndarray([len(data_y), len(accu_dict) + 1])
        for i in range(len(data_x)):
            result_x[i] = data_x[i]
            result_y[i] = change_label_to_one_hot(data_y[i], len(accu_dict))

    else:
        result_y = np.ndarray([len(data_y)])
        for i in range(len(data_x)):
            result_x[i] = data_x[i]
            result_y[i] = data_y[i][0]

    print(datetime.datetime.now() - time)

    return result_x, result_y


# we can only assmue one_hot is true because we are dealing the multi-label
# data_X shape = [-1, embedding_size * (row_size + 2)]
def read_data_in_article_format(file_name, embedding, row_size, dictionary, article_dict, one_hot=True):
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
            l = obj['meta']['relevant_articles']

            article_y = []
            for article in l:
                article_y.append(article_dict[article])

            data_list = change_fact_to_matrices(obj['fact'], embedding, row_size, dictionary)
            for data in data_list:
                data_x.append(data)
                data_y.append(article_y)

            if i % 1000 == 0:
                print("read ", i, "lines")
            line = f.readline()

    result_x = np.ndarray([len(data_x), embedding_size * row_size])
    if one_hot:
        result_y = np.ndarray([len(data_y), len(article_dict) + 1])
        for i in range(len(data_x)):
            result_x[i] = data_x[i]
            result_y[i] = change_label_to_one_hot(data_y[i], len(article_dict))

    else:
        result_y = np.ndarray([len(data_y)])
        for i in range(len(data_x)):
            result_x[i] = data_x[i]
            result_y[i] = data_y[i][0]

    print(datetime.datetime.now() - time)

    return result_x, result_y


def read_data_in_article_format_with_accu(file_name, embedding, row_size, dictionary, article_dict, one_hot=True):
    # control data size
    i = 0
    data_x = []
    data_y = []

    time = datetime.datetime.now()
    with open(file_name, "r", encoding="UTF-8") as f:
        line = f.readline()

        while line and i < 10000:
            i = i + 1
            obj = json.loads(line)
            l = obj['meta']['relevant_articles']
            accusation = obj['meta']['accusation']

            accu = []
            for accusation in accusation:
                accu.append(accu_dict[accusation])

            article_y = []
            for article in l:
                article_y.append(article_dict[article])

            data_list = change_fact_to_matrices(obj['fact'], embedding, row_size, dictionary)
            for data in data_list:
                data_x.append(np.concatenate((data, change_label_to_n_hot(accu, len(accu_dict)))))
                data_y.append(article_y)

            if i % 1000 == 0:
                print("read ", i, "lines")
            line = f.readline()

    result_x = np.ndarray([len(data_x), embedding_size * row_size + 2 * embedding_size])
    if one_hot:
        result_y = np.ndarray([len(data_y), len(article_dict) + 1])
        for i in range(len(data_x)):
            result_x[i] = data_x[i]
            result_y[i] = change_label_to_one_hot(data_y[i], len(article_dict))

    else:
        result_y = np.ndarray([len(data_y)])
        for i in range(len(data_x)):
            result_x[i] = data_x[i]
            result_y[i] = data_y[i][0]

    print(datetime.datetime.now() - time)

    return result_x, result_y


# label : 数据列，one-hot编码之后非零的列
# max : 总类数
def change_label_to_one_hot(label, max):
    result = np.zeros([max + 1])
    for i in label:
        result[i] = 1

    return result


# accusation size = 203
# article size = 184
def change_label_to_n_hot(label, max):
    vector = change_label_to_one_hot(label, max)
    result = np.zeros(shape=(2, embedding_size))
    result[0] = vector[:embedding_size]
    for i in range(0, embedding_size):
        if i + embedding_size < len(vector):
            result[1][i] = vector[i + embedding_size]
        else:
            result[1][i] = 0

    return result.reshape(2 * embedding_size)


accu_dict, reverse_accu_dict = generator.read_accu()
article_dict, reverse_article_dict = generator.read_article()
word_dict, embedding, reverse_dictionary = generator.get_dictionary_and_embedding()
data_x = change_fact_to_matrices(
    "公诉机关指控：2016年3月28日20时许，被告人颜某在本市洪山区马湖新村足球场马路边捡拾到被害人谢某的VIVOX5手机一部,公诉机关指控：2016年3月28日20时许，被告人颜某在本市洪山区马湖新村足球场马路边捡拾到被害人谢某的VIVOX5手机一部,公诉机关指控：2016年3月28日20时许，被告人颜某在本市洪山区马湖新村足球场马路边捡拾到被害人谢某的VIVOX5手机一部",
    embedding, 10, word_dict)
print(data_x)
# for i in range(0, len(data_x)):
#     print(len(data_x[i]))

# test part
# train_data_x, train_data_y = read_data_in_accu_format(constant.DATA_TRAIN, embedding, 10, word_dict, accu_dict,
#                                                       one_hot=True)
# print(train_data_x)
# assert len(train_data_x[0]) == embedding_size * 10
#
# train_data_x, train_data_y = read_data_in_article_format(constant.DATA_TRAIN, embedding, 15, word_dict, article_dict,
#                                                          one_hot=True)
#
# print(train_data_x)
# assert len(train_data_x[0]) == embedding_size * 15
#
# train_data_x, train_data_y = read_data_in_article_format_with_accu(constant.DATA_TRAIN, embedding, 15, word_dict,
#                                                                    article_dict,
#                                                                    one_hot=True)
#
# print(train_data_x)
# assert len(train_data_x[0]) == embedding_size * 15 + 2 * embedding_size
#
# train_data_x, train_data_y = read_data_in_imprisonment_format(constant.DATA_TRAIN, embedding, 13, word_dict, accu_dict)
#
# print(train_data_x)
# assert len(train_data_x[0]) == embedding_size * 13
#
# train_data_x, train_data_y = read_data_in_imprisonment_format_with_accu(constant.DATA_TRAIN, embedding, 12, word_dict,
#                                                                         accu_dict)
#
# print(train_data_x)
# assert len(train_data_x[0]) == embedding_size * 12 + 2 * embedding_size
