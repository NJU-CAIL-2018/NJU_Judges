import numpy as np
import jieba
import legal_instrument.data_util.generate_batch as generator

# param
embedding_size = 128


def change_fact_to_matrices(fact, embedding, row_size, dictionary):
    result = np.zeros(shape=(row_size, embedding_size))
    data_x = []
    row = 0
    for word in list(jieba.cut(fact, cut_all=False)):
        if row == row_size:
            row = 0
            matrix = result.copy()
            data_x.append(matrix)
            result = np.zeros(shape=(embedding_size, row_size))
        if word in dictionary and row < row_size:
            result[row] = embedding[dictionary[word]]
            row += 1
    matrix = result.copy()
    while row < 10:
        result[row] = np.zeros(embedding_size)
        row += 1
    data_x.append(matrix)
    return data_x


word_dict, embedding, reverse_dictionary = generator.get_dictionary_and_embedding()
data_x = change_fact_to_matrices("公诉机关指控：2016年3月28日20时许，被告人颜某在本市洪山区马湖新村足球场马路边捡拾到被害人谢某的VIVOX5手机一部", embedding, 10, word_dict)
print(data_x)