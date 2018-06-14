import pickle
import tensorflow as tf
import numpy as np
import jieba

from .cnn_model_class import AccusationNN
from .cnn_model_class import ArticleNN
from .cnn_model_class import ImprisonmentNN


class Predictor:
    def __init__(self):
        self.batch_size = 1
        self.embedding_size = 128
        self.accu_size = 202
        self.row_size = 10
        # 建立三个模型
        self.accu_model = AccusationNN()
        self.article_model = ArticleNN()
        self.imprisonment_model = ImprisonmentNN()
        # build embedding
        self.dictionary, self.embedding = Predictor.get_dictionary_and_embedding()
        # build session
        self.accu_sess, self.article_sess, self.imprisonment_sess = self.load_model()

    def predict(self, content):
        vector = self.change_fact_to_matrices(content[0])[0]
        vector = vector.reshape([1, self.row_size * self.embedding_size])

        result = []
        for a in range(0, len(content)):
            result.append({
                "accusation": self.get_accu(vector),
                "imprisonment": self.get_imprisonment(vector),
                "articles": self.get_article(vector),
            })
        return result

    # get the result of accusation
    def get_accu(self, fact):
        value, index = self.accu_sess.run([self.accu_model.result_value, self.accu_model.result_index],
                                          feed_dict={self.accu_model.x: fact, self.accu_model.keep_prob: 1.0})
        accu = []
        for i, v in enumerate(value[0]):
            if v >= float(50 / self.accu_model.output_size):
                accu.append(index[0][i])

        return accu

    # get the result of article
    def get_article(self, fact):
        accu_input = np.ndarray([1, 2 * self.embedding_size])
        accu_input[0] = self.change_label_to_n_hot(self.get_accu(fact))
        input = np.concatenate((fact, accu_input), 1)

        value, index = self.article_sess.run([self.article_model.result_value, self.article_model.result_index],
                                          feed_dict={self.article_model.x: input, self.article_model.keep_prob: 1.0})
        article = []
        for i, v in enumerate(value[0]):
            if v >= float(50 / self.article_model.output_size):
                article.append(index[0][i])

        return article

    # get the result of imprisonment
    def get_imprisonment(self, fact):
        #print(fact)
        accu_input = np.ndarray([1, 2 * self.embedding_size])
        accu_input[0] = self.change_label_to_n_hot(self.get_accu(fact))

        input = np.concatenate((fact, accu_input), 1)
        result = self.imprisonment_sess.run(self.imprisonment_model.result,
                                             feed_dict={self.imprisonment_model.x: input, self.imprisonment_model.keep_prob: 1.0})

        #print(result)
        if(result[0][0] < 0):
            if(result[2][0] <= 0):
                return 5
            return int(result[2][0])
        else:
            if result[1][0] >= 0:
                return -1

        return -2

    @staticmethod
    def get_dictionary_and_embedding():
        with open("predictor/word2vec/dump_embedding.txt", "rb") as f:
            embedding = pickle.load(f)
        with open("predictor/word2vec/dump_dict.txt", "rb") as f:
            word_dictionary = pickle.load(f)

        return word_dictionary, embedding

    # label : 数据列，one-hot编码之后非零的列
    # max : 总类数
    def change_label_to_one_hot(self, label):
        result = np.zeros([self.accu_size + 1])
        for i in label:
            result[i] = 1

        return result

    def change_label_to_n_hot(self, label):
        vector = self.change_label_to_one_hot(label)
        result = np.zeros(shape=(2, self.embedding_size))
        result[0] = vector[:self.embedding_size]
        for i in range(0, self.embedding_size):
            if i + self.embedding_size < len(vector):
                result[1][i] = vector[i + self.embedding_size]
            else:
                result[1][i] = 0

        return result.reshape(2 * self.embedding_size)

    # data_X shape = [-1, embedding_size * row_size]
    def change_fact_to_matrices(self, fact):
        result = np.zeros(shape=(self.row_size, self.embedding_size))
        data_x = []
        row = 0
        for word in list(jieba.cut(fact, cut_all=False)):
            if row == self.row_size:
                row = 0
                matrix = result.copy().reshape((1, self.row_size * self.embedding_size))
                data_x.append(matrix[0])
                result = np.zeros(shape=(self.row_size, self.embedding_size))
            if word in self.dictionary and row < self.row_size:
                result[row] = self.embedding[self.dictionary[word]]
                row += 1
        matrix = result.copy().reshape((1, self.row_size * self.embedding_size))
        while row < self.row_size:
            result[row] = np.zeros(self.embedding_size)
            row += 1
        data_x.append(matrix[0])
        return data_x

    def change_fact_to_vector(self, fact):
        result = np.zeros(self.embedding_size)
        count = 0
        for word in list(jieba.cut(fact, cut_all=False)):
            if word in self.dictionary:
                count = count + 1
                result += self.embedding[self.dictionary[word]]

        if count != 0:
            result = result / count

        res = np.ndarray([1, self.embedding_size])
        res[0] = result
        return res

    def load_model(self):
        with self.accu_model.graph.as_default():
            accu_sess = tf.Session(graph=self.accu_model.graph)
            accu_sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state('predictor/accu_cnn_model')
            saver.restore(accu_sess, ckpt.model_checkpoint_path)

        with self.article_model.graph.as_default():
            article_sess = tf.Session(graph=self.article_model.graph)
            article_sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state('predictor/article_cnn_model')
            saver.restore(article_sess, ckpt.model_checkpoint_path)

        with self.imprisonment_model.graph.as_default():
            imprisonment_sess = tf.Session(graph=self.imprisonment_model.graph)
            imprisonment_sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state('predictor/imprisonment_cnn_model')
            saver.restore(imprisonment_sess, ckpt.model_checkpoint_path)

        return accu_sess, article_sess, imprisonment_sess
