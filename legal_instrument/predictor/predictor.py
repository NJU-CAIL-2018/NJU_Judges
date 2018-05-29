import pickle
import tensorflow as tf
import numpy as np
import jieba

from .model_class import AccusationNN
from .model_class import ArticleNN
from .model_class import ImprisonmentNN


class Predictor:
    def __init__(self):
        self.batch_size = 1
        self.embedding_size = 128
        # 建立三个模型
        self.accu_model = AccusationNN()
        self.article_model = ArticleNN()
        self.imprisonment_model = ImprisonmentNN()
        # build embedding
        self.dictionary, self.embedding = Predictor.get_dictionary_and_embedding()
        # build session
        self.accu_sess, self.article_sess, self.imprisonment_sess = self.load_model()

    def predict(self, content):
        vector = self.change_fact_to_vector(content[0])

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
        value, index = self.article_sess.run([self.article_model.result_value, self.article_model.result_index],
                                          feed_dict={self.article_model.x: fact, self.article_model.keep_prob: 1.0})
        article = []
        for i, v in enumerate(value[0]):
            if v >= float(50 / self.article_model.output_size):
                article.append(index[0][i])

        return article

    # get the result of imprisonment
    def get_imprisonment(self, fact):
        result = self.imprisonment_sess.run(self.imprisonment_model.result,
                                             feed_dict={self.imprisonment_model.x: fact, self.imprisonment_model.keep_prob: 1.0})

        return int(result[0][0])

    @staticmethod
    def get_dictionary_and_embedding():
        with open("predictor/word2vec/dump_embedding.txt", "rb") as f:
            embedding = pickle.load(f)
        with open("predictor/word2vec/dump_dict.txt", "rb") as f:
            word_dictionary = pickle.load(f)

        return word_dictionary, embedding

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
            ckpt = tf.train.get_checkpoint_state('predictor/accu_nn_model')
            saver.restore(accu_sess, ckpt.model_checkpoint_path)

        with self.article_model.graph.as_default():
            article_sess = tf.Session(graph=self.article_model.graph)
            article_sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state('predictor/article_nn_model')
            saver.restore(article_sess, ckpt.model_checkpoint_path)

        with self.imprisonment_model.graph.as_default():
            imprisonment_sess = tf.Session(graph=self.imprisonment_model.graph)
            imprisonment_sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state('predictor/imprisonment_nn_model')
            saver.restore(imprisonment_sess, ckpt.model_checkpoint_path)

        return accu_sess, article_sess, imprisonment_sess
