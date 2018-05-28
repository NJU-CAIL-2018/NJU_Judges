import pickle
import tensorflow as tf
import numpy as np
import jieba

from .model_class import AccusationNN


class Predictor:
    def __init__(self):
        self.batch_size = 1
        self.embedding_size = 128
        self.accu_model = AccusationNN()
        self.dictionary, self.embedding = Predictor.get_dictionary_and_embedding()
        self.accu_sess = self.load_model()

    def predict(self, content):
        vector = self.change_fact_to_vector(content[0])

        result = []
        for a in range(0, len(content)):
            result.append({
                "accusation": self.get_accu(vector),
                "imprisonment": 5,
                "articles": [5, 7, 9]
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

        return accu_sess
