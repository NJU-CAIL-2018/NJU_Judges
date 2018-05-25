import json
import pickle
import tensorflow as tf
import numpy as np
import jieba


class Predictor:
    def __init__(self):
        self.batch_size = 1
        self.embedding_size = 128
        self.accu_size = 203

        self.dictionary, self.embedding = Predictor.get_dictionary_and_embedding()
        # build model
        self.accu_x, self.accu_value, self.accu_index, self.accu_sess = self.build_accu_nn_mode()

    def build_accu_nn_mode(self):
        xs = tf.placeholder(tf.float32, [None, self.embedding_size])
        # 添加隐藏层1
        l1 = self.add_layer("layer1", xs, self.embedding_size, 64, activation_function=tf.sigmoid)
        # 添加隐藏层2
        # l2 = self.add_layer("layer2", l1, 256, 256, activation_function=tf.sigmoid)
        # 添加输出层
        prediction = self.add_layer("layer3", l1, 64, self.accu_size, activation_function=tf.nn.softmax)
        value, index = tf.nn.top_k(prediction, 3)
        accu_sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=2)
        ckpt = tf.train.get_checkpoint_state('./accu_nn_model')
        saver.restore(accu_sess, ckpt.model_checkpoint_path)

        return xs, value, index, accu_sess

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
        value, index = self.accu_sess.run([self.accu_value, self.accu_index], feed_dict={self.accu_x: fact})
        accu = []
        for i, v in enumerate(value[0]):
            print(v)
            if v >= float(40 / self.accu_size):
                accu.append(index[0][i])
        if len(accu) == 0:
            accu.append(index[0][0])

        return accu

    def add_layer(slef, layerName, inputs, in_size, out_size, activation_function=None):
        # add one more layer and return the output of this layer
        with tf.variable_scope(layerName, reuse=None):
            Weights = tf.get_variable("weights", shape=[in_size, out_size],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", shape=[1, out_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))

        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, Weights)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    @staticmethod
    def get_dictionary_and_embedding():
        with open("./word2vec/dump_embedding.txt", "rb") as f:
            embedding = pickle.load(f)
        with open("./word2vec/dump_dict.txt", "rb") as f:
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


import legal_instrument.system_path as constant

p = Predictor()
with open(constant.DATA_TEST, "r",
          encoding="UTF-8") as f:
    line = f.readline()
    out_file = open('../judge/output.txt', 'w', )
    while line:
        obj = json.loads(line)
        s = str(p.predict([obj['fact']])[0]) + "\n"
        print(s.replace('\'', '\"'))
        out_file.write(s.replace('\'', '\"'))
        line = f.readline()
print("out")
