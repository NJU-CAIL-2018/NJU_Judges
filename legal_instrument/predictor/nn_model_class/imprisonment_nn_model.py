import tensorflow as tf


##
class ImprisonmentNN:
    def __init__(self):
        # 参数设置
        self.training_batch_size = 256
        self.valid_batch_size = 256
        self.iteration = 100000
        # embediing size = 128
        # accu_size = 203
        self.input_size = 128 + 203
        self.output_size = 3
        # 图
        self._graph = tf.Graph()

        # 建立模型相关量
        self._x, self._y, self._keep_prob = self.build_placeholder()
        self.is_imprisonment_prediction, self.is_life_imprisonment_prediction, self.imprisonment_prediction = self.build_model()
        # 下面是回归任务代码
        self.loss = self.build_regression()
        self._train_op = self.build_train_op()
        self._result = self.get_one_result()

    # 增加一层神经网络的抽象函数
    def _add_layer(self, layerName, inputs, in_size, out_size, activation_function=None):
        # add one more layer and return the output of this layer
        with tf.variable_scope(layerName, reuse=None):
            Weights = tf.get_variable("weights", shape=[in_size, out_size],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)
            biases = tf.get_variable("biases", shape=[1, out_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)

        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, Weights)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    # 返回训练时需要传入的 placeholder 的值
    def build_placeholder(self):
        with self._graph.as_default():
            x = tf.placeholder(tf.float32, [None, self.input_size])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            # dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
            # 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
            keep_prob = tf.placeholder(tf.float32)

        return x, y, keep_prob

    # 建立这个模型
    # 返回值为之后的输出向量，shape为 batch_size * output_size
    def build_model(self):
        with self._graph.as_default():
            # 添加隐藏层1
            l1 = self._add_layer("layer1", self.x, self.input_size, 64, activation_function=tf.sigmoid)
            # 添加隐藏层2
            l2 = self._add_layer("layer2", l1, 64, 128, activation_function=tf.sigmoid)
            l2_drop = tf.nn.dropout(l2, self._keep_prob)
            # 添加输出层
            is_imprisonment_prediction = self._add_layer("layer3_1", l2_drop, 128, 1, activation_function=tf.identity)
            is_life_imprisonment_prediction = self._add_layer("layer3_2", l2_drop, 128, 1,
                                                              activation_function=tf.identity)
            imprisonment_prediction = self._add_layer("layer3_3", l2_drop, 128, 1, activation_function=tf.identity)

        return is_imprisonment_prediction, is_life_imprisonment_prediction, imprisonment_prediction

    # 建立回归预测的损失函数
    def build_regression(self):
        with self.graph.as_default():
            mean_square_1 = tf.reduce_mean(
                tf.square(tf.slice(self.y, [0, 0], [-1, 1]) - self.is_imprisonment_prediction))
            mean_square_2 = tf.reduce_mean(
                tf.square(tf.slice(self.y, [0, 1], [-1, 1]) - self.is_life_imprisonment_prediction))
            mean_square_3 = tf.reduce_mean(tf.square(tf.slice(self.y, [0, 2], [-1, 1]) - self.imprisonment_prediction))
            reg_term = self._build_regular_term()
            loss = mean_square_1 + mean_square_2 + mean_square_3 + reg_term

        return loss

    def _build_regular_term(self):
        with self.graph.as_default():
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
            reg_term = tf.contrib.layers.apply_regularization(regularizer)
        return reg_term

    # 建立训练张量
    def build_train_op(self):
        with self.graph.as_default():
            train_op = tf.train.AdamOptimizer(beta2=0.9999).minimize(self.loss)

        return train_op

    # 拟合数据集
    def get_one_result(self):
        with self.graph.as_default():
            print(self.is_imprisonment_prediction)
            print(self.is_life_imprisonment_prediction)
            print(self.imprisonment_prediction)
            result = tf.concat([tf.concat([self.is_imprisonment_prediction, self.is_life_imprisonment_prediction], 0),
                               self.imprisonment_prediction], 0)

        return result

    @property
    def graph(self):
        return self._graph

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def train_op(self):
        return self._train_op

    @property
    def result(self):
        return self._result
