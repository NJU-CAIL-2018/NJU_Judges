import tensorflow as tf


class ImprisonmentNN:
    def __init__(self):
        # 参数设置
        self.batch_size = 64
        self.input_x = 12
        self.input_y = 128
        self.output_size = 3
        # 多标签预测返回的个数
        self.multi_label_count = 3
        # 正则项系数
        self.regular_scale = 0.001
        self._graph = tf.Graph()

        # 建立模型相关量
        self._x, self._y, self._keep_prob = self.build_placeholder()
        self.is_imprisonment_prediction, self.is_life_imprisonment_prediction, self.imprisonment_prediction = self.build_model()
        # 下面是回归任务代码
        self.loss = self.build_regression()
        self._train_op = self.build_train_op()
        self._result = self.get_one_result()

    # 返回训练时需要传入的 placeholder 的值
    def build_placeholder(self):
        with self.graph.as_default():
            # 声明一个占位符，None表示输入图片的数量不定，28*28图片分辨率
            x = tf.placeholder(tf.float32, [None, self.input_x * self.input_y])
            # 类别是0-9总共10个类别，对应输出分类结果
            y = tf.placeholder(tf.float32, [None, self.output_size])
            # dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
            # 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
            keep_prob = tf.placeholder(tf.float32)

            return x, y, keep_prob

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

    # 建立这个模型
    # 返回值为之后的输出向量，shape为 batch_size * output_size
    def build_model(self):
        with self.graph.as_default():
            # x_image又把xs reshape成了28*28*1的形状，因为是灰色图片，所以通道是1.作为训练时的input，-1代表图片数量不定
            x_image = tf.reshape(self.x, [-1, self.input_x, self.input_y, 1])

            ## 第一层卷积操作 ##
            # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
            w_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1),
                                  collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
            # 对于每一个卷积核都有一个对应的偏置量。
            b_conv1 = tf.Variable(tf.truncated_normal(shape=[32], stddev=0.1))
            # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            ## 第二层卷积操作 ##
            # 32通道卷积，卷积出64个特征
            w_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1),
                                  collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
            # 64个偏执数据
            b_conv2 = tf.Variable(tf.truncated_normal(shape=[64], stddev=0.1))
            # 注意h_pool1是上一层的池化结果，#卷积结果14x14x64
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 原图像尺寸10*128，现在尺寸为3*32，共有64张

            ## 第三层全连接操作 ##
            # 二维张量，第一个参数3*32*64的patch，也可以认为是只有一行3*32*64个数据的卷积，第二个参数代表卷积个数共1024个
            h_pool2_flat = tf.reshape(h_pool2, [-1, int(h_pool2.shape[1]) * int(h_pool2.shape[2]) * 64])
            h_fc1 = self._add_layer("layer1", h_pool2_flat, int(h_pool2.shape[1]) * int(h_pool2.shape[2]) * 64, 1024,
                                                         activation_function=tf.nn.relu)

            # dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
            # 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)  # 对卷积结果执行dropout操作

            # 第四层输出操作 ##
            is_imprisonment_prediction = self._add_layer("layer2_1", h_fc1_drop, 1024, 1,
                                                         activation_function=tf.identity)
            is_life_imprisonment_prediction = self._add_layer("layer2_2", h_fc1_drop, 1024, 1,
                                                              activation_function=tf.identity)
            imprisonment_prediction = self._add_layer("layer2_3", h_fc1_drop, 1024, 1,
                                                      activation_function=tf.identity)

            return is_imprisonment_prediction, is_life_imprisonment_prediction, imprisonment_prediction
            # 建立回归预测的损失函数

    def build_regression(self):
        with self.graph.as_default():
            mean_square_1 = tf.reduce_mean(
                tf.square(tf.slice(self.y, [0, 0], [-1, 1]) - self.is_imprisonment_prediction))
            mean_square_2 = tf.reduce_mean(
                tf.square(tf.slice(self.y, [0, 1], [-1, 1]) - self.is_life_imprisonment_prediction))
            mean_square_3 = tf.reduce_mean(
                tf.square(tf.slice(self.y, [0, 2], [-1, 1]) - self.imprisonment_prediction))
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
            result = tf.concat(
                [tf.concat([self.is_imprisonment_prediction, self.is_life_imprisonment_prediction], 0),
                 self.imprisonment_prediction], 0)
        return result

    @property
    def graph(self):
        return self._graph

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