import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class ArticleNN:
    def __init__(self):
        # 参数设置
        self.batch_size = 64
        self.input_x = 12
        self.input_y = 128
        # article_size = 184
        self._output_size = 184
        # 多标签预测返回的个数
        self.multi_label_count = 3
        # 正则项系数
        self.regular_scale = 0.001
        self._graph = tf.Graph()

        # 建立模型相关量
        self._x, self._y, self._keep_prob = self.build_placeholder()
        self.row_prediction = self.build_model()
        # 下面是对于单标签分类的代码
        # self.loss = self.build_one_lebal_loss()
        # self.train_op = self.build_train_op()
        # self.result = self.get_one_result()
        # self.accuracy = self.one_result_accuracy()
        # 下面是对多标签分类的代码
        self.loss = self.build_muti_lebal_loss()
        self._train_op = self.build_train_op()
        self._result_value, self._result_index = self.get_multi_result()

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
            W_fc1 = tf.Variable(
                tf.truncated_normal(shape=[int(h_pool2.shape[1]) * int(h_pool2.shape[2]) * 64, 1024], stddev=0.1),
                collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
            # 1024个偏执数据
            b_fc1 = tf.Variable(tf.truncated_normal(shape=[1024], stddev=0.1))
            # 将第二层卷积池化结果reshape成只有一行3*32*64个数据# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
            h_pool2_flat = tf.reshape(h_pool2, [-1, int(h_pool2.shape[1]) * int(h_pool2.shape[2]) * 64])
            # 卷积操作，结果是1*1*1024，单行乘以单列等于1*1矩阵，matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘，自动认为是前行向量后列向量
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
            # 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)  # 对卷积结果执行dropout操作

            # 第四层输出操作 ##
            # 二维张量，1*1024矩阵卷积，输出结果与 output_size 大小一致
            W_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, self.output_size], stddev=0.1),
                                collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
            b_fc2 = tf.Variable(tf.truncated_normal(shape=[self.output_size], stddev=0.1))
            # 最后的分类，结果为batch_size * output_size
            prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            return prediction

    # 建立单结果预测的损失函数
    def build_one_lebal_loss(self):
        with self.graph.as_default():
            cross_entropy = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.row_prediction))
            reg_term = self._build_regular_term()
            loss = cross_entropy + reg_term

            return loss

    # 建立正则项
    def _build_regular_term(self):
        with self.graph.as_default():
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.regular_scale)
            reg_term = tf.contrib.layers.apply_regularization(regularizer)
            return reg_term

    # 建立多结果预测的损失函数
    def build_muti_lebal_loss(self):
        with self.graph.as_default():
            cross_entropy = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.row_prediction))
            reg_term = self._build_regular_term()
            loss = cross_entropy + reg_term

            return loss

    # 建立训练张量
    def build_train_op(self):
        with self.graph.as_default():
            train_op = tf.train.AdamOptimizer(beta2=0.9999).minimize(self.loss)

            return train_op

    # 拟合数据集
    def get_one_result(self):
        with self.graph.as_default():
            soft_max = tf.nn.softmax(self.row_prediction)
            result = tf.argmax(soft_max, 1)

            return result

    # 得到单标签准确度
    def one_result_accuracy(self):
        with self.graph.as_default():
            soft_max = tf.nn.softmax(self.row_prediction)
            correct_prediction = tf.equal(tf.argmax(soft_max, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return accuracy

    def get_multi_result(self):
        with self.graph.as_default():
            soft_max = tf.nn.softmax(self.row_prediction)
            # value -> 对应的是 top k 的概率值
            # index -> 对应的是 top k 的下标
            # 举个例子 [1,5,2,4,6]  top 2 : value -> [6, 5]  index -> [4,1]
            value, index = tf.nn.top_k(soft_max, k=self.multi_label_count)

            return value, index

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
    def result_value(self):
        return self._result_value

    @property
    def result_index(self):
        return self._result_index

    @property
    def output_size(self):
        return self._output_size


##############################################
# train part don't place it in the predictor #
##############################################
training_batch_size = 64
valid_batch_size = 128
embedding_size = 128
iteration = 100000
##

import pickle
import legal_instrument.data_util.generate_batch as generator
import legal_instrument.system_path as constant

print("reading data from training set...")
try:
    with open('./dump_data/cnn/dump_train_x.txt', 'rb') as f:
        train_data_x = pickle.load(f)

    with open('./dump_data/cnn/dump_train_y_label.txt', 'rb') as f:
        train_data_y = pickle.load(f)

    with open('./dump_data/cnn/dump_valid_x.txt', 'rb') as f:
        valid_data_x = pickle.load(f)

    with open('./dump_data/cnn/dump_valid_y_label.txt', 'rb') as f:
        valid_data_y = pickle.load(f)

    with open('./dump_data/cnn/dump_test_x.txt', 'rb') as f:
        test_data_x = pickle.load(f)

    with open('./dump_data/cnn/dump_test_y_label.txt', 'rb') as f:
        test_data_y = pickle.load(f)
except:
    print("No dump file read original file! Please wait... "
          "If u want to accelerate this process, please see read_me -> transform_data_to_feature_and_dump")
    accu_dict, reverse_accu_dict = generator.read_accu()
    word_dict, embedding, reverse_dictionary = generator.get_dictionary_and_embedding()

    train_data_x, train_data_y = generator.read_data_in_accu_format(constant.DATA_TRAIN, embedding,
                                                                    word_dict, accu_dict, one_hot=True)
    valid_data_x, valid_data_y = generator.read_data_in_accu_format(constant.DATA_VALID, embedding,
                                                                    word_dict, accu_dict, one_hot=True)
    test_data_x, test_data_y = generator.read_data_in_accu_format(constant.DATA_TEST, embedding,
                                                                  word_dict, accu_dict, one_hot=True)

print("reading complete!")
# just test generate_accu_batch
x, y = generator.generate_batch(training_batch_size, train_data_x, train_data_y)
print(x.shape)

print("data load complete")
print("The model begin here")

print(len(train_data_y[0]))

model = ArticleNN()
# run part
with model.graph.as_default():
    with tf.Session() as sess:
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        # 保存参数所用的保存器
        saver = tf.train.Saver(max_to_keep=1)
        # get latest file
        ckpt = tf.train.get_checkpoint_state('./xkf_text_cnn_model')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 可视化部分
        tf.summary.scalar("loss", model.loss)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./xkf_text_cnn_logs", sess.graph)

        # training part
        for i in range(iteration):
            x, y = generator.generate_batch(training_batch_size, train_data_x, train_data_y)

            x_valid, y_valid = generator.generate_batch(training_batch_size, valid_data_x, valid_data_y)
            x_test, y_test = generator.generate_batch(training_batch_size, test_data_x, test_data_y)

            if i % 100 == 0:
                print("step:", i, "train:",
                      sess.run([model.loss], feed_dict={model.x: x, model.y: y, model.keep_prob: 1}))
                # train_accuracy = sess.run(accuracy, feed_dict={xs: x, ys: y})
                valid_x, valid_y = generator.generate_batch(valid_batch_size, train_data_x, train_data_y)
                print("step:", i, "valid:",
                      sess.run([model.loss], feed_dict={model.x: valid_x, model.y: valid_y, model.keep_prob: 1}))
                # valid_accuracy = sess.run(accuracy, feed_dict={xs: valid_x, ys: valid_y})
                # print("step %d, training accuracy %g" % (i, train_accuracy))
                # print("step %d, valid accuracy %g" % (i, valid_accuracy))
                #
                # y_label_result, y_true_result = sess.run([y_label, y_true], feed_dict={xs: valid_x, ys: valid_y})
                # print("f1_score", sk.metrics.f1_score(y_label_result, y_true_result, average = "weighted"))
                # exit(0)
                # print(y_label)
                # print(_index)

                saver.save(sess, "./xkf_text_cnn_model/cnn", global_step=i)

            _, summary = sess.run([model.train_op, merged], feed_dict={model.x: x, model.y: y, model.keep_prob: 1})
            writer.add_summary(summary, i)
            _, summary = sess.run([model.train_op, merged],
                                  feed_dict={model.x: x_valid, model.y: y_valid, model.keep_prob: 1})
            writer.add_summary(summary, i)
            _, summary = sess.run([model.train_op, merged],
                                  feed_dict={model.x: x_test, model.y: y_test, model.keep_prob: 1})
            writer.add_summary(summary, i)
