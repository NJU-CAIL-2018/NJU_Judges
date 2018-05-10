下面说明包中文件和文件夹内容:

dump_data文件夹：用于保存生成的词向量和词典

logs文件夹：用于保存 tensorflow 训练的可视化文件结果

modelStore文件夹：用于保存 tensorflow 生成的模型。
注意：如果要训练新的模型，请务必删除旧模型的存储点，该存储点的作用是迭代，意思就是这次训练到10000次，下次接着这次训练

generate_batch文件：用于将文本数据转化为向量作为模型的输入，下面说明其主要函数作用：

##不重要
1.read_accu : 生成罪名与罪名index的map和reverse_map，返回值为：
accu_dict : 罪名 -> index  eg. 故意杀人 ->17
reverse_accu_dict : index -> 罪名  eg. 17 -> 故意杀人

##不重要
2.get_dictionary_and_embedding : 加载词典和词向量，返回值为：
word_dictionary : 词典  单词 -> index
embedding : 词向量   index -> vector
reverse_word_dictionary : 反向词典  index -> 单词

##比较重要
3.read_data : 加载数据集
参数：
file_name : 数据文件名, 如DATA_TRAIN, DATA_VALID
accu_size : 罪名字典大小
embedding : 词向量
dictionary : 词典
accu_dict : 罪名词典
返回值：
data_x : 数据集的特征向量
data_y : 数据集的标签，one_hot形式

##重要
4.generate_accu_batch ： 生成预测罪名的特征向量和标签
参数:
batch_size : 每次生成的数据数量
data_x : 数据集的特征向量
data_y : 数据集的标签
accu_size : 罪名字典大小
返回值:
x : 特征向量 shape = [batch_size, 词向量长度]
y : 标签 shape = [batch_size, 罪名词典长度]

nn_baseline文件 : 一个以神经网络作为基线的模型，10000迭代后准确度为0.58  验证集准确度为0.56 目前为欠拟合状态
强烈建议不要修改此文件，新建一个model文件进行修改，因为此文件为基线模型
可以按照文件中的顺序编写自己的模型，即：
读取数据 -> 建立模型 -> 训练 -> 评价
其中涉及到模型的保存和可视化，可以直接挪用到更好的模型中

word2vec文件 : 用于训练词向量的模型，可以不看


重要说明：
请在根目录下建一个名为system_path.py的文件，内容为
DATA_TRAIN = '/Users/SilverNarcissus/Documents/法院文书/good/data_train.json'
DATA_VALID = '/Users/SilverNarcissus/Documents/法院文书/good/data_valid.json'
DATA_TEST = '/Users/SilverNarcissus/Documents/法院文书/good/data_test.json'
FONT_FILE = '"/System/Library/Fonts/PingFang.ttc"'
ACCU_FILE = '/Users/SilverNarcissus/Documents/法院文书/good/accu.txt'
LAW_FILE = '/Users/SilverNarcissus/Documents/法院文书/good/law.txt'
将后面的路径换成自己的文件路径，其中FONT_FILE应该不用修改
