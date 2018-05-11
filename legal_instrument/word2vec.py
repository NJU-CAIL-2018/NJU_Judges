import json

import collections
import numpy as np
import jieba
import math
import random
import tensorflow as tf
import re
import legal_instrument.system_path as constant
import pickle


##
# param
batch_size = 128
embedding_size = 128
skip_window = 2
num_skips = 1
valid_size = 9  # 切记这个数字要和len(valid_word)对应，要不然会报错哦
valid_window = 100
num_sampled = 64  # Number of negative examples to sample.
vocabulary_size = 10000
##

# global
data_index = 0


##

# get fact from the data
def get_fact():
    # 读取停用词
    stop_words = []
    with open('./dump_data/stop_words.txt', "r", encoding="UTF-8") as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))

    result = []
    with open(constant.DATA_TRAIN, "r", encoding="UTF-8") as f:
        line = f.readline()
        while line:
            obj = json.loads(line)
            raw_words = list(jieba.cut(obj['fact'], cut_all=False))
            for word in raw_words:
                if regular_filter(word):
                    if word not in stop_words:
                        result.append(word)
            result.append('++')
            line = f.readline()

    return result

def regular_filter(word):
    if re.match(r'.*某', word) is not None or re.match(r'某.*', word) is not None or re.match(r'.*县', word) is not None or re.match(r'.*乡', word) is not None:
        return False
    return True



def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    print("count", len(count))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    buffer.append(dictionary['++'])
    while dictionary['++'] in buffer:
        fill_buffer(buffer, span)

    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

        while dictionary['++'] in buffer:
            fill_buffer(buffer, span)

    return batch, labels

# generate next span of buffer
def fill_buffer(buffer, span):
    global data_index
    buffer.clear()
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

# execute part
words = get_fact()
data, count, dictionary, reverse_dictionary = build_dataset(words)

del words  # 删除words节省内存
print('Most common words (+UNK)', count[:20])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

x, y = generate_batch(batch_size, num_skips, skip_window)
for i in range(batch_size):
    print(i)
    print(reverse_dictionary[x[i]], reverse_dictionary[y[i][0]])


# print(get_fact())
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# 验证集
# valid_word = ['萧炎', '灵魂', '火焰', '萧薰儿', '药老', '天阶', "云岚宗", "乌坦城", "惊诧"]
# valid_examples = [dictionary[li] for li in valid_word]
graph = tf.Graph()
with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]), dtype=tf.float32)

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed, labels=train_labels,
                       num_sampled=num_sampled, num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 10000
with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 8000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[:top_k]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    if (nearest[k] in reverse_dictionary):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

    f = open('dump_embedding.txt', 'wb')
    pickle.dump(final_embeddings, f)
    f.close()

    f = open('dump_dict.txt', 'wb')
    pickle.dump(dictionary, f)
    f.close()

#exit(0)
#visualize the result
try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    # 为了在图片上能显示出中文
    # for mac
    font = FontProperties(fname=constant.FONT_FILE, size=14)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 150
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    print("here")
    plt.figure(figsize=(18, 18))
    for i in range(plot_only):
        if (i in reverse_dictionary):
            label = reverse_dictionary[i]
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         fontproperties=font,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
            # plot_with_labels(low_dim_embs, labels)

    #plt.show()
    plt.savefig('tsne.png', dpi=600)
    print("finish")



except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")