import pickle

import numpy as np

import legal_instrument.data_util.generate_batch as generator
import legal_instrument.system_path as constant


def dump_data_for_nn():
    word_dict, embedding, reverse_dictionary = generator.get_dictionary_and_embedding()

    print("reading data from training set...")
    train_data_x, train_data_y = generator.read_data_in_imprisonment_format(constant.DATA_TRAIN, embedding,
                                                                    word_dict)
    valid_data_x, valid_data_y = generator.read_data_in_imprisonment_format(constant.DATA_VALID, embedding,
                                                                    word_dict)
    print("reading complete!")

    # 随机打乱数据
    permutation_for_train = np.random.permutation(train_data_x.shape[0])
    train_data_x = train_data_x[permutation_for_train, :]
    train_data_y = train_data_y[permutation_for_train]

    permutation_for_valid = np.random.permutation(valid_data_y.shape[0])
    valid_data_x = valid_data_x[permutation_for_valid, :]
    valid_data_y = valid_data_y[permutation_for_valid, :]


    with open('./dump_data/nn/dump_train_x.txt', 'wb') as f:
        pickle.dump(train_data_x, f)

    with open('./dump_data/nn/dump_train_y_label.txt', 'wb') as f:
        pickle.dump(train_data_y, f)

    with open('./dump_data/nn/dump_valid_x.txt', 'wb') as f:
        pickle.dump(valid_data_x, f)

    with open('./dump_data/nn/dump_valid_y_label.txt', 'wb') as f:
        pickle.dump(valid_data_y, f)

    print("dump complete!")


def dump_data_for_xgboost():
    word_dict, embedding, reverse_dictionary = generator.get_dictionary_and_embedding()

    print("reading data from training set...")
    train_data_x, train_data_y = generator.read_data_in_imprisonment_format(constant.DATA_TRAIN, embedding,
                                                                            word_dict)
    valid_data_x, valid_data_y = generator.read_data_in_imprisonment_format(constant.DATA_VALID, embedding,
                                                                            word_dict)
    print("reading complete!")

    # 随机打乱数据
    permutation_for_train = np.random.permutation(train_data_x.shape[0])
    train_data_x = train_data_x[permutation_for_train, :]
    train_data_y = train_data_y[permutation_for_train]

    permutation_for_valid = np.random.permutation(valid_data_y.shape[0])
    valid_data_x = valid_data_x[permutation_for_valid, :]
    valid_data_y = valid_data_y[permutation_for_valid]


    with open('./dump_data/xgboost/dump_train_x.txt', 'wb') as f:
        pickle.dump(train_data_x, f)

    with open('./dump_data/xgboost/dump_train_y_label.txt', 'wb') as f:

        pickle.dump(train_data_y, f)

    with open('./dump_data/xgboost/dump_valid_x.txt', 'wb') as f:
        pickle.dump(valid_data_x, f)

    with open('./dump_data/xgboost/dump_valid_y_label.txt', 'wb') as f:
        pickle.dump(valid_data_y, f)

    print("dump complete!")

dump_data_for_nn()
#dump_data_for_xgboost()