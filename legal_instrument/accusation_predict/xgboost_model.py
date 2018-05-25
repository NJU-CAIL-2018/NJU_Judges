import pickle

import xgboost as xgb

import legal_instrument.data_util.generate_batch as generator
import legal_instrument.system_path as constant

# param
training_batch_size = 1024
valid_batch_size = 5096
embedding_size = 128
label_size = 1
#

print("reading data from training set...")
try:
    with open('./dump_data/xgboost/dump_train_x.txt', 'rb') as f:
        train_data_x = pickle.load(f)

    with open('./dump_data/xgboost/dump_train_y_label.txt', 'rb') as f:
        train_data_y = pickle.load(f)

    with open('./dump_data/xgboost/dump_valid_x.txt', 'rb') as f:
        valid_data_x = pickle.load(f)

    with open('./dump_data/xgboost/dump_valid_y_label.txt', 'rb') as f:
        valid_data_y = pickle.load(f)
except:
    print("No dump file read original file! Please wait... "
          "If u want to accelerate this process, please see read_me -> transform_data_to_feature_and_dump")
    accu_dict, reverse_accu_dict = generator.read_accu()
    word_dict, embedding, reverse_dictionary = generator.get_dictionary_and_embedding()

    train_data_x, train_data_y = generator.read_data_in_accu_format(constant.DATA_TRAIN, embedding,
                                                                    word_dict, accu_dict, one_hot=False)
    valid_data_x, valid_data_y = generator.read_data_in_accu_format(constant.DATA_VALID, embedding,
                                                                    word_dict, accu_dict, one_hot=False)

print("reading complete!")

# just test generate_accu_batch
train_data_x_for_validate, train_data_y_for_validate = generator.generate_batch(valid_batch_size, train_data_x, train_data_y)

print("data load complete")
print("The model begin here")

clf = xgb.XGBClassifier(learning_rate=0.05, objective='multi:softmax',
                        n_estimators=100, max_depth=4, reg_alpha = 0.2, min_child_weight=3)

print(valid_data_y.shape)
# try to load model
# try:
#     boost = xgb.Booster()
#     boost.load_model('./xgboost_model/1.model')
#     clf.booster = boost
# except:
#     print("No model to read")

# training begin here!
train_data_y = train_data_y.reshape([len(train_data_y)])
valid_data_y = valid_data_y.reshape([valid_data_y.shape[0]])
train_data_y_for_validate = train_data_y_for_validate.reshape(valid_batch_size)
clf.fit(train_data_x, train_data_y,
        eval_set=[(train_data_x_for_validate, train_data_y_for_validate),
                  (valid_data_x, valid_data_y)], eval_metric='merror', verbose=True)

clf.get_booster().save_model('./xgboost_model/1.model')

evals_result = clf.evals_result()

# visualize
try:
    import pandas as pd
    import matplotlib.pylab as plt

    feat_imp = pd.Series(clf.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
except:
    print("please install pandas and matplotlib!")

print(evals_result)
