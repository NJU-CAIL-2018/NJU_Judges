import legal_instrument.generate_batch as generator
import xgboost as xgb
import legal_instrument.system_path as constant
import pickle

# param
training_batch_size = 1024
valid_batch_size = 1024
embedding_size = 128
label_size = 1
#

accu_dict, reverse_accu_dict = generator.read_accu()
word_dict, embedding, reverse_dictionary = generator.get_dictionary_and_embedding()

print("reading data from training set...")
train_data_x, train_data_y = generator.read_data_in_accu_format(constant.DATA_TRAIN, len(accu_dict), embedding,
                                                                word_dict, accu_dict, one_hot=False)
valid_data_x, valid_data_y = generator.read_data_in_accu_format(constant.DATA_VALID, len(accu_dict), embedding,
                                                                word_dict, accu_dict, one_hot=False)
print("reading complete!")

with open('./dump_data/dump_train_x.txt', 'wb') as f:
    pickle.dump(train_data_x, f)

with open('./dump_data/dump_train_y.txt', 'wb') as f:
    pickle.dump(train_data_y, f)

exit(0)
with open('dump_embedding.txt', 'rb') as f:
    train_data_x = pickle.load(f)

# just test generate_accu_batch
x, y = generator.generate_batch(training_batch_size, train_data_x, train_data_y, label_size)

print("data load complete")
print("The model begin here")

clf = xgb.XGBClassifier(learning_rate=0.08, objective='multi:softmax', n_estimators=100, max_depth=6)

# try to load model
# try:
#     boost = xgb.Booster()
#     boost.load_model('./xgboost_model/1.model')
#     clf.booster = boost
# except:
#     print("No model to read")

# training begin here!
train_data_y = train_data_y.reshape([len(train_data_y)])
valid_data_y = valid_data_y.reshape([len(valid_data_y)])
clf.fit(train_data_x, train_data_y,
        eval_set=[(train_data_x, train_data_y)], eval_metric='merror', verbose=True)

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
