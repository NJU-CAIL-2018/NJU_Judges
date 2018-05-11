import legal_instrument.generate_batch as generator
import xgboost as xgb
import legal_instrument.system_path as constant
#import pandas as pd

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

# just test generate_accu_batch
x, y = generator.generate_batch(training_batch_size, train_data_x, train_data_y, label_size)

print("data load complete")
print("The model begin here")



dtrain = xgb.DMatrix(train_data_x, train_data_y)

clf = xgb.XGBClassifier(learning_rate=0.08, objective='multi:softmax', n_estimators=100, max_depth=6)

# try to load model
try:
    boost = xgb.Booster()
    boost.load_model('./xgboost_model/1.model')
    clf.booster = boost
except:
    print("No model to read")

train_data_y = train_data_y.reshape([len(train_data_y)])
valid_data_y = valid_data_y.reshape([len(valid_data_y)])
clf.fit(train_data_x, train_data_y,
        eval_set=[(train_data_x, train_data_y),
                  (valid_data_x, valid_data_y)], eval_metric='merror', verbose=True)

clf.get_booster().save_model('./xgboost_model/1.model')

evals_result = clf.evals_result()

print(evals_result)
