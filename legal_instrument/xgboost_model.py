import legal_instrument.generate_batch as generator
import xgboost as xgb
import legal_instrument.system_path as constant

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

y = y.reshape([training_batch_size])
clf.fit(x, y,
        eval_set=[(x, y)], eval_metric='merror', verbose=True)
evals_result = clf.evals_result()

print(evals_result)
