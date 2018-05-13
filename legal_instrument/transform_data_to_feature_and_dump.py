import legal_instrument.generate_batch as generator
import legal_instrument.system_path as constant
import pickle

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

with open('./dump_data/dump_train_y_label.txt', 'wb') as f:
    pickle.dump(train_data_y, f)

with open('./dump_data/dump_valid_x.txt', 'wb') as f:
    pickle.dump(valid_data_x, f)

with open('./dump_data/dump_valid_y_label.txt', 'wb') as f:
    pickle.dump(valid_data_y, f)

print("dump complete!")