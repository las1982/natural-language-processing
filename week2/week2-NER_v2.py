from utils import *
from BiLSTMModel import *


tf.reset_default_graph()
s = "hpe"
x_train, y_train = read_data('data/'+ s + '/train.txt')
x_val, y_val = read_data('data/'+ s + '/validation.txt')
x_test, y_test = read_data('data/'+ s + '/test.txt')

x_special = ['<UNK>', '<PAD>']
y_special = ['O']

# Create dictionaries
token2idx, idx2token = build_dict(x_train, x_special)
tag2idx, idx2tag = build_dict(y_train, y_special)
helper = Helper(token2idx, idx2token, tag2idx, idx2tag)

model = BiLSTMModel(
    vocabulary_size=len(token2idx),
    n_tags=14,
    embedding_dim=300,
    n_hidden_rnn=200,
    PAD_index=token2idx["<PAD>"]
)

batch_size = 16
n_epochs = 5

learning_rate = 0.005
learning_rate_decay = 1.11
dropout_keep_probability = 0.7

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Start training... \n')
for epoch in range(n_epochs):
    # For each epoch evaluate the model on train and validation data
    print('-' * 20 + ' Epoch {} '.format(epoch + 1) + 'of {} '.format(n_epochs) + '-' * 20)
    print('Train data evaluation:')
    helper.eval_conll(model, sess, x_train, y_train, short_report=True)
    print('Validation data evaluation:')
    helper.eval_conll(model, sess, x_val, y_val, short_report=True)

    # Train the model
    for x_batch, y_batch, lengths in helper.batches_generator(batch_size, x_train, y_train):
        model.train_on_batch(sess, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability)

    # Decaying the learning rate
    learning_rate = learning_rate / learning_rate_decay

print('...training finished.')

print('-' * 20 + ' Train set quality: ' + '-' * 20)
train_results = helper.eval_conll(model, sess, x_train, y_train, short_report=False)

print('-' * 20 + ' Validation set quality: ' + '-' * 20)
validation_results = helper.eval_conll(model, sess, x_val, y_val, short_report=False)

print('-' * 20 + ' Test set quality: ' + '-' * 20)
test_results = helper.eval_conll(model, sess, x_test, y_test, short_report=False)
