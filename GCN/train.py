'''
Created on 25-Apr-2018

@author: agarr
'''

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import *
from models import GCN

if __name__ == '__main__':
    pass

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("cora")

features = preprocess_features(features)

support = [preprocess_adj(adj)]
num_supports = 1
model_func = GCN

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

model = model_func(placeholders, input_dim=features[2][1], logging=True)

sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


sess.run(tf.global_variables_initializer())

cost_val = []
accuracy = []
loss = []
accuracy_val = []

# Train model
for epoch in range(200):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: 0.5})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    
    accuracy.append(outs[2])
    loss.append(outs[1])
    accuracy_val.append(acc)
    
    cost_val.append(cost)
    
    
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    
    
plt.plot(accuracy)
plt.plot(accuracy_val)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
 
#-- summarize history for loss
plt.plot(loss)
plt.plot(cost_val)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()   
