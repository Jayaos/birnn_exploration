import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import random
import os

class BIRNNLS(tf.keras.Model):
    def __init__(self, config):
        super(BIRNNLS, self).__init__()
        self.vocabsize = config["vocabsize"]
        self.embedding_dim = config["embedding_dim"]
        self.optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
        self.unit_dim = config["unit_dim"]
        self.masking_layer = tf.keras.layers.Masking(mask_value=0.0)
        self.lstm_backward = tf.keras.layers.LSTM(self.unit_dim, go_backwards=True)
        self.lstm_forward = tf.keras.layers.LSTM(self.unit_dim, return_sequences=True)
        self.dropout = tf.keras.layers.Dropout(config["dropout_rate"])
        self.prediction = tf.keras.layers.Dense(config["vocabsize"], activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.001))

    def initParams(self):
        print("use randomly initialzed value...")
        self.embedding = tf.Variable(tf.random.normal([self.vocabsize, self.embedding_dim], 0, 0.01))      
   
    def loadParams(self, pretrained_emb):
        print("use pre-trained embeddings...")
        self.embedding = tf.Variable([pretrained_emb])

    def call(self, x, training):
        x = tf.matmul(x, self.embedding)
        x = tf.math.tanh(x)
        x = self.masking_layer(x)
        x = self.dropout(x, training=training)
        x_forward = self.lstm_forward(x)
        x_backward = self.lstm_backward(x)
        x_backward = tf.reshape(x_backward, shape=[x_backward.shape[0], 1, x_backward.shape[1]])
        x_backward = tf.tile(x_backward, [1, x_forward.shape[1], 1])

        return tf.concat([x_forward, x_backward], axis=-1)

def calculate_loss(model, x, y, mask, logEps=1e-8):
    """calculate the sequential cross-entropy loss"""

    output_seqs = model(x, training=True) # (batch_size, max_seq, unit_dim)
    y_hat = model.prediction(output_seqs)
    y_hat = tf.clip_by_value(y_hat, 1e-8, 1-(1e-8))
    cross_entropy = tf.reduce_sum(tf.negative(y * tf.math.log(y_hat) + (1 -  y) * tf.math.log(1 - y_hat)), axis=-1)
    cross_entropy = tf.multiply(cross_entropy, mask)

    return tf.reduce_mean(cross_entropy)

def train_birnnls(patient_record_path, output_path, vocabsize, test_size, validate_size, epochs, 
                batch_size, learning_rate, dropout_rate, unit_dim, embedding_dim, num_layers, k, set_seed, pretrained_embedding):

    config = locals().copy()

    print("load data...")
    patient_record = load_data(patient_record_path)
    train_x, test_x, valid_x = train_test_validate_split(patient_record, test_size=test_size, validate_size=validate_size, seed=set_seed)

    print("build and initialize model...")
    birnnls = BIRNNLS(config)

    if pretrained_embedding != None:
        birnnls.loadParams(np.load(pretrained_embedding))
    else:
        birnnls.initParams()

    print("start training...")
    best_epoch = 0
    best_recall = 0
    best_model = None
    training_loss = []
    training_recall_record = []
    validation_recall_record = []
    batch_num = int(np.ceil(len(train_x) / batch_size))

    for epoch in range(epochs): 
        progbar = tf.keras.utils.Progbar(batch_num)
        loss_record = []

        for i in random.sample(range(batch_num), batch_num): # shuffling the data 
            x_batch = train_x[batch_size * i:batch_size * (i+1)]
            x, y, mask = pad_matrix(x_batch, config["vocabsize"])

            with tf.GradientTape() as tape:
                    loss = calculate_loss(birnnls, x, y, mask)
            gradients = tape.gradient(loss, birnnls.trainable_variables)
            birnnls.optimizer.apply_gradients(zip(gradients, birnnls.trainable_variables))
            loss_record.append(loss.numpy())        
            progbar.add(1)

        print('epoch:{e}, mean loss:{l:.6f}'.format(e=epoch+1, l=np.mean(loss_record)))
        training_loss.append(np.mean(loss_record))
        training_recall = calculate_topk_recall(birnnls, train_x, config["vocabsize"], config["batch_size"], k)
        validation_recall = calculate_topk_recall(birnnls, valid_x, config["vocabsize"], config["batch_size"], k)
        print('epoch:{e}, training recall at {k}:{l:.6f}'.format(e=epoch+1, k=k, l=training_recall))
        training_recall_record.append(training_recall)        
        print('epoch:{e}, validation recall at {k}:{l:.6f}'.format(e=epoch+1, k=k, l=validation_recall))
        validation_recall_record.append(validation_recall)

        if validation_recall > best_recall:
            best_recall = validation_recall
            best_model = birnnls.get_weights()
            best_epoch = epoch + 1

    print("saving training record...")
    if pretrained_embedding != None:
        mode = "pretrained"
    else:
        mode = "emb"

    print("saving the loss and recall...")
    save_data(os.path.join(output_path, "training_loss_birnnls{l}_{m}_e{e}_lr{lr}_b{b}_do{do}_d{dim}_k{k}.pkl".format(l=num_layers, m=mode, 
    e=epochs, lr=learning_rate, b=batch_size, do=dropout_rate, dim=unit_dim, k=k)), training_loss)
    save_data(os.path.join(output_path, "training_recall_birnnls{l}_{m}_e{e}_lr{lr}_b{b}_do{do}_d{dim}_k{k}.pkl".format(l=num_layers, m=mode, 
    e=epochs, lr=learning_rate, b=batch_size, do=dropout_rate, dim=unit_dim, k=k)), training_recall_record)
    save_data(os.path.join(output_path, "validation_recall_birnnls{l}_{m}_e{e}_lr{lr}_b{b}_do{do}_d{dim}_k{k}.pkl".format(l=num_layers, m=mode, 
    e=epochs, lr=learning_rate, b=batch_size, do=dropout_rate, dim=unit_dim, k=k)), validation_recall_record)

    print('Best model: at epoch {e}'.format(e=best_epoch))
    birnnls.set_weights(best_model)

    print("saving the best model...")
    np.save(os.path.join(output_path, "best_model_birnnls{l}_{m}_e{e}_lr{lr}_b{b}_do{do}_d{dim}_k{k}.npy".format(l=num_layers, m=mode, 
    e=epochs, lr=learning_rate, b=batch_size, do=dropout_rate, dim=unit_dim, k=k)), best_model)
    save_data(os.path.join(output_path, "config_birnnls{l}_{m}_e{e}_lr{lr}_b{b}_do{do}_d{dim}_k{k}.pkl".format(l=num_layers, m=mode, 
    e=epochs, lr=learning_rate, b=batch_size, do=dropout_rate, dim=unit_dim, k=k)), config)

    print("calculate evaluation metrics for the best model on the test set...")
    avg_topk_recall = calculate_topk_recall(birnnls, test_x, config["vocabsize"], config["batch_size"], k)
    print('average top-k recall of the best model on the test set:{r:.6f}'.format(r=avg_topk_recall))
    save_data(os.path.join(output_path, "test_recall_birnnls{l}_{m}_e{e}_lr{lr}_b{b}_do{do}_d{dim}_k{k}.pkl".format(l=num_layers, m=mode, 
    e=epochs, lr=learning_rate, b=batch_size, do=dropout_rate, dim=unit_dim, k=k)), avg_topk_recall)

def pad_matrix(records, vocabsize):
    lengths = np.array([len(recs) for recs in records]) - 1
    max_len = np.max(lengths)
    n_patient = len(records)
    num_class = vocabsize

    x = np.zeros(shape=(n_patient, max_len, num_class))
    y = np.zeros(shape=(n_patient, max_len, num_class))
    mask = np.zeros(shape=(n_patient, max_len))

    for idx, recs in enumerate(records):
        for xvec, xrec in zip(x[idx, :, :], recs[:-1]):
            xvec[xrec] = 1.
        for yvec, yrec in zip(y[idx, :, :], recs[1:]):
            yvec[yrec] = 1.
        mask[idx, lengths[idx]-1] = 1. # only the last visit for each patient is marked with 1

    return x, y, mask

def train_test_validate_split(mydata, test_size, validate_size, seed):
    test_validate_size = test_size + validate_size
    train_set, test_validate_set = train_test_split(mydata, train_size=1-test_validate_size, random_state=seed)
    test_size = test_size / test_validate_size
    test_set, validate_set = train_test_split(test_validate_set, train_size=test_size, random_state=seed)

    return train_set, test_set, validate_set

def load_data(data_path):
    my_data = pickle.load(open(data_path, 'rb'))

    return my_data

def save_data(output_path, mydata):
    with open(output_path, 'wb') as f:
        pickle.dump(mydata, f)

def calculate_topk_recall_on_batch(y_hat, y, mask, k): 
    """ 
    --y_hat: (batch_size, max_seq, vocabsize)
    --y: (batch_size, max_seq, vocabsize)
    --k: the number of concept to recall
    --mask: (n_patients, max_len)
    """

    y_hat = np.reshape(y_hat, (-1, y_hat.shape[-1])) # (batch_size * max_seq, vocabsize)
    y = np.reshape(y, (-1, y.shape[-1])) # (batch_size * max_seq, vocabsize)
    mask = np.where(np.reshape(mask, -1) != 0)[0]
    y_hat = y_hat[mask, :]
    y = y[mask, :]

    topk_concepts = np.argsort(y_hat, )[:, -k:]

    recall_list = []
    for i in range(y.shape[0]):
        recall_concepts = set(list(topk_concepts[i, ]))
        true_concepts = set(np.where(y[i,] != 0)[0])
        tp = len(set.intersection(recall_concepts, true_concepts))
        recall_list.append(tp / len(true_concepts))

    return np.average(recall_list)

def calculate_topk_recall(model, test_x, vocabsize, batch_size, k):

    batch_num = int(np.ceil(len(test_x) / batch_size))
    topk_recall_list = []

    for i in range(batch_num):
        x_batch = test_x[batch_size * i:batch_size * (i+1)]
        x, y, mask = pad_matrix(x_batch, vocabsize)
        output_seqs = model(x) # (batch_size, max_seq, embedding_dim)
        y_hat = model.prediction(output_seqs)
        topk_recall = calculate_topk_recall_on_batch(y_hat, y, mask, k)
        topk_recall_list.append(topk_recall)
    
    return np.average(topk_recall_list)