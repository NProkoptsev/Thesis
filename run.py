import argparse
import os
import pickle
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import sparse

import sklearn
import tensorflow as tf
from utils.graph import get_chebyshev_polynomials, get_renormalized_adj_matrix
from models.gcgru_model import gcgru_model
from models.indrnn_model import indrnn_model
from models.lstm_model import lstm_model
from models.var_model import var_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lr_decay_fn(lr, step):
    step = tf.cast(step, tf.float32)
    step = tf.maximum(step - 4000, 0.)
    ticks = step // 1000
    return lr / (np.sqrt(10)**ticks)


def model_fn(features, labels, mode, params, config):
    adj = np.load(params.adj)

    conv_matrix = tf.constant(get_renormalized_adj_matrix(
        adj-np.eye(params.n_nodes)), dtype=tf.float32)
    conv_matrix2 = tf.constant(
        get_chebyshev_polynomials(adj, 2), dtype=tf.float32)
    if params.model_type == 'indrnn':
        predictions = indrnn_model(features, labels, conv_matrix, mode, params)
    if params.model_type == 'gcgru':
        predictions = gcgru_model(features, labels, conv_matrix2, mode, params)
    if params.model_type == 'lstm':
        predictions = lstm_model(features, labels, mode, params)
    if params.model_type == 'var':
        predictions = var_model(features, labels, mode, params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.mean_squared_error(
            labels=labels, predictions=predictions)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_or_create_global_step(),
            learning_rate=params.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            learning_rate_decay_fn=lr_decay_fn,
            clip_gradients=0.25)
        specs = dict(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )
    if mode == tf.estimator.ModeKeys.EVAL:
        mse = tf.metrics.mean_squared_error(
            labels=labels, predictions=predictions)

        # , 'metrics/MAE' : mae, 'metrics/RMSE' : rmse}
        metrics = {'metrics/MSE': mse}
        loss = tf.losses.mean_squared_error(
            labels=labels, predictions=predictions)
        specs = dict(
            mode=mode,
            loss=loss,
            eval_metric_ops=metrics
        )
    if mode == tf.estimator.ModeKeys.PREDICT:
        specs = dict(
            mode=mode,
            predictions=predictions
        )
    return tf.estimator.EstimatorSpec(**specs)


def input_fn(mode,
             dataset,
             batch_size,
             input_steps,
             output_steps,
             sample_buffer_size=1000):
    num_epochs = None if mode == tf.estimator.ModeKeys.TRAIN else 1
    dataset = tf.data.Dataset.from_tensor_slices(dataset.astype(np.float32))
    if mode != tf.estimator.ModeKeys.PREDICT:
        dataset = dataset.apply(
            tf.contrib.data.sliding_window_batch(input_steps + output_steps))
        dataset = dataset.map(lambda x: (x[:input_steps], x[input_steps:]))
        dataset = dataset.shuffle(sample_buffer_size)
    else:
        dataset = dataset.apply(
            tf.contrib.data.sliding_window_batch(input_steps))
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset


def model(mode, model_name, model_type, batch_size, n_steps, eval_secs, n_layers, n_nodes, 
        hidden_size, n_input_steps, n_output_steps, learning_rate, sampling_start):

    config = dict(
        model_name=model_name,
        batch_size=batch_size,
        n_steps=n_steps,
        eval_secs=eval_secs)

    params = dict(
        model_type=model_type,
        n_layers=n_layers,
        n_nodes=n_nodes,
        hidden_size=hidden_size,
        n_input_steps=n_input_steps,
        n_output_steps=n_output_steps,
        learning_rate=learning_rate,
        adj='data/processed/adj_256.npy',
        sampling_start=sampling_start
    )

    hparams = tf.contrib.training.HParams(**params)
    model_dir = os.path.join('output', config['model_name'])

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_summary_steps=10, save_checkpoints_steps=1000,
                                        save_checkpoints_secs=None)
    estimator = tf.estimator.Estimator(
        model_fn, config=run_config, params=hparams)
    if mode == 'train':
        X_train = np.load('data/processed/train_256.npy')
        X_valid = np.load('data/processed/valid_256.npy')

        def train_input_fn(): return input_fn(tf.estimator.ModeKeys.TRAIN, X_train, config['batch_size'],
                                              hparams.n_input_steps, hparams.n_output_steps, 1000)

        def val_input_fn(): return input_fn(tf.estimator.ModeKeys.EVAL, X_valid, config['batch_size']*4,
                                            hparams.n_input_steps, hparams.n_output_steps, 1000)
        train_spec = tf.estimator.TrainSpec(
            train_input_fn, max_steps=config['n_steps'])
        eval_spec = tf.estimator.EvalSpec(
            val_input_fn, steps=None, throttle_secs=config['eval_secs'])

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    else:
        with open('data/processed/std_scaler_256.pkl', 'rb') as file:
            std_scaler = pickle.load(file)
        X_test = np.load('data/processed/test_256.npy')

        def predict_input_fn(): return input_fn(
            tf.estimator.ModeKeys.PREDICT, X_test[:-12], 256, 40, 12, 1000)

        y_true = [[], [], [], []]
        y_pred = [[], [], [], []]
        # , checkpoint_path=ckpt):
        for x in estimator.predict(predict_input_fn):
            y_pred[0].append(std_scaler.inverse_transform(x[0]))
            y_pred[1].append(std_scaler.inverse_transform(x[2]))
            y_pred[2].append(std_scaler.inverse_transform(x[5]))
            y_pred[3].append(std_scaler.inverse_transform(x[11]))

        y_true[0] = [std_scaler.inverse_transform(
            x.flatten()) for x in X_test[40:-11]]
        y_true[1] = [std_scaler.inverse_transform(
            x.flatten()) for x in X_test[42:-9]]
        y_true[2] = [std_scaler.inverse_transform(
            x.flatten()) for x in X_test[45:-6]]
        y_true[3] = [std_scaler.inverse_transform(
            x.flatten()) for x in X_test[51:]]

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        "LSTM"
        print('MSE for 5 15 30 60 mins')
        for i in range(4):
            print(np.sqrt(sklearn.metrics.mean_squared_error(
                y_true[i].reshape(-1, 1), y_pred[i].reshape(-1, 1))))

        print('MAE for 5 15 30 60 mins')
        for i in range(4):
            print(sklearn.metrics.mean_absolute_error(
                y_true[i].reshape(-1, 1), y_pred[i].reshape(-1, 1)))

        print('MAPE for 5 15 30 60 mins')
        for i in range(4):
            print(mean_absolute_percentage_error(
                y_true[i].reshape(-1, 1), y_pred[i].reshape(-1, 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument("mode", help="train or infer")

    parser.add_argument(
        "model_name", help='Name of the model, if there exist checkpoint with this model than restore')

    parser.add_argument(
        "model_type", help='Type of the model: indrnn, gcgru, lstm, var')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--n_steps',
        type=int,
        default=15000
    )

    parser.add_argument(
        '--eval_secs',
        type=int,
        default=600
    )

    parser.add_argument(
        '--n_layers',
        help='Number of layers',
        type=int,
        default=7
    )

    parser.add_argument(
        '--n_nodes',
        help='Number of nodes in data',
        type=int,
        default=256
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=64
    )
    parser.add_argument(
        '--n_input_steps',
        type=int,
        default=40
    )

    parser.add_argument(
        '--n_output_steps',
        type=int,
        default=12
    )

    parser.add_argument(
        '--learning_rate',
        help='initial leraning rate',
        type=float,
        default=1e-3
    )
    parser.add_argument(
        '--sampling_start',
        help='Start scheduled sampling from this iteration',
        type=int,
        default=10000
    )
    args = parser.parse_args()
    arguments = args.__dict__
    print(arguments)
    model(**arguments)
