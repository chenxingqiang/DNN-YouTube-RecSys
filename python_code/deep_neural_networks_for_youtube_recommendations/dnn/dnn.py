# -*- coding: utf-8 -*-

import os
import json
import tensorflow as tf

def parse_fn(example):
    example_fmt = {
        "visit_items_index": tf.FixedLenFeature([5], tf.int64),
        "continuous_features_value": tf.FixedLenFeature([16], tf.float32),
        "next_visit_item_index": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    next_visit_item_index = parsed.pop("next_visit_item_index")
    return parsed, next_visit_item_index

def input_fn(path, parallel_num, epoch_num, batch_size):
    files = tf.data.Dataset.list_files(path, shuffle=True)
    data_set = files.apply(
        tf.contrib.data.parallel_interleave(
            map_func=lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=parallel_num))
    data_set = data_set.repeat(epoch_num)
    data_set = data_set.map(map_func=parse_fn, num_parallel_calls=parallel_num)
    data_set = data_set.prefetch(buffer_size=256)
    data_set = data_set.batch(batch_size=batch_size)
    return data_set

def model_fn(features, labels, mode, params, config):

    visit_items_index = features["visit_items_index"]    # num * 5
    continuous_features_value = features["continuous_features_value"]  # num * 16
    next_visit_item_index = labels    # num
    keep_prob = params["keep_prob"]
    embedding_size = params["embedding_size"]
    item_num = params["item_num"]
    learning_rate = params["learning_rate"]
    top_k = params["top_k"]

    # items embedding 初始化
    initializer = tf.initializers.random_uniform(minval=-0.5 / embedding_size, maxval=0.5 / embedding_size)
    partitioner = tf.fixed_size_partitioner(num_shards=embedding_size)
    item_embedding = tf.get_variable("item_embedding", [item_num, embedding_size],
                                     tf.float32, initializer=initializer, partitioner=partitioner)

    visit_items_embedding = tf.nn.embedding_lookup(item_embedding, visit_items_index)       # num * 5 * embedding_size
    visit_items_average_embedding = tf.reduce_mean(visit_items_embedding, axis=1)     # num * embedding_size
    input_embedding = tf.concat([visit_items_average_embedding, continuous_features_value], 1)   # num * (embedding_size + 16)
    kernel_initializer_1 = tf.initializers.random_normal(mean=0.0, stddev=0.1)
    bias_initializer_1 = tf.initializers.random_normal(mean=0.0, stddev=0.1)
    layer_1 = tf.layers.dense(input_embedding, 64, activation=tf.nn.relu,
                              kernel_initializer=kernel_initializer_1,
                              bias_initializer=bias_initializer_1, name="layer_1")
    layer_dropout_1 = tf.nn.dropout(layer_1, keep_prob=keep_prob, name="layer_dropout_1")
    kernel_initializer_2 = tf.initializers.random_normal(mean=0.0, stddev=0.1)
    bias_initializer_2 = tf.initializers.random_normal(mean=0.0, stddev=0.1)
    layer_2 = tf.layers.dense(layer_dropout_1, 32, activation=tf.nn.relu,
                              kernel_initializer=kernel_initializer_2,
                              bias_initializer=bias_initializer_2, name="layer_2")
    layer_dropout_2 = tf.nn.dropout(layer_2, keep_prob=keep_prob, name="layer_dropout_2")
    # user vector, num * embedding_size
    kernel_initializer_3 = tf.initializers.random_normal(mean=0.0, stddev=0.1)
    bias_initializer_3 = tf.initializers.random_normal(mean=0.0, stddev=0.1)
    user_vector = tf.layers.dense(layer_dropout_2, embedding_size, activation=tf.nn.relu,
                                  kernel_initializer=kernel_initializer_3,
                                  bias_initializer=bias_initializer_3, name="user_vector")

    if mode == tf.estimator.ModeKeys.TRAIN:
        # 训练
        output_embedding = tf.nn.embedding_lookup(item_embedding, next_visit_item_index)  # num * embedding_size
        logits = tf.matmul(user_vector, output_embedding, transpose_a=False, transpose_b=True)  # num * num
        yhat = tf.nn.softmax(logits)  # num * num
        cross_entropy = tf.reduce_mean(-tf.log(tf.matrix_diag_part(yhat) + 1e-16))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(cross_entropy, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=cross_entropy, train_op=train)

    if mode == tf.estimator.ModeKeys.EVAL:
        # 评估
        output_embedding = tf.nn.embedding_lookup(item_embedding, next_visit_item_index)  # num * embedding_size
        logits = tf.matmul(user_vector, output_embedding, transpose_a=False, transpose_b=True)  # num * num
        yhat = tf.nn.softmax(logits)  # num * num
        cross_entropy = tf.reduce_mean(-tf.log(tf.matrix_diag_part(yhat) + 1e-16))
        return tf.estimator.EstimatorSpec(mode, loss=cross_entropy)

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits_predict = tf.matmul(user_vector, item_embedding, transpose_a=False, transpose_b=True)  # num *  item_num
        yhat_predict = tf.nn.softmax(logits_predict)  # num *  item_num
        _, indices = tf.nn.top_k(yhat_predict, k=top_k, sorted=True)
        index = tf.identity(indices, name="index")  # num * top_k
        # 预测
        predictions = {
            "user_vector": user_vector,
            "index": index
        }
        export_outputs = {
            "prediction": tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

def build_estimator():
    params = {"keep_prob": 0.5, "embedding_size": 16, "item_num": 500, "learning_rate": 0.05, "top_k": 2}
    session_config = tf.ConfigProto(device_count={"CPU": 1}, allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(
        model_dir="D:\Pycharm\PycharmProjects\deep_neural_networks_for_youtube_recommendations\dnn\ckpt",
        tf_random_seed=2019,
        save_checkpoints_steps=100,
        session_config=session_config,
        keep_checkpoint_max=5,
        log_step_count_steps=100
    )
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=params)
    return estimator

def set_dist_env():
    if FLAGS.is_distributed:
        ps_hosts = FLAGS.strps_hosts.split(",")
        worker_hosts = FLAGS.strwork_hosts.split(",")
        job_name = FLAGS.job_name
        task_index = FLAGS.task_index
        chief_hosts = worker_hosts[0:1]  # get first worker as chief
        worker_hosts = worker_hosts[2:]  # the rest as worker

        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
                     'task': {'type': job_name, 'index': task_index}}
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

def train_eval_save():

    set_dist_env()

    estimator = build_estimator()

    # 训练
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(
            path='D:\Pycharm\PycharmProjects\deep_neural_networks_for_youtube_recommendations\dnn\\tfrecords\\train\\train.tfrecords',
            parallel_num=32,
            epoch_num=11,
            batch_size=32),
        max_steps=1600
    )
    # 评估
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(
            path='D:\Pycharm\PycharmProjects\deep_neural_networks_for_youtube_recommendations\dnn\\tfrecords\evaluation\evaluation.tfrecords',
            parallel_num=32,
            epoch_num=1,
            batch_size=32),
        steps=15,     # 验证集评估多少批数据
        start_delay_secs=1,    # 在多少秒后开始评估
        throttle_secs=20  # evaluate every 20seconds
    )
    # 训练和评估
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # 模型保存
    features_spec = {
        "visit_items_index": tf.placeholder(tf.int64, shape=[None, 5], name="visit_items_index"),
        "continuous_features_value": tf.placeholder(tf.float32, shape=[None, 16], name="continuous_features_value")
    }
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features_spec)
    estimator.export_savedmodel(
        "D:\Pycharm\PycharmProjects\deep_neural_networks_for_youtube_recommendations\dnn\modelpath",
        serving_input_receiver_fn)

def main(_):
    train_eval_save()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_boolean("is_distributed", False, "是否分布式训练")
    tf.app.flags.DEFINE_string("strps_hosts", "localhost:2000", "参数服务器")
    tf.app.flags.DEFINE_string("strwork_hosts", "localhost:2100,localhost:2200,localhost:2300,localhost:2400", "工作服务器")
    tf.app.flags.DEFINE_string("job_name", "ps", "参数服务器或者工作服务器")
    tf.app.flags.DEFINE_integer("task_index", 0, "job的task索引")
    tf.app.run(main=main)