# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import initializers

class SampleLayer(tf.keras.layers.Layer):
    def __init__(self, is_training, top_k, item_num,
                 kernel_initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1), **kwargs):
        self.is_training = is_training
        self.top_k = top_k
        self.item_num = item_num
        self.kernel_initializer = kernel_initializer
        super(SampleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        input_shape0 = input_shape[0]
        # 为该层创建一个可训练的权重
        partitioner = tf.compat.v1.fixed_size_partitioner(num_shards=int(input_shape0[1]))
        self.kernel = self.add_weight(name="item_embedding",
                                      shape=(self.item_num, int(input_shape0[1])),
                                      initializer=self.kernel_initializer,
                                      trainable=True,
                                      partitioner=partitioner)
        # 一定要在最后调用它
        super(SampleLayer, self).build(input_shape)

    def train_output(self, inputs0, inputs1):
        output_embedding = tf.nn.embedding_lookup(self.kernel, inputs1)  # num * embedding_size
        logits = tf.matmul(inputs0, output_embedding, transpose_a=False, transpose_b=True)  # num * num
        yhat = tf.nn.softmax(logits)  # num * num
        cross_entropy = tf.reduce_mean(-tf.log(tf.matrix_diag_part(yhat) + 1e-16))
        return cross_entropy

    def predict_output(self, inputs0):
        logits_predict = tf.matmul(inputs0, self.kernel, transpose_a=False, transpose_b=True)  # num * item_num
        yhat_predict = tf.nn.softmax(logits_predict)  # num * item_num
        _, indices = tf.nn.top_k(yhat_predict, k=self.top_k, sorted=True)  # indices是: num * top_k
        indices = tf.cast(indices, tf.float32)  # tf.keras.backend.switch输出类型必须一样, 所以将int转为float
        return indices

    def func1(self, inputs):
        assert len(inputs) == 2
        inputs1 = tf.cast(inputs[1], tf.int32)
        return inputs1

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        inputs0 = inputs[0]  # 上一层的输出
        inputs1_default = tf.zeros([inputs0.shape[0]], dtype=tf.int32)  # 另外一个输入, 这是默认值
        inputs1 = tf.cond(self.is_training, lambda: self.func1(inputs), lambda: inputs1_default)
        # 如果训练的话, 输出是损失值; 如果预测的话, 输出是相似的top_k索引
        train_predict_output = tf.cond(self.is_training, lambda: self.train_output(inputs0, inputs1),
                                       lambda: self.predict_output(inputs0))
        return train_predict_output

    def func2(self, input_shape):
        input_shape0 = input_shape[0]
        return (input_shape0[0], self.top_k)

    def compute_output_shape(self, input_shape):
        output_shape = tf.cond(self.is_training, lambda: (), lambda: self.func2(input_shape))
        return output_shape

    def get_config(self):
        config = {
            'is_training': self.is_training,
            'top_k': self.top_k,
            'item_num': self.item_num,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(SampleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    inputs0 = tf.constant([[0.1, 0.2, 0.6, 0.3, 0.5], [0.8, 0.6, 0.9, 0.3, 0.5]])
    inputs1 = tf.constant([0, 3])
    sample_layer = SampleLayer(tf.constant(True), 3, 10, name="abc")
    result = sample_layer([inputs0, inputs1])
    print(result)
    print(sample_layer.trainable_weights)