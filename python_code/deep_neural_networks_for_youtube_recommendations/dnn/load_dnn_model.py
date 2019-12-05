# -*- coding: utf-8 -*-

import tensorflow as tf

class dataProcess(object):

    def parse_fn(self, example):
        example_fmt = {
            "visit_items_index": tf.FixedLenFeature([5], tf.int64),
            "continuous_features_value": tf.FixedLenFeature([16], tf.float32),
            "next_visit_item_index": tf.FixedLenFeature([], tf.int64)
        }
        parsed = tf.parse_single_example(example, example_fmt)
        parsed.pop("next_visit_item_index")
        return parsed

    def next_batch(self, batch_size):
        files = tf.data.Dataset.list_files(
            'D:\Pycharm\PycharmProjects\deep_neural_networks_for_youtube_recommendations\dnn\\tfrecords\\train\\train.tfrecords', shuffle=False
        )
        data_set = files.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=16))
        data_set = data_set.map(map_func=self.parse_fn, num_parallel_calls=16)
        data_set = data_set.prefetch(buffer_size=256)
        data_set = data_set.batch(batch_size=batch_size)
        iterator = data_set.make_one_shot_iterator()
        features = iterator.get_next()
        return features

if __name__ == "__main__":
    # 数据预处理#
    dataProcess = dataProcess()
    features = dataProcess.next_batch(batch_size=16)

    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    with tf.Session() as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                                    "D:\Pycharm\PycharmProjects\deep_neural_networks_for_youtube_recommendations\dnn\modelpath\\1575536466")
        print(meta_graph_def)
        signature = meta_graph_def.signature_def
        visit_items_index_tensor_name = signature[signature_key].inputs["visit_items_index"].name
        visit_items_index_tensor = sess.graph.get_tensor_by_name(visit_items_index_tensor_name)
        continuous_features_value_tensor_name = signature[signature_key].inputs["continuous_features_value"].name
        continuous_features_value_tensor = sess.graph.get_tensor_by_name(continuous_features_value_tensor_name)
        user_vector_tensor_name = signature[signature_key].outputs["user_vector"].name
        user_vector_tensor = sess.graph.get_tensor_by_name(user_vector_tensor_name)
        index_tensor_name = signature[signature_key].outputs["index"].name
        index_tensor = sess.graph.get_tensor_by_name(index_tensor_name)

        features_result = sess.run(features)
        feed_dict = {visit_items_index_tensor: features_result["visit_items_index"], continuous_features_value_tensor: features_result["continuous_features_value"]}
        predict_outputs = sess.run([user_vector_tensor, index_tensor], feed_dict=feed_dict)
        print(predict_outputs[0])
        print("==========")
        print(predict_outputs[1])