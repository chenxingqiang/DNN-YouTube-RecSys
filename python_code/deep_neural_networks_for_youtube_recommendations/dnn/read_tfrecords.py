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

if __name__ == "__main__":
    files = tf.data.Dataset.list_files('D:\Pycharm\PycharmProjects\deep_neural_networks_for_youtube_recommendations\dnn\\tfrecords\\train\\train.tfrecords', shuffle=True)
    data_set = files.apply(
        tf.contrib.data.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=16))
    data_set = data_set.repeat(1)
    data_set = data_set.map(map_func=parse_fn, num_parallel_calls=16)
    data_set = data_set.prefetch(buffer_size=64)
    data_set = data_set.batch(batch_size=16)
    iterator = data_set.make_one_shot_iterator()
    res1, res2 = iterator.get_next()

    with tf.Session() as sess:
        for i in range(5):
            result1, result2 = sess.run([res1, res2])
            print("第{}批:".format(i), end=" ")
            print("result1是:", result1)
            print("result2是:", result2)