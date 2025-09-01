import tensorflow as tf

def parse_fn(example):
    example_fmt = {
        "embedding_average": tf.FixedLenFeature([8], tf.float32),
        "one_hot": tf.SparseFeature(index_key=["index"],
                                    value_key="value",
                                    dtype=tf.float32,
                                    size=[15])   # size必须写死, 不能传超参
    }
    parsed = tf.parse_single_example(example, example_fmt)
    return parsed["embedding_average"], tf.sparse_tensor_to_dense(parsed["one_hot"])

if __name__ == "__main__":
    files = tf.data.Dataset.list_files('../../data/tfrecords_methods/tfrecords/data1.tfrecords', shuffle=True)
    data_set = files.apply(
        tf.contrib.data.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=15))
    data_set = data_set.repeat(1)
    data_set = data_set.map(map_func=parse_fn, num_parallel_calls=15)
    data_set = data_set.prefetch(buffer_size=30)
    data_set = data_set.batch(batch_size=15)
    iterator = data_set.make_one_shot_iterator()
    embedding, one_hot = iterator.get_next()

    with tf.Session() as sess:
        for i in range(5):
            embedding_result, one_hot_result = sess.run([embedding, one_hot])
            print("第{}批:".format(i), end=" ")
            print("embedding是:", embedding_result, end=" ")
            print("one_hot是:", one_hot_result)