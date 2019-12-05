import tensorflow as tf

def parse_fn(example):
    example_fmt = {
        "embedding_average": tf.FixedLenFeature([8], tf.float32),
        "index": tf.FixedLenFeature([], tf.int64),
        "value": tf.FixedLenFeature([], tf.float32),
        "size": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    sparse_tensor = tf.SparseTensor([[parsed["index"]]], [parsed["value"]], [parsed["size"]])   # 这种方法读取稀疏向量在有的平台可能不行
    return parsed["embedding_average"], tf.sparse_tensor_to_dense(sparse_tensor)

if __name__ == "__main__":
    files = tf.data.Dataset.list_files('D:\Pycharm\PycharmProjects\deep_neural_networks_for_youtube_recommendations\\tfrecords_methods\\tfrecords\data1.tfrecords', shuffle=True)
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