import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    per_item_sample_num = 20
    item_num = 15
    embedding_size = 8
    filename = "D:\Pycharm\PycharmProjects\deep_neural_networks_for_youtube_recommendations\\tfrecords_methods\\tfrecords\\data1.tfrecords"
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(per_item_sample_num):
        for j in range(item_num):
            embedding_average = np.random.uniform(low=j, high=j + 1.0, size=[embedding_size])
            index = j
            example = tf.train.Example(features=tf.train.Features(feature={
                "embedding_average": tf.train.Feature(float_list=tf.train.FloatList(value=embedding_average)),
                "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                "value": tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])),
                "size": tf.train.Feature(int64_list=tf.train.Int64List(value=[item_num]))
            }))
            writer.write(example.SerializeToString())
    writer.close()