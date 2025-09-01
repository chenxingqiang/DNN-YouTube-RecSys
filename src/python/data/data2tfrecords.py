import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    sample_num = 5000
    item_num = 500
    sample_set = []
    for i in range(sample_num):
        visit_items_index = np.random.randint(low=0, high=item_num, size=[5])
        continuous_features_value = np.random.uniform(low=-5.0, high=5.0, size=[16])
        next_visit_item_index = np.random.randint(low=0, high=item_num)
        sample = [visit_items_index, continuous_features_value, next_visit_item_index]
        sample_set.append(sample)

    # 训练数据
    filename = "../../data/tfrecords/train/train.tfrecords"
    writer = tf.python_io.TFRecordWriter(filename)
    for sample in sample_set:
        visit_items_index = sample[0]
        continuous_features_value = sample[1]
        next_visit_item_index = sample[2]
        example = tf.train.Example(features=tf.train.Features(feature={
            "visit_items_index": tf.train.Feature(int64_list=tf.train.Int64List(value=visit_items_index)),
            "continuous_features_value": tf.train.Feature(
                float_list=tf.train.FloatList(value=continuous_features_value)),
            "next_visit_item_index": tf.train.Feature(int64_list=tf.train.Int64List(value=[next_visit_item_index]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

    # 评估数据, 由于数据是随机生成, 所以评估数据从训练数据中取
    filename = "../../data/tfrecords/evaluation/evaluation.tfrecords"
    writer = tf.python_io.TFRecordWriter(filename)
    i = 0
    for sample in sample_set:
        if i % 10 == 0:
            visit_items_index = sample[0]
            continuous_features_value = sample[1]
            next_visit_item_index = sample[2]
            example = tf.train.Example(features=tf.train.Features(feature={
                "visit_items_index": tf.train.Feature(int64_list=tf.train.Int64List(value=visit_items_index)),
                "continuous_features_value": tf.train.Feature(
                    float_list=tf.train.FloatList(value=continuous_features_value)),
                "next_visit_item_index": tf.train.Feature(int64_list=tf.train.Int64List(value=[next_visit_item_index]))
            }))
            writer.write(example.SerializeToString())
        i = i + 1
    writer.close()