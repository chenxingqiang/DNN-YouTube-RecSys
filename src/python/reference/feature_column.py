import numpy as np
import tensorflow as tf

member_id = "member_id_{}".format(1)
gds_cd = "gds_cd_{}".format(1)
age = np.random.randint(18, 60)
height = np.random.uniform(170.0, 190.0)
example = tf.train.Example(features=tf.train.Features(feature={
    "member_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(member_id)])),
    "gds_cd": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(gds_cd)])),
    "age": tf.train.Feature(int64_list=tf.train.Int64List(value=[age])),
    "height": tf.train.Feature(float_list=tf.train.FloatList(value=[height]))
}))
serialized_example = example.SerializeToString()

example_fmt = {
    "member_id": tf.FixedLenFeature([1], tf.string),
    "gds_cd": tf.FixedLenFeature([1], tf.string),
    "age": tf.FixedLenFeature([1], tf.int64),
    "height": tf.FixedLenFeature([1], tf.float32)
}
parsed = tf.parse_single_example(serialized_example, example_fmt)

member_id = tf.feature_column.categorical_column_with_hash_bucket("member_id", hash_bucket_size=3)
gds_cd = tf.feature_column.categorical_column_with_hash_bucket("gds_cd", hash_bucket_size=3)
age = tf.feature_column.categorical_column_with_vocabulary_list("age", [i for i in range(3)], dtype=tf.int64,
                                                                default_value=0)
height = tf.feature_column.numeric_column("height")
member_id_indicator = tf.feature_column.indicator_column(member_id)
gds_cd_indicator = tf.feature_column.indicator_column(gds_cd)
age_indicator = tf.feature_column.indicator_column(age)
feature_columns = [member_id_indicator, gds_cd_indicator, age_indicator, height]
_result = tf.feature_column.input_layer(parsed, feature_columns)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    parsed_result = sess.run([parsed])
    print("parsed_result是:", parsed_result)
    result = sess.run([_result])
    print("result是:", result)










