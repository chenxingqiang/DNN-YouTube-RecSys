package explore

import org.tensorflow.example._
import org.tensorflow.spark.shaded.com.google.protobuf.ByteString

object FeatureBuilder {
  def s(strings: String*): Feature = {
    val b = BytesList.newBuilder
    for (s <- strings) {
      b.addValue(ByteString.copyFromUtf8(s))
    }
    Feature.newBuilder.setBytesList(b).build
  }

  def f(values: Float*): Feature = {
    val b = FloatList.newBuilder
    for (v <- values) {
      b.addValue(v)
    }
    Feature.newBuilder.setFloatList(b).build
  }

  def i(values: Int*): Feature = {
    val b = Int64List.newBuilder
    for (v <- values) {
      b.addValue(v)
    }
    Feature.newBuilder.setInt64List(b).build
  }
}
