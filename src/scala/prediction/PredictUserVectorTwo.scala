package explore

import org.tensorflow._
import org.tensorflow.example._
import explore.FeatureBuilder._
import sparkapplication.BaseSparkOnline
import scala.collection.JavaConverters._

object PredictUserVectorMakeDataTwo extends BaseSparkOnline {
  def main(args:Array[String]):Unit = {
    val spark = this.basicSpark
    import spark.implicits._

    val modelHdfsPath = "hdfs路径"
    val modelTag = "serve"

    val dataValidation = spark.read.format("tfrecords")
      .option("recordType", "Example")
      .load("hdfs路径")
      .rdd.map{row =>
      val embeddingAverage = row.getAs[scala.collection.mutable.WrappedArray[Float]]("embedding_average").toArray
      val gender = row.getAs[String]("gender")
      val cityCd = row.getAs[String]("city_cd")
      val featuresBuilder = Features.newBuilder
        .putFeature("embedding_average", f(embeddingAverage:_*))
        .putFeature("gender", s(gender))
        .putFeature("city_cd", s(cityCd))
      featuresBuilder.build()
      val features = Example.newBuilder.setFeatures(featuresBuilder).build.toByteArray
      features
    }
    println(s"验证集数据dataValidation总数为:${dataValidation.count},数据格式如下:")
    dataValidation.toDF("features").show(5, false)

    val userVectorAll = dataValidation.mapPartitions(lineIterator => {
      val featuresArray = lineIterator.toArray
      val model = SavedModelBundle.load(modelHdfsPath, modelTag)
      val sess = model.session()
      val featuresArrayTensor = Tensor.create(featuresArray)
      val userVectorResult = predictUserVector(sess, featuresArrayTensor)
      userVectorResult.toIterator
    })

    val result = userVectorAll.map(k => k.mkString("@")).toDF("user_vector")
    result.show(10, false)
  }

  private def predictUserVector(sess: Session, featuresArrayTensor: Tensor[_], featuresArrayName: String = "input_example_tensor:0", userVectorName: String = "user_vector/Relu:0") = {
    val resultBuffer = sess.runner
      .feed(featuresArrayName, featuresArrayTensor)
      .fetch(userVectorName)
      .run.asScala

    val userVector = resultBuffer.head
    val userVectorShape: Array[Int] = userVector.shape.map(_.toInt)
    val userVectorResult = Array.ofDim[Float](userVectorShape.head, userVectorShape(1))
    userVector.copyTo(userVectorResult)

    userVectorResult
  }

}
