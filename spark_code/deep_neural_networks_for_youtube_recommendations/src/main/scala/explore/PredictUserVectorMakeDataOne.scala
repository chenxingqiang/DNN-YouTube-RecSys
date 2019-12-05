package explore

import org.tensorflow._
import sparkapplication.BaseSparkOnline
import scala.collection.JavaConverters._

object PredictUserVectorMakeDataOne extends BaseSparkOnline {
  def main(args:Array[String]):Unit = {
    val spark = this.basicSpark
    import spark.implicits._

    val modelHdfsPath = "hdfs路径"
    val modelTag = "serve"

    val dataValidation = spark.read.format("tfrecords")
      .option("recordType", "Example")
      .load("hdfs路径")
      .rdd.map{row =>
      val embeddingAverage = row.getAs[scala.collection.mutable.WrappedArray[Float]]("embedding_average")
      embeddingAverage.toArray
    }
    println(s"验证集数据dataValidation总数为:${dataValidation.count},数据格式如下:")
    dataValidation.toDF("embedding_average").show(5, false)

    val userVectorAll = dataValidation.mapPartitions(lineIterator => {
      val embeddingAverageArray = lineIterator.toArray
      val model = SavedModelBundle.load(modelHdfsPath, modelTag)
      val sess = model.session()
      val embeddingAverageArrayTensor = Tensor.create(embeddingAverageArray, classOf[java.lang.Float])
      val userVectorResult = predictUserVector(sess, embeddingAverageArrayTensor)
      userVectorResult.toIterator
    })

    val result = userVectorAll.map(k => k.mkString("@")).toDF("user_vector")
    result.show(10, false)
  }

  private def predictUserVector(sess: Session, embeddingAverageArrayTensor: Tensor[_], embeddingAverageArrayName: String = "Placeholder:0", userVectorName: String = "user_vector/Relu:0") = {
    val resultBuffer = sess.runner
      .feed(embeddingAverageArrayName, embeddingAverageArrayTensor)
      .fetch(userVectorName)
      .run.asScala

    val userVector = resultBuffer.head
    val userVectorShape: Array[Int] = userVector.shape.map(_.toInt)
    val userVectorResult = Array.ofDim[Float](userVectorShape.head, userVectorShape(1))
    userVector.copyTo(userVectorResult)

    userVectorResult
  }

}
