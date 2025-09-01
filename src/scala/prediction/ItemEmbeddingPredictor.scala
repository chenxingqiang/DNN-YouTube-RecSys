package explore

import org.tensorflow._
import org.tensorflow.example._
import explore.FeatureBuilder._
import sparkapplication.BaseSparkOnline
import scala.collection.JavaConverters._

object ItemEmbeddingMakeDataTwo extends BaseSparkOnline {
  def main(args:Array[String]):Unit = {
    val spark = this.basicSpark
    import spark.implicits._

    val modelHdfsPath = "hdfs路径"
    val modelTag = "serve"

    val embeddingAverage = Array.fill[Float](8)(0.1F)
    val gender = "male"
    val cityCd = "city_cd_100"
    val featuresBuilder = Features.newBuilder
      .putFeature("embedding_average", f(embeddingAverage:_*))
      .putFeature("gender", s(gender))
      .putFeature("city_cd", s(cityCd))
    featuresBuilder.build()
    val features = Example.newBuilder.setFeatures(featuresBuilder).build.toByteArray

    val model = SavedModelBundle.load(modelHdfsPath, modelTag)
    val sess = model.session()
    val embeddingAverageArrayTensor = Tensor.create(Array(features))
    val itemEmbeddingResult = getItemEmbedding(sess, embeddingAverageArrayTensor)

    val result = spark.sparkContext.parallelize(itemEmbeddingResult.map(k => k.mkString("@"))).toDF("item_embedding")
    result.show(50, false)
  }

  private def getItemEmbedding(sess: Session, featuresArrayTensor: Tensor[_], featuresArrayName: String = "input_example_tensor:0", itemEmbeddingName: String = "item_embedding:0") = {
    val resultBuffer = sess.runner
      .feed(featuresArrayName, featuresArrayTensor)
      .fetch(itemEmbeddingName)
      .run.asScala

    val itemEmbedding = resultBuffer.head
    val itemEmbeddingShape: Array[Int] = itemEmbedding.shape.map(_.toInt)
    val itemEmbeddingResult = Array.ofDim[Float](itemEmbeddingShape.head, itemEmbeddingShape(1))
    itemEmbedding.copyTo(itemEmbeddingResult)

    itemEmbeddingResult
  }
}
