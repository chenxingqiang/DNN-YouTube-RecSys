package explore

import org.tensorflow._
import sparkapplication.BaseSparkOnline
import scala.collection.JavaConverters._

object ItemEmbeddingMakeDataOne extends BaseSparkOnline {
  def main(args:Array[String]):Unit = {
    val spark = this.basicSpark
    import spark.implicits._

    val modelHdfsPath = "hdfs路径"
    val modelTag = "serve"

    val embeddingAverageArray = Array(Array.fill[Float](8)(0.1F))
    val model = SavedModelBundle.load(modelHdfsPath, modelTag)
    val sess = model.session()
    val embeddingAverageArrayTensor = Tensor.create(embeddingAverageArray, classOf[java.lang.Float])
    val itemEmbeddingResult = getItemEmbedding(sess, embeddingAverageArrayTensor)

    val result = spark.sparkContext.parallelize(itemEmbeddingResult.map(k => k.mkString("@"))).toDF("item_embedding")
    result.show(10, false)
  }

  private def getItemEmbedding(sess: Session, embeddingAverageArrayTensor: Tensor[_], embeddingAverageArrayName: String = "Placeholder:0", itemEmbeddingName: String = "item_embedding:0") = {
    val resultBuffer = sess.runner
      .feed(embeddingAverageArrayName, embeddingAverageArrayTensor)
      .fetch(itemEmbeddingName)
      .run.asScala

    val itemEmbedding = resultBuffer.head
    val itemEmbeddingShape: Array[Int] = itemEmbedding.shape.map(_.toInt)
    val itemEmbeddingResult = Array.ofDim[Float](itemEmbeddingShape.head, itemEmbeddingShape(1))
    itemEmbedding.copyTo(itemEmbeddingResult)

    itemEmbeddingResult
  }
}
