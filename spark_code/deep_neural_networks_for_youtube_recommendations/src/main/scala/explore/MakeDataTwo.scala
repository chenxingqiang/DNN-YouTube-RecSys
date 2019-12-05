package explore

import java.util.Random
import org.apache.spark.sql._
import sparkapplication.BaseSparkOnline
import scala.collection.mutable.ArrayBuffer

object MakeDataTwo extends BaseSparkOnline {
  def main(args:Array[String]):Unit = {
    val spark = this.basicSpark
    import spark.implicits._

    // 训练数据
    var perItemSampleNum = 20
    var itemNum = 15
    var embeddingSize = 8
    val trainData = ArrayBuffer[(Array[Double], String, String, Long)]()
    for(i <- 0 until perItemSampleNum) {
      for(j <- 0 until itemNum){
        val embeddingAverage = Array.fill[Double](embeddingSize)(1.0*j + (new Random).nextDouble())
        val gender = if(j % 2 == 0) "male" else "female"
        val cityCd = "city_cd_" + (new Random).nextInt(200).toString
        trainData.append((embeddingAverage, gender, cityCd, j.toLong))
      }
    }
    val trainDataFrame = spark.sparkContext.parallelize(trainData, 10).toDF("embedding_average", "gender", "city_cd", "index")

    // Save DataFrame as TFRecords
    trainDataFrame.write.mode(SaveMode.Overwrite).format("tfrecords").option("recordType", "Example").save("hdfs路径")

    // Read TFRecords into DataFrame.
    val trainDataTfrecords: DataFrame = spark.read.format("tfrecords").option("recordType", "Example").load("hdfs路径")
    println("trainDataFrame重新加载tfrecords格式的数据,数据格式如下:")
    trainDataTfrecords.show(10, false)

    // 评估数据
    perItemSampleNum = 10
    itemNum = 15
    embeddingSize = 8
    val evaluationData = ArrayBuffer[(Array[Double], String, String, Long)]()
    for(i <- 0 until perItemSampleNum) {
      for(j <- 0 until itemNum){
        val embeddingAverage = Array.fill[Double](embeddingSize)(1.0*j + (new Random).nextDouble())
        val gender = if(j % 2 == 0) "male" else "female"
        val cityCd = "city_cd_" + (new Random).nextInt(200).toString
        evaluationData.append((embeddingAverage, gender, cityCd, j.toLong))
      }
    }
    val evaluationDataFrame = spark.sparkContext.parallelize(evaluationData, 10).toDF("embedding_average", "gender", "city_cd", "index")

    // Save DataFrame as TFRecords
    evaluationDataFrame.write.mode(SaveMode.Overwrite).format("tfrecords").option("recordType", "Example").save("hdfs路径")

    // Read TFRecords into DataFrame.
    val evaluationDataTfrecords: DataFrame = spark.read.format("tfrecords").option("recordType", "Example").load("hdfs路径")
    println("evaluationData重新加载tfrecords格式的数据,数据格式如下:")
    evaluationDataTfrecords.show(10, false)

  }
}
