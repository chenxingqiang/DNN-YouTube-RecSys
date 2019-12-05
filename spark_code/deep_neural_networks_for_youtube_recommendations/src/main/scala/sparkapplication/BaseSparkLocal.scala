package sparkapplication

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

trait BaseSparkLocal {
  //本地
  def basicSpark: SparkSession =
    SparkSession
      .builder
      .config(getSparkConf)
      .master("local[1]")
      .getOrCreate()

  def getSparkConf: SparkConf = {
    val conf = new SparkConf()
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.network.timeout", "600")
      .set("spark.streaming.kafka.maxRatePerPartition", "200000")
      .set("spark.streaming.kafka.consumer.poll.ms", "5120")
      .set("spark.streaming.concurrentJobs", "5")
      .set("spark.sql.crossJoin.enabled", "true")
      .set("spark.driver.maxResultSize", "1g")
      .set("spark.rpc.message.maxSize", "1000") // 1024 max
    conf
  }
}
