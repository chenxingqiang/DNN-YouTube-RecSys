package example

import sparkapplication.BaseSparkLocal

object Example2 extends BaseSparkLocal {
  def main(args:Array[String]):Unit = {
//    val spark = this.basicSpark
//    import spark.implicits._

    val gdsVector = "[8534.033203125,-6634.611328125,-20669.0703125,-9483.734375,8790.3935546875,15647.646484375,-15543.39453125,34464.3203125,-1275.48974609375,28998.267578125,2446.0126953125,32628.033203125,1429.67431640625,37169.6640625,1902.3770751953125,-31038.359375]"
    val gds123 = gdsVector.replace("[", "").replace("]", "").split(",", -1).map(_.toDouble)
    gds123.foreach(println(_))

  }
}
