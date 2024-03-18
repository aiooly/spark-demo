import com.linkedin.nn.SparkUtils
import com.linkedin.nn.algorithm.JaccardMinHashNNS
import org.apache.spark.ml.feature.{Tokenizer, Word2Vec}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector, Vectors}

object Demo extends SparkUtils {
  def main(args: Array[String]): Unit = sparkTest("Demo") {
    run()
  }

  private def run(): Unit = {
    val df = sparkSession.read.option("header", true).csv("src\resources\text01.csv")
    println(df.count())
    val tokenizer = new Tokenizer().setInputCol("question_text").setOutputCol("words")
    val tokenized = tokenizer.transform(df)

    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("vectors")
      .setVectorSize(3)
      .setMinCount(0)
    val word2VecModel = word2Vec.fit(tokenized)
    val result = word2VecModel.transform(tokenized)

    val items = result.select("id", "vectors").rdd.map(row => (row.getString(0).toLong, row.getAs[Vector](1)))

    val numFeatures = items.values.take(1)(0).size

    val model = new JaccardMinHashNNS()
      .setNumHashes(200)
      .setSignatureLength(10)
      .setBucketLimit(10)
      .setJoinParallelism(5)
      .createModel(numFeatures)


    val nbrs = model.getSelfAllNearestNeighbors(items, 10)

    nbrs.collect().foreach(row => println(row))
  }
}