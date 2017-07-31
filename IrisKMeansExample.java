/*
 * https://archive.ics.uci.edu/ml/datasets/iris
 */
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import org.apache.spark.sql.SparkSession;

import java.util.List;

public class IrisKMeansExample {

  public static void main(String[] args) {

    // Create a SparkSession
    SparkSession spark = SparkSession
      .builder()
      .appName("IrisKMeansExample")
      .getOrCreate();

    // Loads data
    Dataset<Row> dataset = spark.read().format("csv")
                     .option("inferSchema","true")
                     .option("header","false")
                     .load("/dataset/iris/iris.data")
                     .toDF("sepal_length","sepal_width","petal_length","petal_width","label");

    // Create features column
    String[] featureColumns = {"sepal_length","sepal_width","petal_length","petal_width"} ;
    VectorAssembler assembler = new VectorAssembler()
                     .setInputCols(featureColumns)
                     .setOutputCol("features");
    Dataset<Row> training = assembler.transform(dataset);
    training.show();
    
    // Trains a k-means model
    KMeans kmeans = new KMeans()
                     .setK(3)           // number of clusters
                     .setMaxIter(20)    // max itrations
                     .setSeed(1L);
    KMeansModel model = kmeans.fit(training);

    // Show results 
    Dataset<Row> results = model.transform(training); 
    List<Row> rows = results.collectAsList();
    for (Row row : rows) {
       System.out.print(row.getAs("prediction") + " ");
    }
    System.out.println();

    // Shows the cluster centers
    Vector[] centers = model.clusterCenters();
    System.out.println("Cluster Centers: ");
    for (Vector center: centers) {
      System.out.println(center);
    }

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    double SSE = model.computeCost(training);
    System.out.println("Within Set Sum of Squared Errors = " + SSE);

    spark.stop();
  }
}
