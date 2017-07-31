/*
 * http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
 */
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import static org.apache.spark.sql.functions.*;

import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.regression.LabeledPoint;

public class KDDCupLRExample {
  public static void main(String[] args) throws Exception {
   
    SparkSession spark = SparkSession
      .builder()
      .appName("KDDCupLRExample")
      .getOrCreate();

    String[] header = {
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label"};

    // Load training data
    Dataset<Row> training = spark.read().format("csv")
                               .option("inferSchema","true")
                               .option("header","false")
                               .load("/dataset/kddcup/kddcup.data")
                               .toDF(header);

    // Replace string label to 0.0 (good) and 1.0 (bad)
    training = training.withColumn("label", 
                   when (training.col("label").equalTo("normal."),0.0)
                   .otherwise(1.0)
    );

    // Create features column
    String[] trainingCols =  {
        "duration", 
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"};

    VectorAssembler assembler = new VectorAssembler()
                 .setInputCols(trainingCols)
                 .setOutputCol("features");
    Dataset<Row> input = assembler.transform(training);

    // Create and fit the model
    LogisticRegression lr = new LogisticRegression()
      .setMaxIter(100)
      .setStandardization(true)
      .setFeaturesCol("features")
      .setLabelCol("label");
    LogisticRegressionModel lrModel = lr.fit(input);

    // Load testing data
    Dataset<Row> testing = spark.read().format("csv")
                               .option("inferSchema","true")
                               .option("header","false")
                               .load("/dataset/kddcup/corrected")
                               .toDF(header);

    // Replace string label to 0.0 (good) and 1.0 (bad)
    testing = testing.withColumn("label", 
                when (testing.col("label").equalTo("normal."),0.0)
                .otherwise(1.0)
    );

    // Create features column
    Dataset<Row> newInput = assembler.transform(testing);

    // Test the model with new data
    Dataset<Row> output = lrModel.transform(newInput);

    // Check accuracy
    List<Row> rows = output.select("label","prediction").collectAsList();
    long correct = 0;
    for (Row row: rows) {
       double label = row.getDouble(0);
       double prediction = row.getDouble(1);
       if (label == prediction) correct++;
    }
    long total = newInput.count();
    System.out.println("Accuracy = " + (double) correct / total);

    spark.stop();
  }
}
