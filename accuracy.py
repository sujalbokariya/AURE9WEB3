from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.shell import spark
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def load_Dataset() :
    # File location and type
    file_location = "D:\\EDI\\dataset.csv"
    file_type = "csv"

    # CSV options
    infer_schema = "true"
    first_row_is_header = "true"
    delimiter = ","
    spark = SparkSession.builder.appName('models').getOrCreate()

    # The applied options are for CSV files. For other file types, these will be ignored.
    df = spark.read.format(file_type) \
        .option("inferSchema", infer_schema) \
        .option("header", first_row_is_header) \
        .option("sep", delimiter) \
        .load(file_location)

    from pyspark.sql.types import DoubleType

    df = df.withColumn("label", df["Class"].cast(DoubleType()))
    return df

def load_test() :
    df = spark.read.parquet("D:\\EDI\\test")
    return df

df=load_Dataset()


def split() :
    train, test = df.randomSplit([0.8, 0.2], seed=0)
    train.write.parquet("D:\\EDI\\train")
    test.write.parquet("D:\\EDI\\test")



split()
def logistic() :
    #spark = SparkSession.builder.appName('models').getOrCreate()
    model = PipelineModel.load("D:\EDI\logistic")
    testdf = load_test()
    test = model.transform(testdf)
    evaluator1 = MulticlassClassificationEvaluator(metricName='f1')
    evaluator2 = MulticlassClassificationEvaluator(metricName='accuracy')
    evaluator3 = MulticlassClassificationEvaluator(metricName='logLoss')
    print("f1 = %g" % evaluator1.evaluate(test))
    print("accuracy= %g" % evaluator2.evaluate(test))
    print("logloss= %g" % evaluator3.evaluate(test))

logistic()

