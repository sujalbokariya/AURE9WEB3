from pyspark.sql import SparkSession


def load_Dataset():
    # File location and type
    file_location = "D:\\EDI\\dataset.csv"
    file_type = "csv"

    # CSV options
    infer_schema = "true"
    first_row_is_header = "true"
    delimiter = ","
    spark = SparkSession.builder.appName('models').master("local[16]").config("spark.driver.memory","8g").config("spark.executor.memory",'8g').getOrCreate()

    df = spark.read.format(file_type) \
        .option("inferSchema", infer_schema) \
        .option("header", first_row_is_header) \
        .option("sep", delimiter) \
        .load(file_location).repartition(128)
    # df = spark.read.parquet("D:\\EDI\\pd_speech_features")
    print(df.columns)
    return df
