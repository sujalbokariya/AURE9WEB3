from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import PolynomialExpansion
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA

stages = list()


def load_Dataset():
    # File location and type
    file_location = "D:\\EDI\\dataset.csv"
    file_type = "csv"

    # CSV options
    infer_schema = "true"
    first_row_is_header = "true"
    delimiter = ","
    spark = SparkSession.builder.appName('models').getOrCreate()

    df = spark.read.format(file_type) \
        .option("inferSchema", infer_schema) \
        .option("header", first_row_is_header) \
        .option("sep", delimiter) \
        .load(file_location)
    # df = spark.read.parquet("D:\\EDI\\pd_speech_features")
    print(df.columns)
    return df


def create_dummy_variables(cat_cols):
    # StringIndexer - maps a string column of labels to an ML column of label indices
    # OneHotEncoderEstimator - creates n-1 columns for column with 'n' categories
    # maps a column of category indices to a column of binary vectors
    # VectorAssembler - merges multiple columns into a vector column

    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

    categorical_indexed = [item + "_indexed" for item in cat_cols]
    categorical_encoded = [item + "_encoded" for item in cat_cols]

    indexer = StringIndexer(inputCols=cat_cols, outputCols=categorical_indexed, handleInvalid="keep")
    stages.append(indexer)

    encoder = OneHotEncoder(inputCols=categorical_indexed, outputCols=categorical_encoded)
    stages.append(encoder)

    categorical_assembler = VectorAssembler(inputCols=categorical_encoded, outputCol="categorical_cols")
    stages.append(categorical_assembler)


def create_model():
    df = load_Dataset()
    num_cols = df.columns
    num_cols.remove('class')
    numeric_assembler = VectorAssembler(inputCols=num_cols, outputCol="numerical_cols")
    stages.append(numeric_assembler)

    #px = PolynomialExpansion(degree=2, inputCol="numerical_cols", outputCol="poly_cols")
    #stages.append(px)

    scaler = MinMaxScaler(inputCol="numerical_cols", outputCol="numerical_scaled")
    stages.append(scaler)

    pca = PCA(k=50, inputCol="numerical_scaled", outputCol="pcacol")
    stages.append(pca)

    train, test = df.randomSplit([0.8, 0.2], seed=0)
    evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC', labelCol='class')
    rf = RandomForestClassifier(featuresCol='pcacol', labelCol='class')
    stages.append(rf)

    pipeline = Pipeline(stages=stages)
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [3, 5]) \
        .addGrid(rf.numTrees, [40, 50]) \
        .build()


    # three fold cross validation
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3, seed=1)

    model = crossval.fit(train)
    model.bestModel.save("D:/edi/randomforest")

    test = model.transform(test)
    print(evaluator.evaluate(test))


create_model()
