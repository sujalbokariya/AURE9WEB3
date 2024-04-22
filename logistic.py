from pyspark.ml.feature import VectorAssembler, MinMaxScaler, PCA, PolynomialExpansion
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from LoadDataset import load_Dataset

stages = list()


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

    pca = PCA(k=69, inputCol="numerical_scaled", outputCol="pcacol")
    stages.append(pca)

    train, test = df.randomSplit([0.8, 0.2], seed=0)
    evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC', labelCol='class')
    lr = LogisticRegression(featuresCol='pcacol', labelCol='class')
    stages.append(lr)

    pipeline = Pipeline(stages=stages)
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.maxIter, [20]) \
        .addGrid(lr.regParam, [0.1]) \
        .build()

    # three fold cross validation
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3, seed=1)

    model = crossval.fit(train)
    model.bestModel.save("D:/edi/logistic")

    test = model.transform(test)
    print(evaluator.evaluate(test))


create_model()
