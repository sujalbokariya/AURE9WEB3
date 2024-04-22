from pyspark.ml.feature import VectorAssembler, MinMaxScaler, PCA, PolynomialExpansion
from pyspark.ml import Pipeline
from LoadDataset import load_Dataset
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('models').master("local[*]").config("spark.driver.memory","6g").config("spark.executor.memory",'6g').getOrCreate()

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
    num_cols = ["gender", "PPE", "DFA", "RPDE", "numPulses", "numPeriodsPulses", "meanPeriodPulses",
                "stdDevPeriodPulses", "locPctJitter", "locAbsJitter", "rapJitter", "ppq5Jitter", "ddpJitter",
                "locShimmer", "locDbShimmer", "apq3Shimmer", "apq5Shimmer", "apq11Shimmer", "ddaShimmer",
                "meanAutoCorrHarmonicity",
                "meanNoiseToHarmHarmonicity", "meanHarmToNoiseHarmonicity", "minIntensity", "maxIntensity",
                "meanIntensity", "f1", "f2", "f3", "f4", "b1", "b2", "b3", "b4", "GQ_prc5_95", "GQ_std_cycle_open",
                "GQ_std_cycle_closed", "GNE_mean", "GNE_std", "GNE_SNR_TKEO", "GNE_SNR_SEO", "GNE_NSR_TKEO",
                "GNE_NSR_SEO", "VFER_mean", "VFER_std", "VFER_entropy", "VFER_SNR_TKEO", "VFER_SNR_SEO",
                "VFER_NSR_TKEO",
                "VFER_NSR_SEO",
                'VFER_NSR_SEO', 'IMF_SNR_SEO', 'IMF_SNR_TKEO', 'IMF_SNR_entropy', 'IMF_NSR_SEO', 'IMF_NSR_TKEO',
                'IMF_NSR_entropy', 'mean_Log_energy', 'mean_MFCC_0th_coef', 'mean_MFCC_1st_coef', 'mean_MFCC_2nd_coef',
                'mean_MFCC_3rd_coef', 'mean_MFCC_4th_coef', 'mean_MFCC_5th_coef', 'mean_MFCC_6th_coef',
                'mean_MFCC_7th_coef', 'mean_MFCC_8th_coef', 'mean_MFCC_9th_coef', 'mean_MFCC_10th_coef',
                'mean_MFCC_11th_coef', 'mean_MFCC_12th_coef', 'mean_delta_log_energy', 'mean_0th_delta',
                'mean_1st_delta',
                'mean_2nd_delta', 'mean_3rd_delta', 'mean_4th_delta', 'mean_5th_delta', 'mean_6th_delta',
                'mean_7th_delta',
                'mean_8th_delta', 'mean_9th_delta', 'mean_10th_delta', 'mean_11th_delta', 'mean_12th_delta',
                'mean_delta_delta_log_energy', 'mean_delta_delta_0th', 'mean_1st_delta_delta', 'mean_2nd_delta_delta',
                'mean_3rd_delta_delta', 'mean_4th_delta_delta', 'mean_5th_delta_delta', 'mean_6th_delta_delta',
                'mean_7th_delta_delta', 'mean_8th_delta_delta', 'mean_9th_delta_delta', 'mean_10th_delta_delta',
                'mean_11th_delta_delta', 'mean_12th_delta_delta', 'std_Log_energy', 'std_MFCC_0th_coef',
                'std_MFCC_1st_coef', 'std_MFCC_2nd_coef', 'std_MFCC_3rd_coef', 'std_MFCC_4th_coef'

                ]
    df = load_Dataset()
    numeric_assembler = VectorAssembler(inputCols=num_cols, outputCol="numerical_cols")
    stages.append(numeric_assembler)

    scaler = MinMaxScaler(inputCol="numerical_cols", outputCol="numerical_scaled")
    stages.append(scaler)

    pca = PCA(k=78, inputCol="numerical_scaled", outputCol="pcacol")
    stages.append(pca)

    pipeline = Pipeline(stages=stages)

    model = pipeline.fit(df)
    model.transform(df).write.parquet("D:/edi/data2")


create_model()

