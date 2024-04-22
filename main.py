from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
import os
from flask import Flask, request, render_template


#spark = SparkSession.builder.appName('models').master("local[*]").getOrCreate()
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


UPLOAD_FOLDER = "D:\\"   # Set the path to the directory where you want to save the uploaded file
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])

def upload():
    # check if the post request has the file part
    if 'image_uploads' not in request.files:
        return 'No file part'
    file = request.files['image_uploads']
    # print(file)
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return 'No selected file'
    # print("jello")

    # validate file extension
    if not file.filename.endswith('.csv'):
        return 'Invalid file type. Please upload a CSV file.'

    # save the file to disk
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return (lg(file.filename))

    #return 'File uploaded successfully!'

def lg(filename):
    model = PipelineModel.load("D:\EDI\EDI Sem3\logistic")

    file_location="D://" + filename
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

    if (model.transform(df).toPandas()["prediction"][0]) == 0.0:
        print("The person doesnt have Parkinson's Disease")
        return render_template('no.html')

    elif (model.transform(df).toPandas()["prediction"][0]) == 1.0:
        print("The person has Parkinson's Disease ")
        return render_template('yes.html')


if __name__ == '__main__':
    app.run(debug=True)