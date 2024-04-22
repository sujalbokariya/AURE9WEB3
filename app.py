from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from flask import Flask, request, render_template


#spark = SparkSession.builder.appName('models').master("local[*]").getOrCreate()
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


UPLOAD_FOLDER = "D:\\EDI\\"   # Set the path to the directory where you want to save the uploaded file
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/submit', methods=['POST'])
def upload():
    # check if the post request has the file part
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return 'No selected file'

    # validate file extension
    if not file.filename.endswith('.csv'):
        return 'Invalid file type. Please upload a CSV file.'

    # save the file to disk
    file.save(UPLOAD_FOLDER + file.filename)
    return 'File uploaded successfully!'

@app.route('/predict', methods=['POST'])
def predict():
    # load the trained model
    model = PipelineModel.load("D:\EDI\logistic")

    # get the path of the uploaded file
    file_path = UPLOAD_FOLDER + request.form['file_name']

    # read the csv file into a Spark dataframe
    df = spark.read.format('csv').options(header='true', inferSchema='true').load(file_path)

    # make predictions using the model
    predictions = model.transform(df)

    # return the predictions as a JSON response
    return predictions.toJSON().collect()

if __name__ == '__main__':
    app.run(debug=True)
