from flask import Flask, request, render_template
import csv
import os
import numpy as np
import pandas as pd
import generateOutput 

app = Flask(__name__)
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop\\')
app.config['UPLOAD_FOLDER'] = desktop


@app.route('/')
def default():
    return render_template("index.html")

@app.route('/index.html')
def index():
    return render_template("index.html")

@app.route('/sales.html')
def sales():
    return render_template("sales.html")

@app.route('/predict')
def predict():
    if not (os.path.exists(f'{desktop}/data.csv')):
        return 'Data file does not exist. Please enter data first'
    else:
        return generateOutput.begin()
        #run model get results


@app.route('/form1data', methods =['GET','POST'])
def form1():
    date = request.form['date']
    item_id = request.form['itemid']
    quantity = request.form['quantity']

    file_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'data.csv')
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(['id', 'date', 'store', 'item', 'sales'])

        if file_exists:
            with open(file_path, 'r') as csvfile:
                last_id = sum(1 for _ in csvfile) - 1
        else:
            last_id = -1

        new_id = last_id + 1
        writer.writerow([new_id, date, 1, item_id, quantity])
    return render_template("sales.html")



@app.route('/form2data', methods=['POST'])
def form2():
    file_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'data.csv')
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(['id', 'date', 'store', 'item', 'sales'])

        if file_exists:
            with open(file_path, 'r') as csvfile:
                last_id = sum(1 for _ in csvfile) - 1
        else:
            last_id = -1

        datafile = request.files['datafile']
        if datafile:
            reader = csv.reader(datafile.read().decode('utf-8').splitlines())
            next(reader)
            for row in reader:
                date = row[0]
                item_id = row[1]
                quantity = row[2]

                new_id = last_id + 1
                writer.writerow([new_id, date, 1, item_id, quantity])
                last_id += 1

    return render_template("sales.html")


if __name__ == "__main__":
    app.run(debug=True)