from flask import Flask, request, render_template
import csv
import os
import numpy as np
import pandas as pd

app = Flask(__name__)
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop\\')
app.config['UPLOAD_FOLDER'] = desktop


@app.route('/')
def default():
    return render_template("index.html")

@app.route('/sales.html')
def sales():
    return render_template("sales.html")

@app.route('/predict')
def predict():
    if not (os.path.exists(f'{desktop}/data.csv')):
        return 'Data file does not exist. Please enter data first'
    else:
        print()
        #run model get results


@app.route('/form1data', methods =['GET','POST'])
def form1():

    date = request.form['date']
    itemid = request.form['itemid']
    quantiy = request.form['quantity']
    desktop_path = os.path.expanduser('~/Desktop')

    os.makedirs(desktop_path, exist_ok=True)
    file_path = os.path.join(desktop_path, 'data.csv')

    if not (os.path.exists(f'{desktop}/data.csv')):
        
        data = np.array([
            ['id', 'date', 'store','item','sales'],
            ['0', date, '1',itemid,quantiy]
        ])

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    else:
        num_rows=0
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            num_rows = sum(1 for row in reader)

        with open(file_path, mode='a') as file:
             writer = csv.writer(file)
             writer.writerow([num_rows, date, 1, itemid, quantiy])

    return render_template("sales.html")



@app.route('/form2data', methods=['POST'])
def form2():
    datafile = request.files['datafile']
    df = pd.read_csv(datafile)

    desktop_path = os.path.expanduser('~/Desktop')
    os.makedirs(desktop_path, exist_ok=True)
    file_path = os.path.join(desktop_path, 'data.csv')

    num_rows=0
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        num_rows = sum(1 for row in reader)

    for i in range(0, df.shape[0]):
        with open(file_path, mode='a') as file:
            print(df.shape)
            print(num_rows)
            writer = csv.writer(file)
            writer.writerow([num_rows, df.iloc[i,0], 1, df.iloc[i,1], df.iloc[i,2]]) #id date store itemid sales

    return render_template("sales.html")


if __name__ == "__main__":
    app.run(debug=True)