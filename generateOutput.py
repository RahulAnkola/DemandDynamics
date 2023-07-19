import base64
import calendar
import io
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
matplotlib.use('Agg')

def begin():
    # Define the function to create sequences
    def create_sequences(data, seq_length):
        xs = []
        ys = []

        for i in range(len(data) - seq_length - 1):
            x = data.iloc[i:(i+seq_length)].values
            y = data.iloc[i+seq_length]
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    # Load the data
    data_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'data.csv')
    #data_path = r"C:\Users\ksshr\Downloads\output.csv"  # Update with your file path
    data = pd.read_csv(data_path)

    # Convert 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'], format='mixed')
    # Set 'date' as the index
    data.set_index('date', inplace=True)

    # Get a list of unique items
    items = data['item'].unique()

    # Create a dictionary to store the predicted and actual sales for each item
    sales_data = {}

    # For each item
    for item in items:
        # Select the data for the current item
        data_item = data[data['item'] == item]

        # Scale 'sales' data to be between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_item['sales'] = scaler.fit_transform(data_item['sales'].values.reshape(-1, 1))

        # Split the data into training and test sets in an 80-20 ratio, excluding the last 90 days for testing
        data_item, future_90_days = data_item[:-90], data_item[-90:]
        train, test = train_test_split(data_item, test_size=0.2, shuffle=False)

        # Create sequences from the training data
        seq_length = 30
        X_train, y_train = create_sequences(train['sales'], seq_length)

        # Reshape the inputs to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Define the model
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=20, verbose=1)

        # Use the model to predict the sales for the "future" 90 days
        predicted_sales = []
        for _ in range(90):
            last_sequence = np.array(train['sales'][-seq_length:])
            last_sequence = last_sequence.reshape((1, seq_length, 1))
            prediction = model.predict(last_sequence)
            
            # Append the predicted sales to the list
            predicted_sales.append(prediction[0][0])
            
            # Add the predicted sales to the training data
            train = train._append({'sales': prediction[0][0]}, ignore_index=True)

        # Inverse the scaling of the prediction to get the real sales number
        prediction_inv = scaler.inverse_transform(np.array(predicted_sales).reshape(-1, 1))

        # Get the actual sales for the "future" 90 days and inverse transform them
        actual_sales = future_90_days['sales'].values
        actual_sales_inv = scaler.inverse_transform(actual_sales.reshape(-1, 1))

        # Store the predicted and actual sales in the dictionary
        sales_data[item] = {'Predicted': prediction_inv.flatten(), 'Actual': actual_sales_inv.flatten()}
        predicted_sales_data = {}

    # Loop over the sales_data dictionary
    for item, data in sales_data.items():
        # Store only the predicted sales
        predicted_sales_data[item] = {'Predicted': data['Predicted']}

    # Convert the keys to integers and sort the dictionary
    predicted_sales_data = {int(k): v for k, v in sorted(predicted_sales_data.items(), key=lambda item: int(item[0]))}
    output = str(predicted_sales_data)
    
    sorted_sales_data = {int(k): v for k, v in sorted(sales_data.items(), key=lambda item: int(item[0]))}
    item_groups = [list(sorted_sales_data.keys())[i:i+15] for i in range(0, len(sorted_sales_data), 15)]

    # Create the HTML table
    table_html = "<style>"
    table_html += "table {border-collapse: collapse; width: 100%; margin-bottom: 20px;}"
    table_html += "th, td {border: 1px solid black; padding: 10px;}"
    table_html += "th {background-color: #008000; color: #ffffff;}"
    table_html += "tr:not(:first-child) {background-color: #ccffcc;}"
    table_html += "th, td:first-child {padding-left: 20px;}"
    table_html += "th:first-child, td:first-child {padding-top: 16px;}"
    table_html += "</style>"
    table_html += "<table>"
    table_html += "<tr><th style='width: 10%;'>Item ID</th><th style='width: 10%;'>Total Sales (Month 1)</th><th style='width: 10%;'>Total Sales (Month 2)</th><th style='width: 10%;'>Total Sales (Month 3)</th><th style='width: 10%;'>Total Sales (All Months)</th></tr>"

    # Loop over the sorted sales_data dictionary
    for item, data in sorted_sales_data.items():
        # Get the item ID and predicted sales
        item_id = str(item)
        predicted_sales = data['Predicted']

        # Calculate the total sales for each month
        total_sales = []
        for month in range(1, 4):
            start_day = calendar.monthrange(2023, month)[0] + 1
            end_day = start_day + calendar.monthrange(2023, month)[1]
            monthly_sales = round(sum(predicted_sales[start_day-1:end_day]))
            total_sales.append(monthly_sales)

        # Calculate the total sales for all months
        total_sales_all = round(sum(total_sales), 2)

        # Add a row to the HTML table
        table_html += "<tr>"
        table_html += f"<td>{item_id}</td>"
        for sales in total_sales:
            table_html += f"<td>{sales}</td>"
        table_html += f"<td>{total_sales_all}</td>"
        table_html += "</tr>"

    # Close the HTML table
    table_html += "</table>"

    # Create the grouped bar charts
    chart_html = ""
    for item_group in item_groups:
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6)

        # Prepare data for the item group
        labels = []
        sales_month1 = []
        sales_month2 = []
        sales_month3 = []

        for item in item_group:
            labels.append(str(item))
            sales_month1.append(sorted_sales_data[item]['Predicted'][0])
            sales_month2.append(sorted_sales_data[item]['Predicted'][1])
            sales_month3.append(sorted_sales_data[item]['Predicted'][2])

        # Set the width of the bars
        bar_width = 0.2

        # Set the positions of the bars
        positions_month1 = range(len(item_group))
        positions_month2 = [pos + bar_width for pos in positions_month1]
        positions_month3 = [pos + bar_width for pos in positions_month2]

        # Create the grouped bar chart
        ax.bar(positions_month1, sales_month1, bar_width, label='Month 1')
        ax.bar(positions_month2, sales_month2, bar_width, label='Month 2')
        ax.bar(positions_month3, sales_month3, bar_width, label='Month 3')

        # Set the x-axis labels
        ax.set_xticks([p + bar_width for p in positions_month2])
        ax.set_xticklabels(labels)

        # Set the y-axis label
        ax.set_ylabel('Total Sales')

        # Set the chart title
        ax.set_title('Total Sales per Item')

        # Add a legend
        ax.legend()

        # Set the background color of the plot
        ax.set_facecolor('#f0f0f0')

        # Save the chart to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Encode the image as base64
        chart_image = base64.b64encode(buffer.getvalue()).decode()

        # Generate the chart HTML
        chart_html += f'<img src="data:image/png;base64,{chart_image}" style="margin-bottom: 20px;" />'

        # Close the chart
        plt.close(fig)

    # Combine the table and chart HTML
    output_html = table_html + chart_html


    # Return the combined HTML string
    return output_html