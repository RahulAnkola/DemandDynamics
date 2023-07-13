import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt

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
data_path = r".\train_3yr_new.csv"  # Update with your file path
data = pd.read_csv(data_path)

# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

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
    model.fit(X_train, y_train, epochs=1, verbose=1)

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

# Plot the predicted and actual sales for each item
'''
for item, data in sales_data.items():
    predicted = data['Predicted']
    actual = data['Actual']

    # Create a DataFrame from the predicted and actual sales
    df = pd.DataFrame({'Predicted': predicted, 'Actual': actual})

    # Plot the DataFrame
    df.plot(kind='bar', figsize=(10, 4))
    plt.title(f'Actual vs Predicted Sales for Item {item} for the "Future" 90 Days')
    plt.xlabel('Day')
    plt.ylabel('Sales')
    plt.show()
'''

# In[2]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Calculate accuracy metrics for each item
for item, data in sales_data.items():
    predicted = data['Predicted']
    actual = data['Actual']
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    
    print(f'Item: {item}')
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}\n')


# In[3]:


# Define evaluation function
def evaluate_accuracy(error):
    if error <= 10:
        return 'Very High'
    elif error <= 20:
        return 'High'
    elif error <= 30:
        return 'Medium'
    elif error <= 40:
        return 'Low'
    else:
        return 'Very Low'

# Create a DataFrame to store the accuracy metrics and levels
accuracy_df = pd.DataFrame(columns=['Item', 'MAE', 'MAE Level', 'MSE', 'MSE Level', 'RMSE', 'RMSE Level'])

# Calculate accuracy metrics for each item
for item, data in sales_data.items():
    predicted = data['Predicted']
    actual = data['Actual']
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    
    accuracy_df = accuracy_df._append({
        'Item': item,
        'MAE': mae,
        'MAE Level': evaluate_accuracy(mae),
        'MSE': mse,
        'MSE Level': evaluate_accuracy(mse),
        'RMSE': rmse,
        'RMSE Level': evaluate_accuracy(rmse)
    }, ignore_index=True)

print(accuracy_df)



from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Calculate MAPE for each item
mape_values = []
for item, data in sales_data.items():
    predicted = data['Predicted']
    actual = data['Actual']
    
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    mape_values.append(mape)

# Create a DataFrame to store the MAPE values
mape_df = pd.DataFrame({'Item': list(sales_data.keys()), 'MAPE': mape_values})

# Plot the MAPE values
plt.figure(figsize=(10, 6))
plt.plot(mape_df['Item'], 100 - mape_df['MAPE'], marker='o')
plt.xlabel('Item')
plt.ylabel('Accuracy (%)')
plt.title('Forecast Accuracy Over Time')
plt.grid(True)
plt.show()


# In[5]:


from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Calculate MAPE for each item
mape_values = []
for item, data in sales_data.items():
    predicted = data['Predicted']
    actual = data['Actual']
    
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    mape_values.append(mape)

# Calculate the overall accuracy
overall_accuracy = 100 - np.mean(mape_values)

# Create a DataFrame to store the MAPE values
mape_df = pd.DataFrame({'Item': list(sales_data.keys()), 'MAPE': mape_values})

# Plot the MAPE values
plt.figure(figsize=(10, 6))
plt.plot(mape_df['Item'], 100 - mape_df['MAPE'], marker='o')
plt.axhline(y=overall_accuracy, color='r', linestyle='--')
plt.xlabel('Item')
plt.ylabel('Accuracy (%)')
plt.title('Forecast Accuracy Over Time')
plt.text(0, overall_accuracy + 2, f'Average Accuracy: {overall_accuracy:.2f}%', color='r')
plt.grid(True)
plt.show()