# DemandDynamics
![image](https://github.com/RahulAnkola/DemandDynamics/assets/77481473/29d162da-face-4d16-9e5a-0f43e669b3ec)
![image](https://github.com/RahulAnkola/DemandDynamics/assets/77481473/41c56406-cc81-4bae-ae72-997624a81c37)

Traditional grocery store inventory management lacks accuracy and adaptability, leading to waste and lost sales. The absence of a Deep Learning solution analysing historical sales data hampers optimal stock levels, profitability, and customer satisfaction. To overcome this problem, we are building a model that gives retailers:
- the capacity to save costs
- boost profits
- provides customers a smooth shopping experience by precisely forecasting demand


This model is a combination of a Convolutional Neural Network (CNN) and a Multi-Layer Perceptron (MLP). It uses a 1D Convolutional layer and a Max Pooling layer, which are characteristic of CNNs, to extract features from the input sequence. After these layers, the Flatten layer transforms the 2D output of the previous layers into a 1D array, which can then be fed into the Dense layers. The Dense layers form a Multi-Layer Perceptron that can learn from the extracted features to make the final prediction.
![image](https://github.com/RahulAnkola/DemandDynamics/assets/77481473/b7e1b511-d93b-4f9d-8dfd-702ae361f088)
![image](https://github.com/RahulAnkola/DemandDynamics/assets/77481473/01e5c75f-17a3-4474-8a0a-3df53b7d77e2)

Accuracy = 100 - MAPE (Mean Absolute Percentage Error)

Steps to run:
1. Download required dependencies: pip install -r requirements.txt
2. Run app.py
3. Open browser: http://127.0.0.1:5000/

![image](https://github.com/RahulAnkola/DemandDynamics/assets/77481473/6921f68e-ea0e-40d0-bb52-296595d5964f)
