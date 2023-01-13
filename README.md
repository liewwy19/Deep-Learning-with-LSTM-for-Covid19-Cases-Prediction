
# Deep Learning with LSTM for COVID-19 Cases Prediction

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

## Summary
This project implement Long short-term memory (LSTM) algorithm for COVID-19 cases prediction in python.

## Abstract
The year 2020 was a catastrophic year for humanity. Pneumonia of unknown 
aetiology was first reported in December 2019., since then, COVID-19 spread to 
the whole world and became a global pandemic. More than 200 countries were 
affected due to pandemic and many countries were trying to save precious lives 
of their people by imposing travel restrictions, quarantines, social distances, event 
postponements and lockdowns to prevent the spread of the virus. However, due 
to lackadaisical attitude, efforts attempted by the governments were jeopardised, 
thus, predisposing to the wide spread of virus and lost of lives. 

The scientists believed that the absence of AI assisted automated tracking and 
predicting system is the cause of the wide spread of COVID-19 pandemic. Hence, 
the scientist proposed the usage of deep learning model to predict the daily 
COVID cases to determine if travel bans should be imposed or rescinded

## Data Set
This model use [public COVID-19 data repository](https://github.com/MoH-Malaysia/covid19-public) provided in the [Official Github account of Malaysia's Ministry of Health](https://github.com/MoH-Malaysia). A subset of 680 days of data using as train dataset and additional 100 days of data using as test dataset are provided in the 'Datasets' folder within this repository.

## The Model
With the Sequntial API from Tensorflow Keras, this model is constructed using 3 set of LSTM layers with a Dropout layer follow immediately to reduce the overfitting. Each LSTM layers is constructed with 64 nodes. 

Since this is a regression model, mean squared error (MSE) is choosen as the loss function. 'Adam' optimizer is used as it is straightforward to implement and computationally efficient with little memory requirements.


![](https://github.com/liewwy19/Deep-Learning-with-LSTM-for-Covid19-Cases-Prediction/blob/main/model_summary.png?raw=True)

![](https://miro.medium.com/max/720/1*7cMfenu76BZCzdKWCfBABA.webp)

Image credit: Saul Dobilas on medium

## Analysis
Overall, the model did able to predict the trend quite well. Deep learning predition model like this one does play an important role in the battle with COVID-19. 

Due to the unprecedented natural of this COVID-19 pandemic, a lots of external factor affecting the outcomes of the results, therefore, no concrete conclusion can be made with just solely analyzing using just a single feature. 

![](https://github.com/liewwy19/Deep-Learning-with-LSTM-for-Covid19-Cases-Prediction/blob/main/chart_actual_vs_predicted.png?raw=True)

## Results
| MAPE | MAE | RMSE |
| --- | ---- | --- |
|  0.184176  |  2167.01 | 3467.39 |


## Contributing

This project welcomes contributions and suggestions. 

    1. Open issues to discuss proposed changes 
    2. Fork the repo and test local changes
    3. Create pull request against staging branch


## Acknowledgements

 - [GitHub - MoH-Malaysia/covid19-public: Official data on the COVID-19 epidemic in Malaysia. Powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera.](https://github.com/MoH-Malaysia/covid19-public)
 - [LSTM Recurrent Neural Networks — How to Teach a Network to Remember the Past](https://towardsdatascience.com/lstm-recurrent-neural-networks-how-to-teach-a-network-to-remember-the-past-55e54c2ff22e)
 - [Gentle Introduction to the Adam Optimization Algorithm for Deep Learning](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
 - [Selangor Human Resource Development Centre (SHRDC)](https://www.shrdc.org.my/)

