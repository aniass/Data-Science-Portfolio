# Data Science Projects Portfolio

The repository contains my data science, data analysis, SQL databases and python programming projects which show my all self-study progress. Some projects are still in progress.
In portfolio includes projects from:
- **Data Analysis** (data visualization, data cleaning, and data exploration) with python and SQL. 
- **Machine Learning** (supervised & unsupervised) such as linear regression, classification, prediction, recommendation, customer segmentation and anomaly detection.
- **Natural Language Processing**: text classification, sentiment analysis and text summarization.
- **Deep Learning/Computer Vision**: image recognition.
- **Python projects**: web applications with Flask and  Streamlit, simple pipeline with python, automating with python; 
- **SQL and Python projects**: ETL process, basic CRUD.


## Projects:

## Machine learning:
**ML supervised & unsupervised**:

### [Advertisement click prediction](https://github.com/aniass/ad-click-prediction)
The project concerns prediction of the advertisement click using the machine learning. The main aim of this project is predict who is going to click ad on a website in the future. The analysis includes data analysis, data preparation and creation model by different machine learning models.
- **Models used:** Logistic Regression, Linear SVC, Decision Tree, Random Forest, AdaBoost.
- **Keywords**: Ad click prediction, Python: pandas, scikit-learn, seaborn, matplotlib.

### [Churn prediction](https://github.com/aniass/Churn-prediction)
The project concerns churn prediction in the bank customers. It includes data analysis, data preparation and create model by using different machine learning algorithms to predict whether the client is going to leave the bank or not.
- **Models used:** Logistic Regression, Random Forest, KNN, SVC, XGBoost;
- **Keywords**: Churn prediction, Python: pandas, scikit-learn, seaborn, matplotlib, xgboost.

### [Books Recommendation System](https://github.com/aniass/books-recommender-system)
The project concerns the books recommendation system. It includes data analysis, data preparation and build model by using colaborative filtering and matrix factorization to get books recommendations.
- **Models used:** KNN, colaborative filtering, matrix factorization;
- **Keywords**: recommendation system , python: pandas, scikit-learn, seaborn, matplotlib.

### [Customer segmentation ](https://github.com/aniass/Customer-segmentation)
The project contains customer segmentation by using the RFM method (RFM score) and K-Means clustering for creating customer segments based on data provided.
- **Models used:** K-Means, RFM method;
- **Keywords**: RFM, K-Means clustering, Python: pandas, scikit-learn, scipy, matplotlib. 

### [Fraud Detection](https://github.com/aniass/Fraud-Detection)
The project concerns the anomaly detection in credit cards transactions using machine learning models and Autoencoders. The main aim of this project is predict whether a given transaction was a fraud or not.
- **Models used:** Isolation Forest, Local Outlier Factor, Support Vector Machine (OneClassSVM), Autoencoder.
- **Keywords**: Anomaly detection, Python: pandas, scikit-learn, tensorflow, seaborn, matplotlib.

### [Sales forecasting](https://github.com/aniass/Sales-forecasting)
The project concerns sales forecasting by using time series model. The project includes sales data analysis and forecast of the number of orders by using Prophet library.
- **Models used:** Time Series.
- **Keywords**: Exploratory data analysis, Prophet, python.

### [Real Estate price prediction](https://github.com/aniass/Real-Estate-price-prediction)
The project concerns real estate price prediction using linear regression models. I have build a model which predict real estate price based on historical data.
- **Models used:** Ridge, Lasso, Elastic Net, Random Forest, Gradient Descent, XGBoost.
- **Keywords**: Linear regression, Python: pandas, numpy, scikit-learn.

## Natural Language Processing:

### [Product Categorization](https://github.com/aniass/Product-Categorization-NLP)
The project concerns product categorization (make-up products) based on their description. I have build multi-class text classification model (with ML algorithms, MLP, CNN and Distilbert model) to predict the category (type) of a product. From the data I also have trained Word2vec and Doc2vec model and created Topic Modeling and EDA analysis.

- **Models used:** MLP, CNN, Distilbert, Logistic Regression, SVM, Naive Bayes, Random Forest; Word2vec, Doc2vec.
- **Keywords**: NLP, text classification, transformers, topic modeling; Python: nltk, gensim, scikit-learn, keras, tensorflow, Hugging Face, LDA.

### [Text Summarization](https://github.com/aniass/text-summarizer)
Text summarization based on extractive and abstractive methods by using python. The analysis includes text summary by calculating word frequency with spacy library, TFIDF vectorizer implementation, automatic text summarization with gensim library and abstractive techniques by using Hugging Face library.

- **Models used:** word frequency, TFIDF vectorizer, BART.
- **Keywords**: text summarization, transformers, BART, Python: spacy, nltk, scikit-learn, gensim.

### [Spam detection](https://github.com/aniass/Spam-detection)
The project concerns spam detection in SMS messages to determine whether the messages is spam or not. I have build model by using pretrained BERT model and different machine learning algorithms. The analysis includes also text mining with NLP methods to prepare and clean data.

- **Models used:** BERT, Logistic Regression, Naive Bayes, SVM, Random Forest.
- **Keywords**: NLP, transformers, spam detection, smote sampling, Python: nltk, scikit-learn, Hugging Face, imbalanced-learn.

### [Sentiment analysis reviews](https://github.com/aniass/Sentiment-analysis-reviews)
The project concerns sentiment analysis of women's clothes reviews. I have built model to predict if the review is positive or negative. I have used different machine learning algorithms and a pre-trained Glove word embeddings with Bidirectional LSTM. The project also includes EDA analysis and sentiment analysis by using Vader and TextBlob methods.

- **Models used:** LSTM, Glove, Logistic Regression, Naive Bayes, SVM.
- **Keywords**: NLP, sentiment analysis, TextBlob, Vader; Keras, TensorFlow, nltk, scikit-learn, pandas.

## Computer vision/Image processing:

### [Plant pathology](https://github.com/aniass/Plant-pathology)
The project concerns recognition diseases on apple leaves based on their images. The solution includes data analysis, data preparation, CNN model with data augmentation and transfer learning to recognition of leaves diseases.

- **Models used:** Convolutional Neural Network, MobileNet V2.
- **Keywords**: Image Recognition, data augumentation, transfer learning; Python: tensorflow, keras, pandas, numpy, scikit-learn, seaborn, pillow, opencv.

### [Waste Classification](https://github.com/aniass/Waste-Classification)
The project concerns waste classification to determine if it may be recycle or not. In the analysis I have used Convolutional Neural Network (CNN) model with data augumentation and transfer learning with pre-trained MobileNet V2 model.

- **Models used:** Convolutional Neural Network.
- **Keywords**: Image Recognition, data augumentation, Python: tensorflow, keras, numpy, matplotlib.

### [Face Detection](https://github.com/aniass/Face-Detection-with-OpenCV)
In the project I have used OpenCV library to detect faces, eyes and smile in an image.

- **Models used:** OpenCV: Harr Classifier.
- **Keywords**: Face detection, Python: OpenCV, pillow, numpy, matplotlib.

## Data analysis:

### [Market Basket analysis](https://github.com/aniass/Market-basket-analysis)
The project concerns market basket analysis and product recommendation by using the association methods. I have build model by using the Apriori algorithm to products recomendation based on our data.
- **Models used:** Apriori algorithm.
- **Keywords**: product recomendation, data analysis, python, MLxtend.

### [World happiness reports analysis](https://github.com/aniass/world-happiness-report-analysis)
The project includes world happiness analysis over 5 years (2015-2019). For analysis I have used SQL (SQLite) and python.

- **Keywords**: data analysis, SQL, Python: SQLite3, pandas, matplotlib, seaborn.

### [IT job market analysis](https://github.com/aniass/IT-job-market-analysis)
The project concerns the analysis of the IT job market using data from GitHub, StackOverflow and Web scraping data. I have used SQL, Google Big Query and Python (pandas, numpy, matplotlib, seaborn) to analyze the data.

- **Keywords**: data preprocessing, data cleaning, EDA, Python: pandas, numpy, seaborn; SQL, Google BigQuery. 

### [Air quality analysis](https://github.com/aniass/Air-quality-analysis)
The project includes data analysis and outliers detection of air quality data. The outliers detection have been made with a few methods such as Tukey's method (IQR) and Isolation Forest algorithm.

- **Models used:** Isolation Forest.
- **Keywords**: data analysis, outliers detection, Python: pandas, numpy, scikit-learn, seaborn.

### [Sales Dashboard](https://github.com/aniass/sales-dashboard)
The project allows to build interactive dashboard from sales data by using pandas-bokeh library.

- **Keywords**: data analysis, data visualization, dashboard, python, pandas, pandas-bokeh.

## Python projects

### [Sentiment analysis app](https://github.com/aniass/sentiment-app)
The REST API Web App for Sentiment analysis of clothes reviews by using Flask and Machine Learning model. 

- **Keywords**: Flask, HTML, Python: pandas, scikit-learn, regex, nltk.

### [Waste app](https://github.com/aniass/Waste-app)
It is Streamlit application with using a Deep Learning model to determine if a given waste are recycle or organic. I have used a previous trained CNN (Convolutional Neural Networks) algorithm to detect waste.

- **Keywords**: python, streamlit, tensorflow, pillow.

### [Excel report](https://github.com/aniass/excel-report)
Automating the Excel report with python and openpyxl library. 
- **Keywords**: python, openpyxl, pandas.

### [CSV Report Processing](https://github.com/aniass/CSV_Report_Processing)
This Python script allows to read a CSV file entered by the user, changes the data contained in it and returns the transformed data as a new CSV one.

- **Keywords**: python, pycountry, csv.

### [Extracting data using API](https://github.com/aniass/Extracting-data-using-API)
In the project I have used the API to get the data and create a dataset. I have created two examples of get the data from an API. The data received was saved in json format and they were exported to a csv file.

- **Keywords**: python, pandas, requests, json.

## SQL and Python projects
### [ETL in python and SQLite](https://github.com/aniass/ETL-python-SQLite)
The project includes a simple ETL process using Python and SQLite database. This pipeline allows to match reported chargebacks (Excel file) with transactions from the database.
- **Keywords**: ETL, python, SQLite, pandas.

### [CRUD in python and SQLite](https://github.com/aniass/crud-sqlite3)
The script allows to make a basic crud operations by using python and SQLite3. 

- **Keywords**: python, SQLite.

