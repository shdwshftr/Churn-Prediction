import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle as pkl

class model:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

    def preprocess_data(self):
        self.df = self.df.drop(columns = ['Unnamed: 0', 'CustomerId', 'id', 'Surname'])
        self.df['CreditScore'].fillna(self.df['CreditScore'].median(), inplace = True)

    def encode_scale(self):
        self.encode_binary_features()
        self.encode_label_features()
        self.scale_numerical_features()

    def encode_binary_features(self):
        binary_encode = {"Gender": {"Male": 1, "Female": 0}}
        self.x_train = self.x_train.replace(binary_encode)
        self.x_test = self.x_test.replace(binary_encode)
        self.x_val = self.x_val.replace(binary_encode)
        
        self.binary_encode = binary_encode
        
    def encode_label_features(self):
        label_encode = {'Geography': {"France":0, "Germany":1, "Spain":2}}
        self.x_train = self.x_train.replace(label_encode)
        self.x_test = self.x_test.replace(label_encode) 
        self.x_val = self.x_val.replace(label_encode)

        self.label_encode = label_encode

    def scale_numerical_features(self):
        scaler = RobustScaler()
        self.x_train = scaler.fit_transform(self.x_train[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                             'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']])
        self.x_test = scaler.transform(self.x_test[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                             'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']])
        self.x_val = scaler.transform(self.x_val[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                             'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']])

        self.scaler = scaler

    def train_test_split(self):
        input_df = self.df.drop('churn', axis = 1)
        output_df = self.df['churn']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(input_df, output_df, test_size = 0.2,
                                                                                stratify = output_df, random_state = 42)
        self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(self.x_test, self.y_test, test_size = 0.5,
                                                                            stratify = self.y_test, random_state = 42)

    def train_random_forest(self):
        self.rf_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = 10)
        self.rf_classifier.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        y_predict_rf = self.rf_classifier.predict(self.x_test)
        print('\nClassification Report Random Forest:\n')
        print(classification_report(self.y_test, y_predict_rf, target_names=['0', '1']))

    def save_model_artifacts(self, model_filename = 'RF_model.pkl', scaler_filename = 'scaler.pkl',
                             binary_encode_filename = 'binary_encode.pkl', label_encode_filename = 'label_encode.pkl'):
        pkl.dump(self.rf_classifier, open(model_filename, 'wb'))
        pkl.dump(self.scaler, open(scaler_filename, 'wb'))
        pkl.dump(self.binary_encode, open(binary_encode_filename, 'wb'))
        pkl.dump(self.label_encode, open(label_encode_filename, 'wb'))

    def visualize_data(self):
        self.df.boxplot(column = ['CreditScore'])
        plt.title('Boxplot of Credit Scores')
        plt.ylabel('Credit Score')
        plt.show()

        churn_count = self.df['churn'].value_counts()
        plt.figure(figsize = (6, 6))
        plt.pie(churn_count, labels = churn_count.index, autopct='%1.1f%%')
        plt.title('Churn Distribution')
        plt.show()

        self.x_train.hist(figsize = (12, 10))
        plt.show()

        fig, axs = plt.subplots(1, 4, figsize = (15, 5))
        numerical_columns = ['CreditScore', 'Age', 'Balance', 'NumOfProducts']

        for i, col in enumerate(numerical_columns):
            axs[i].boxplot(self.x_train[col])
            axs[i].set_title(col)
            axs[i].set_ylabel('Value')

        plt.show()

        country_counts = self.x_train['Geography'].value_counts()
        gender_counts = self.x_train['Gender'].value_counts()

        fig, axs = plt.subplots(1, 2, figsize = (10, 5))

        axs[0].pie(country_counts, labels = country_counts.index, autopct = '%1.1f%%')
        axs[0].set_title('Country Distribution')

        axs[1].pie(gender_counts, labels = gender_counts.index, autopct = '%1.1f%%')
        axs[1].set_title('Gender Distribution')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    churn_model = model('C:/BINUS/SEM_4/Model Deployment/UTS/2602075491_BrandonRitchieYang_UTS_ModelDeployment/data_A.csv')
    churn_model.preprocess_data()
    churn_model.train_test_split()
    churn_model.visualize_data()
    churn_model.encode_scale()
    churn_model.train_random_forest()
    churn_model.evaluate_model()
    churn_model.save_model_artifacts()

