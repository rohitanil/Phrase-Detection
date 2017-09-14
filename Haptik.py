
import pandas as pd
import re
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import pickle
import warnings
warnings.filterwarnings("ignore")
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


#Message preprocessing method
def msg_processing(raw_msg):
    words=raw_msg.lower().split()
    stops=set(stopwords.words("english"))
    m_w=[w for w in words if not w in stops]
    wnl=WordNetLemmatizer()
    return (" ".join([wnl.lemmatize(i) for i in m_w]))

def dataBuilder():
    df= pd.read_csv("/Users/continuumlabs/Downloads/Assignment_2017/training_data.tsv",sep='\t', header=0)
    df.columns = ["message", "label"]
    #Convert message column to string
    df['message'] = df['message'].astype(str)
    #utf-8 encoding
    df.message.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

    num_msg=df["message"].size
    clean_msg=[]
    for i in range(0,num_msg):
        clean_msg.append(msg_processing(df["message"][i]))

    df['Processed_Msg']=clean_msg
    df['Processed_Msg'] = df['Processed_Msg'].str.replace('\d+', '')
    cols=['Processed_Msg','label']

    #Filtering rows having labels whose frequency is lesser than 2
    filtered = df.groupby('label').filter(lambda x: len(x) > 2)
    filtered=filtered[cols]

    X_train=filtered["Processed_Msg"][:4000]
    Y_train=filtered["label"][:4000]
    X_test=filtered["Processed_Msg"][4000:4400]
    Y_test=filtered["label"][4000:4400]
    vectorizer=CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

    #Converting train and test data to feature matrices
    train_data_features=vectorizer.fit_transform(X_train)
    train_data_features=train_data_features.toarray()
    test_data_features=vectorizer.transform(X_test)
    test_data_features=test_data_features.toarray()

    return X_train,train_data_features,Y_train,test_data_features,Y_test


def modelBuilder(train_data_features,Y_train,test_data_features,Y_test):
    #Support Vector Model with linear kernel and class_weight= 'balanced' due to unbalanced dataset
    clf=svm.SVC(kernel='linear',C=10,class_weight='balanced')
    print "Training Model!!"
    clf.fit(train_data_features,Y_train)
    print "Training Completed!!"
    print "Creating Pickle File!!"
    filename='final_model.sav'
    pickle.dump(clf,open(filename,'wb'))
    print "Pickle Created!!"

    #Metrics
    predicted=clf.predict(test_data_features)
    accuracy=clf.score(test_data_features,Y_test)
    print "Accuracy: ",accuracy
    print "f1 score ",f1_score(Y_test, predicted, average="macro")

 

if __name__=='__main__':
    x,xtrain,ytrain,xtest,ytest = dataBuilder()
    modelBuilder(xtrain,ytrain,xtest,ytest)


 
   
    






