# IMPORT PACKAGES

# !pip install textblob
# !pip install sklearn

# Usefull library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm import tqdm_notebook
from math import floor, ceil
import json
import warnings
import io

# text library
import csv
import sys
import spacy
import re
# from langdetect import detect
"""!python -m spacy download en_core_web_sm
nlp_en = spacy.load('en_core_web_sm')"""
# import en_core_web_sm
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import remove_stopwords
import gensim
import nltk
from textblob import TextBlob
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('vader_lexicon')
nltk.download('punkt')

# Machine Learning library
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from lightgbm import LGBMRegressor, LGBMClassifier

# Deep Learning library
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM, Embedding
from keras.optimizers import Adam
from keras.layers import SpatialDropout1D, Dropout, Bidirectional, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping


# TEXT PRETREATMENT

# JOIN PROCESSING

def droping_data(df: pd.core.frame.DataFrame,
                 percentage_no_missing: int) -> pd.core.frame.DataFrame:
    """ Documentation
        Parameters:
            param1: dataframe
            param2: percentage of missing value

        Out : dataframe without the columns with a percentage
        of missing data above the threshold
    """
    limitPer = len(df) * percentage_no_missing  # threshold calculation
    return df.dropna(thresh=limitPer, axis=1)

def detect_sentences(comment):
    """Function to cut the entire comment into sentences
    
    Parameters : 
        comment : a comment you want to be sliced into sentences
        
    Attributes :
        tokenizer : english model to be able to cut the comment
        sentences : list of sentences of the comment
        
    Out : sentences
    
    """
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(str(comment))
    return sentences

def duplicate(df_, column_name):
    """Function which duplicates comments times its number of sentences and 
    associates them with each of the sentences

    Parameters : 
        df : matrix containing all comments imported from xlsx

    Attributes :
        Data : all sentences that are associated with its original comment
        listSentence : list of sentences
        listeReview : list of comments
        listeIndex : list of index of the original comment

    Out : Data
    """

    Data = pd.DataFrame()
    listSentence = []
    listeReview = []
    listeIndex = []
    listeScore = []
    try:
        df_['index'] = [i for i in range(len(df_))]
    except:
        print("Index already exist")

    for i in tqdm_notebook(df_.index):
        try:
            comments = detect_sentences(df_[column_name][i])
            for sent in range(len(comments)):
                listSentence.append(comments[sent])
                listeReview.append(df_[column_name][i])
                listeIndex.append(i)
                listeScore.append(df_["Cabin_Staff_Service_3class"][i])
        except:
            listSentence.append("")
            listeReview.append("")
            listeIndex.append(i)
            listeScore.append(df_["Cabin_Staff_Service_3class"][i])

    Data["index"] = listeIndex
    Data[column_name] = listeReview
    Data["Sentence"] = listSentence
    Data["Score"] = listeScore

    return Data

def database(file: pd.core.frame.DataFrame,
             column: str) -> pd.core.frame.DataFrame:
    """ Documentation
        Parameters:
            param1: dataframe
            param2: string of the column name

        Out : dataframe with index and only column name
    """
    file['index'] = file.index  # create index
    sentences = file.loc[:, lambda file: ["index", column]]
    return sentences


def clean_data(sentences: pd.core.frame.DataFrame, column: str) -> list:
    """ Documentation
        Parameters:
            param1: dataframe
            param2: string of the column name

        Out : list of sentences clean
    """
    nlp_en = spacy.load('en_core_web_sm')  # load spacy english vocabulary

    sentences_clean = []

    for i in sentences["index"]:
        # delete "points négatifs et positifs"
        sentence = str(sentences[column][i]).replace('Points positifs', ' ').replace(
            'Points négatifs', ' ') 
        # delete "Verified"
        sentence = sentence.replace('Trip Verified', ' ').replace(
            'Not Verified', ' ').replace('Verified Review', ' ')
        # delete link
        sentence = re.sub(
            r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', sentence)  
        # find hashtag and doubled
        sentence = sentence + ' ' + ' '.join(re.findall(r"#(\w+)", sentence))
        sentence = sentence + ' ' + \
            ' '.join(re.findall(r"@(\w+)", sentence))  # find @ and doubled
        sentence = strip_punctuation(sentence)  # delete punctuation
        semi_clean_sentence = ''
        comments = nlp_en(sentence.lower())  # lower comments
        if len(comments) != 0:  # no-empty comments
            try:
                if detect(str(comments)) == 'en':  # english comments
                    for token in comments:
                        # add lemmatizer
                        semi_clean_sentence = semi_clean_sentence + token.lemma_ + ' '  
                    semi_clean_sentence = semi_clean_sentence.replace(
                        '-PRON-', '')  # delete "PRON"
                    semi_clean_sentence = remove_stopwords(
                        strip_short(semi_clean_sentence))  # delete short words
                    sentences_clean.append([i, semi_clean_sentence])
            except:
                continue
        print(str(i) + '/' + str(len(sentences)), end="\r")
    return sentences_clean


def create_dataframe(dataframe: pd.core.frame.DataFrame, column: str,
                     version="0") -> pd.core.frame.DataFrame:
    """ Documentation
        Parameters:
            param1: list of sentences clean
            param1: list of sentences clean
            param2: list of sentences clean

        Out : dataframe of tf_idf
    """
    for i in tqdm(column):
        df = pd.DataFrame()
        df = dataframe.join(data[i], how='left')
        df[i] = df[i].replace(' ', np.nan)
        df = df.dropna(subset=[i])
        df[i] = df[i].replace(' N/A', np.nan)
        df = df.dropna(subset=[i])
        df.to_csv(r'Dataframe/df_' + str(i)+'_V'+str(version)+'.csv')
        print('DataFrame '+str(i)+' correctly create')


# TFIDF

def create_tfidf(sentences_clean: list) -> pd.core.frame.DataFrame:
    """ Documentation
        Parameters:
            param1: list of sentences clean

        Out : dataframe of tf_idf
    """
    comments = [i[1] for i in sentences_clean]  # recover comments
    index = [i[0] for i in sentences_clean]  # recover index

    vectorizer = TfidfVectorizer(
        stop_words="english", min_df=0.005, max_df=0.7)
    X = vectorizer.fit_transform(comments)

    # creation of matrice
    M = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

    tfidf = np.concatenate((pd.DataFrame(index), pd.DataFrame(
        comments), M), axis=1)  # add comments to tfidf
    col = vectorizer.get_feature_names()
    col = ['index', 'commentaire'] + col  # rename columns

    return(pd.DataFrame(tfidf, columns=col).set_index('index'))


# WORD EMBEDDING
def coordinates(comments):
    """Documentation

    Parameters :
        comments (DataFrame) : All cleaned comments

    Attributes:
        coordinates (Array) : All coordinates of comments
        coordinate(Array): coordinates of one comment
        counter ( integer): Count number of words in a comment

    Out :
        coordinates (Array) : All coordinates of comments
    """

    coordinates = np.array([[0]*201]) #Initializes array of coordonates 

    for i in comments['index']:
        #Create array of coordonates for one comment 
        coordinate = np.array([[0]*200]) 
        counter = 0
        try:
            #Iterate through the comments 
            for word in comments['commentaire'][i].split(): 
                try:
                     #Add coordinates for each word 
                    coordinate = coordinate + np.array([dico[word]])
                    counter += 1
                except:
                    word
            coordinate = np.concatenate((np.array([[i]]), coordinate/counter)
            , axis=1) #Coordinates of comment 
            if (i==0):
                coordinates = coordinate
            else:
                # Add coordonates of one comment to the array of coordinates
                coordinates = np.concatenate((coordinates, coordinate),
                                             axis=0)
        except:
            i
        print(str(i) + '/' + str(len(comments)), end="\r")
    return coordinates


def matrix(coordinates, comments):

    """Documentation

    Parameters :
        comments (DataFrame) : All clean comments
        coordinates (Array) : All coordinates of comments

    Attributes:
        coordinates (Array) : All coordinates of comments
        liste(list):columns's Names

    Out :
        embedding ( Array): Coordinates + comments
    """
    liste = ['index']
    for i in range(1, 201):
        liste.append(i)   # Create index list
    # Transfer array to dataFrame
    coordinates = pd.DataFrame(coordinates, columns=liste)
    # Join the comments to the coordinates
    embedding = comments.join(coordinates, on=['index'], how='inner',
                              lsuffix='_caller', rsuffix='_other')

    return embedding


# FEATURES

def nbpunctuation(data: pd.core.frame.DataFrame, punct: str,
                  columns: str) -> pd.core.frame.DataFrame:
    """ Documentation
        Calculation of the number of punctuation per commment

        Parameters:
            param1: dataframe used
            param2: variable whose punctuation number is to be counted
            param3: columns use for the count of punctuation

        Out : dataframe plus a column with the number of punctuation
        in the columns
    """
    df = pd.DataFrame(data)
    nbpunct = []
    for i in df.index:
        comment = df[columns][i]
        cpt = 0
        for j in range(len(comment)):
            if comment[j] == punct:
                cpt += 1
        nbpunct.append(cpt)
    df['nbpunct_' + punct] = nbpunct
    return df


def number_sentence(data: pd.core.frame.DataFrame,
                    column: str) -> pd.core.frame.DataFrame:
    """ Documentation
        Counts the number of sentences in comments

        Parameters:
            param1: dataframe
            param2: name of the text column

        Out : dataframe with the number of sentences per comment features
        in new colum
    """
    df = pd.DataFrame(data)
    sentence = df[column]
    count_sentence = []
    for i in sentence.index:
        sentence[i] = sentence[i].replace("''", "")
        count_sentence.append(len(sent_tokenize(sentence[i])))
    df['count_sentence'] = count_sentence
    return df['count_sentence']


def get_word_list(commentary: str, list_w: str) -> int:

    """ Documentation
        Counts the number of negative words for each commentary
        First, a list of negative words must be defined,
        here it is 'negative_words'

        Parameters:
            param1 : commentary of the dataframe
            
        Out : number of negative word per comment
    """
    tokens = commentary.split()
    number_negword = np.sum([tokens.count(i) for i in list_w])
    return number_negword


def get_number_word_commentary(data: pd.core.frame.DataFrame,
                               column: str) -> pd.core.frame.DataFrame:
    """ Documentation
        Calculation of the number of words per commment

        Parameters:
            param1: dataframe used
            param2: variable whose word number is to be counted

        Out : dataframe plus a column with the number of words in the columns
    """
    df = pd.DataFrame(data)
    size = []
    for i in df.index:
        comm = df[column][i].split()
        size.append(len(comm))
    df['nb_word_comment'] = size
    return df


def sentiment_analyse(df: pd.core.frame.DataFrame,
                      review: str) -> pd.core.frame.DataFrame:
    """ Documentation
        Sentiment analysis of the comments

        Parameters:
            param1: dataframe
            param2: name of the text column


        Out : dataframe with sentiment analysis features in new colums
        polarity and subjectivity from TextBlob module
        polarity (neg, neu, pos, compound) from Vader

    References: https://www.nltk.org/api/nltk.sentiment.html


    """

    df['polarity'] = df[review].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df[review].apply(
        lambda x: TextBlob(x).sentiment.subjectivity)

    sid = SentimentIntensityAnalyzer()
    df['sentiments'] = df[review].apply(lambda x: sid.polarity_scores(x))
    df = pd.concat([df.drop(['sentiments'], axis=1),
                    df['sentiments'].apply(pd.Series)], axis=1)
    return df


def freq_token_name(text: str, token_name: str) -> float:

    """Documentation

     Parameters:
     text: text or comment to tokenize
     token_name: name of the searched token

     Out:
     value: number of the searched token in the text
     """

    # Tokenize the text
    word_token = word_tokenize(text)
    text_tokenize = nltk.pos_tag(word_token)

    # Count the number of words corresponding to token_name and return the
    # corresponding frequency
    elements = [i[1] for i in text_tokenize]
    value = elements.count(token_name)*100/len(text_tokenize)

    return value


def freq_token(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """Documentation
        Give the frequency of each token type

       Parameter:
          param1: dataframe
          out : dataframe plus the columns frequency

    """

    list_columns = ['PRP', 'VBD', 'IN', 'NNP', 'CD', 'CC', 'RB', 'JJ', 'DT',
                    'NN', 'VBG', 'PRP$', 'VBN', 'TO', 'VB', 'NNS', 'MD',
                    'VBP', 'VBZ', '.']

    for i in list_columns:
        df["Freq_" + i] = [freq_token_name(x, i) for x in df['Review']]


# OTHER FONCTION


def convert_values(liste_element: np.ndarray):
    """ Documentation
        Parameter : array of numbers
        Purpose : Convert numbers <= 1 to the minimal mark allowed (1)
    """

    for i in range(len(liste_element)):
        if liste_element[i] < 1:
            liste_element[i] = 1


def round_nb(numbers: np.ndarray):
    """ Documentation

       Purpose:
          - round each float number of an array, to the upper integer
          if the decimal part is >=0.5, to the lower integer if not

        Parameters:
            param1: array of numbers

        Out : the rounded columns

    """
    for i in range(0, len(numbers)):
        if (numbers[i] % 1 >= 0.5):
            numbers[i] = ceil(numbers[i])
        else:
            numbers[i] = floor(numbers[i])


def gather_rating_3class(df: pd.core.frame.DataFrame,
                         name: str) -> pd.core.frame.DataFrame:
    """Documentation
       Purpose:
          - Transform values >= 4 in 1, values <=2 in -1 ans ==3 in 0
          to give a new notation in tree classes

        Parameters:
            param1: dataframe
            param2: column to transform

        Out : the dataframe plus a columns with the new notation
    """
    df[name + '_3class'] = [np.nan]*len(df)
    for index, row in df.iterrows():
        if row[name] >= 4:
            df.loc[index, name + '_3class'] = 1  # good
        elif row[name] <= 2:
            df.loc[index, name + '_3class'] = -1  # bad
        elif row[name] == 3:
            df.loc[index, name + '_3class'] = 0  # neutral
            
            
def stratify_data(column_name: str, name_dataset: pd.core.frame.DataFrame,
                  lenght: int)-> pd.core.frame.DataFrame:
    """
        Documentation
              Parameters:
                     column_name: the column created on wich we want to stratify
                     name_dataset: the loaded dataset above
                           example :dfe = pd.read_csv(path)
                     length:length of modality of  variable
              out:
                   Stratified DataFrame with selected lenght
    """
    # liste of each name_dataset[column_name]'s element taken one time (unique)
    list_values = name_dataset[column_name].unique()
    # new empty DataFrame
    df = pd.DataFrame()
    for elt in list_values:
        df = df.append(name_dataset[name_dataset[column_name] == elt][:lenght])
    return df

# MODELS

def ML_train_test(data: pd.core.frame.DataFrame, tsize: float,
                  column: str, algorithm: str):

    """ Documentation
        preparation train and test set and apply machine learning algorithm to
        the dataset

        Parameters:
            param1: dataframe used
            param2: allows the dataset to be separated in a given proportion
            param3: columns use for the train and test
            param4: algorithm choose for apply

        Out : train_X, train_y, test_y, Ypred
    """

    # Separation training variables and variable to predict
    y = data[column]
    x = data.drop([column], axis='columns')

    # Scale X dataset
    scaler = MinMaxScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    # Separation train and test set
    train_X, test_X, train_y, test_y = train_test_split(x, y,
                                                        train_size=1-tsize,
                                                        test_size=tsize)
    # Apply algorithm to datasets
    algorithm.fit(train_X, train_y)
    Ypred = algorithm.predict(test_X)
    return train_X, test_X, train_y, test_y, Ypred


# MODELS EVALUATION

def evaluation(model, X_train: pd.core.frame.DataFrame,
               y_train: pd.core.series.Series,
               y_test: pd.core.series.Series, y_pred: pd.core.frame.DataFrame,
               type_mod: str, cv=5):
    """
        Documentation
        Parameters:
            model: machine learning model
            X_train: Train samples
            y_train: True labels for X_train
            y_test: True labels for X_test
            y_pred: Predict labels for X_test
            type_mod: type of model (possible values: regressor or classifier)
            cv: cross_validation splitting strategy

        Out:
          Evaluation of the cross_validation with train sample metrics : 
          Depends of type_mod parameter:
          regressor: RMSE
          classifer: precison, recall, F1 Score, Support
          
          Evaluation of the test prediction : 
          confusion_matrix
          metrics_test (precison, recall, F1 Score, Support)
          accuracy_test

    """
    cv = dict()

    # Cross Validation
    if type_mod == 'classifier':
        score = cross_validate(model, X_train, y_train, cv=5, scoring=[
            'accuracy', 'precision_weighted', 'recall_weighted',
            'f1_weighted'])
        cv['accuracy_cv'] = np.mean(score['test_accuracy'])
        cv['precision_cv'] = np.mean(score['test_precision_weighted'])
        cv['recall_cv'] = np.mean(score['test_recall_weighted'])
        cv['f1_score_cv:'] = np.mean(score['test_f1_weighted'])

    elif type_mod == 'regressor':
        score = cross_validate(model, X_train, y_train, cv=5,
                               scoring=['neg_mean_squared_error'])
        cv['RMSE'] = (
            np.mean(np.sqrt(-score['test_neg_mean_squared_error'])))

    # Round the prediction for regressor models
        y_pred = np.round(y_pred)
    for i in range(0, len(y_pred)):
        if y_pred[i] < -1:
            y_pred[i] = -1
        elif y_pred[i] > 1:
            y_pred[i] = 1

    metrics_cv = 'metrics_crossval: '+str(cv)

    # Test evaluation metrics
    confusion_mat = confusion_matrix(y_test, y_pred)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    metrics_test = 'metrics_test: ' + str(metrics['weighted avg'])
    accuracy_test = 'accuracy_test: '+str(accuracy_score(y_test, y_pred))

    return (print(metrics_cv), print(confusion_mat), print(metrics_test),
            print(accuracy_test))


# LIGHTGBM


def preprocessing_lgbm(dfe: pd.core.frame.DataFrame, name: str):
    """Documentation
        Parameters:
            dfe : Dataframe
            name : Name of variable to predict

        Out:
            X_train: Train samples
            y_train: True labels for X_train
            X_val: Validation samples
            y_val: True labels for Valiation
            X_test: Test samples
            y_test: True labels for X_test
    """
    df_lgbm = dfe
    kf = KFold(n_splits=2, shuffle=True, random_state=2)
    result = next(kf.split(df_lgbm), None)
    train_test_lgbm = df_lgbm.iloc[result[0]]
    val_lgbm = df_lgbm.iloc[result[1]]
    df_6k_train_test_lgbm = pd.DataFrame()
    df_6k_train_test_lgbm = df_6k_train_test_lgbm.append(
        train_test_lgbm[train_test_lgbm[str(name)] == -1][:2000])
    df_6k_train_test_lgbm = df_6k_train_test_lgbm.append(
        train_test_lgbm[train_test_lgbm[str(name)] == 0][:2000])
    df_6k_train_test_lgbm = df_6k_train_test_lgbm.append(
        train_test_lgbm[train_test_lgbm[str(name)] == 1][:2000])
    X_lgbm = df_6k_train_test_lgbm.drop([str(name)], axis=1)
    y_lgbm = df_6k_train_test_lgbm[str(name)]
    X_train_lgbm, X_test_lgbm, y_train_lgbm, y_test_lgbm = train_test_split(
        X_lgbm, y_lgbm, train_size=0.80, test_size=0.20, random_state=123)
    X_val_lgbm = val_lgbm.drop([str(name)], axis=1)
    y_val_lgbm = val_lgbm[str(name)]
    return (X_train_lgbm, X_test_lgbm, X_val_lgbm, y_train_lgbm,
            y_test_lgbm, y_val_lgbm)


def execute_lgbm_regressor(X_train: pd.core.frame.DataFrame,
                           y_train: pd.core.frame.DataFrame,
                           X_val: pd.core.frame.DataFrame,
                           y_val: pd.core.frame.DataFrame,
                           X_test: pd.core.frame.DataFrame,
                           y_test: pd.core.frame.DataFrame):
    """ Documentation
        Parameters:
            X_train: Train samples
            y_train: True labels for X_train
            X_val: Validation samples
            y_val: True labels for Valiation
            X_test: Test samples
            y_test: True labels for X_test

        Out:
          accuracy_score
          confusion_matrix
          metrics_test (precison, recall, F1 Score, Support)
          rmse cross_validation

    """

    train_data = lgbm.Dataset(X_train, y_train)
    valid_data = lgbm.Dataset(X_val, y_val, reference=train_data)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 10,
        'num_leaves': 100,
        'max_depth': 8
    }
    clf_lgbm = lgbm.train(params=params, train_set=train_data,
                          num_boost_round=2000, early_stopping_rounds=50,
                          valid_sets=[valid_data], verbose_eval=10)
    predicted_LGBM = clf_lgbm.predict(X_test)
    y_pred = np.round(predicted_LGBM)

    print("==================================================")
    print("accuracy_score : ", accuracy_score(y_pred, y_test))
    print("confusion_matrix : \n", confusion_matrix(y_pred, y_test))
    metrics = classification_report(y_test, y_pred, output_dict=True)
    print('metrics_test: ' + str(metrics['weighted avg']))
    cv_results = lgbm.cv(params, train_data, num_boost_round=1000,
                         metrics='rmse', early_stopping_rounds=50,
                         nfold=5, stratified=False)
    print("rmse cross_validation : ", np.mean(cv_results['rmse-mean']))
    print("==================================================")


def execute_lgbm_classifier(X_train: pd.core.frame.DataFrame,
                            y_train: pd.core.series.Series,
                            X_val: pd.core.frame.DataFrame,
                            y_val: pd.core.frame.DataFrame,
                            X_test: pd.core.frame.DataFrame,
                            y_test: pd.core.series.Series, model):
    """
        Documentation
        Parameters:
            X_train: Train samples
            y_train: True labels for X_train
            X_val: Validation samples
            y_val: True labels for Valiation
            X_test: Test samples
            y_test: True labels for X_test

        Out:
          metrics croos validation (accuracy_score precison, recall, F1 Score,
          Support)

    """
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              early_stopping_rounds=50)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    print("==================================================")
    print(evaluation(model, X_train, y_train, y_test, y_pred,
                     'classifier', cv=5))
    print("==================================================")
    
# CNN
    
def CNN(max_features, max_words, epochs, X_train, Y_train, X_val, Y_val,
        X_test, batch_size=128, num_classes=3):
    """
    Documentation

    Parameters:
        max_features: the number of rows to take in input
        max_words: the maximum amout of words that will be encoded for one row
        batch_size: number of training examples utilized in one iteration
        epochs: number of iterations
        num_classes: number of labels to predict

    Output:
        y_pred5: prediction of y_test by the trained model
    """
    model5_CNN = Sequential()
    model5_CNN.add(Embedding(max_features, 100, input_length=max_words))
    model5_CNN.add(Dropout(0.2))
    model5_CNN.add(Conv1D(64, kernel_size=3, padding='same',
                          activation='relu', strides=1))
    model5_CNN.add(GlobalMaxPooling1D())
    model5_CNN.add(Dense(128, activation='sigmoid'))
    model5_CNN.add(Dropout(0.4))
    model5_CNN.add(Dense(num_classes, activation='sigmoid'))
    model5_CNN.compile(loss='binary_crossentropy',
                       optimizer='adam', metrics=['accuracy'])
    model5_CNN.summary()
    # % % time
    history5 = model5_CNN.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                              epochs=epochs, batch_size=batch_size, verbose=1)
    y_pred5 = model5_CNN.predict_classes(X_test, verbose=1)
    return y_pred5

