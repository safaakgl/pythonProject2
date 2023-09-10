import numpy as np
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt
import pyodbc
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import os
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

server = 'tcp:edufate.database.windows.net,1433'
database = 'Miuul'
username = 'safa'
password = 'Miuul0x*'

connection = pyodbc.connect(f'Driver=ODBC Driver 18 for SQL Server;SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;')

query = "SELECT * FROM Students"

df = pd.read_sql(query, connection)

df['Target'] = df['Target'].apply(lambda x: 1 if x == 'Graduate' else 0)
df.head()
df.shape
df.info()

df = df.reset_index(drop=False)
Student_ID = 'Student_ID'
df.rename(columns={"index": Student_ID}, inplace=True)
df["Student_ID"] = df["Student_ID"] + 1

# Değişken isimleri büyütmek ve boşluk varsa ortadan kaldırmak
df.columns = [col.upper().replace(" ", "_") for col in df.columns]


# df matrisinden anlamsız gelen 1. dönem sonu ve 2. dönem sonu değerleri çıkabilir. Pek anlaşılmıyor. Şimdilik yapmadım.
# eliminated_columns = df.iloc[:, 19:31]
# df = df.drop(columns=eliminated_columns)

##################################
# GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=15, car_th=30):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # Threshld üstünde kalan ama kategorik olan değişkenleri manuel ekledim
    deltas = ['APPLICATION_MODE', 'PREVIOUS_QUALIFICATION', 'NACIONALITY', "MOTHER'S_QUALIFICATION",
              "FATHER'S_QUALIFICATION", "MOTHER'S_OCCUPATION", "FATHER'S_OCCUPATION", "COURSE"]

    cat_cols += [deltas for deltas in deltas]
    num_cols = list(set(num_cols) - set(deltas))
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car

for col in cat_cols:
    cat_summary(df, col)


##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=False)

#####
# PCA Görselleştirme
#####
y = df["TARGET"]
X = df[num_cols]


def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df


pca_df = create_pca_df(X, y)


def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


plot_pca(pca_df, "TARGET")


##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "TARGET", col)


##################################
# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "TARGET", col)


##################################
# KORELASYON
##################################

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


correlation_matrix(df, num_cols)
df.corrwith(df["TARGET"]).sort_values(ascending=False)
df[num_cols].corr()

##################################
# Data Preprocessing & Feature Engineering
##################################
# EKSİK DEĞER ANALİZİ
df.isnull().sum()


# Outlier Analizi
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col, 0.01, 0.99))


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


check_df(df)

cat_cols = [col for col in cat_cols if "TARGET" not in col]

df = one_hot_encoder(df, cat_cols, drop_first=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=15, car_th=30)

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

y = df["TARGET"]
X = df.drop(["TARGET", "STUDENT_ID"], axis=1)

check_df(X)


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


missing_values_table(df)

##################################
# BASE MODEL KURULUMU
##################################


cat_cols = [col for col in cat_cols if col not in ["TARGET"]]
cat_cols

check_df(df)
y = df["TARGET"]
X = df.drop(["TARGET", "STUDENT_ID"], axis=1)

#####################
# Holdout yöntemi Başarı Değerlendirme (LogisticRegression Modeli için)
#####################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
LR1 = LogisticRegression(random_state=17).fit(X_train, y_train)

# Train Hatası
y_pred = LR1.predict(X_train)
y_prob = LR1.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# Test Hatası
y_pred = LR1.predict(X_test)
y_prob = LR1.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

#####################
# CV ile Başarı Değerlendirme
#####################

models = [('LR', LogisticRegression(random_state=17)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=17)),
          ('RF', RandomForestClassifier(random_state=17)),
          ('SVM', SVC(gamma='auto', random_state=17)),
          ('XGB', XGBClassifier(random_state=17)),
          ("LightGBM", LGBMClassifier(random_state=17))]

# CatBoost'da hata verdi. Çıkarttım sonra bakarız. "CatBoost", CatBoostClassifier(verbose=-1, random_state=12345)
for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# Model Başarıları
########## LR ##########
""" Accuracy: 0.8513
Auc: 0.9255
Recall: 0.8846
Precision: 0.8301
F1: 0.8561
########## KNN ##########
Accuracy: 0.7796
Auc: 0.8462
Recall: 0.8728
Precision: 0.7359
F1: 0.7982
########## CART ##########
Accuracy: 0.7869
Auc: 0.7869
Recall: 0.7963
Precision: 0.7816
F1: 0.7884
########## RF ##########
Accuracy: 0.847
Auc: 0.9222
Recall: 0.8791
Precision: 0.8264
F1: 0.8516
########## SVM ##########
Accuracy: 0.8395
Auc: 0.9222
Recall: 0.8936
Precision: 0.8073
F1: 0.8478
########## XGB ##########
Accuracy: 0.8463
Auc: 0.9203
Recall: 0.8769
Precision: 0.8271
F1: 0.8509
########## LightGBM ##########
Accuracy: 0.8515
Auc: 0.9263
Recall: 0.8855
Precision: 0.8295
F1: 0.8563 """

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_  # {'max_depth': None, 'max_features': 7, 'min_samples_split': 2, 'n_estimators': 500}

rf_best_grid.best_score_  # 0.8458

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
round(cv_results['test_accuracy'].mean(), 4)  # 0.8458
round(cv_results['test_roc_auc'].mean(), 4)  # 0.9212
round(cv_results['test_recall'].mean(), 4)  # 0.8855
round(cv_results['test_precision'].mean(), 4)  # 0.8208
round(cv_results['test_f1'].mean(), 4)  # 0.8516

########## RF ########## Hyperparametre ile karşılaştırma
# Accuracy: 0.8470 --> 0.8458
# Auc: 0.9222 --> 0.9212
# Recall: 0.8791 --> 0.8855
# Precision: 0.8264 --> 0.8208
# F1: 0.8516 --> 0.8516


xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_best_grid.best_params_  # {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1000}

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
round(cv_results['test_accuracy'].mean(), 4)  # 0.8594
round(cv_results['test_roc_auc'].mean(), 4)  # 0.9319
round(cv_results['test_recall'].mean(), 4)  # 0.8963
round(cv_results['test_precision'].mean(), 4)  # 0.8346
round(cv_results['test_f1'].mean(), 4)  # 0.8643

# Xgboost modeli standar scaler olsa da aynı sonucu verdi ama RF daha iyileşti

########## XGBoost ########## Hyperparametre ile karşılaştırma
# Accuracy: 0.8463 ---> 0.8594
# Auc: 0.9203      ---> 0.9319
# Recall: 0.8769   ---> 0.8963
# Precision: 0.8271---> 0.8346
# F1: 0.8509       ---> 0.8643

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.001, 0.01, 0.1],  # 0.001
               "n_estimators": [100, 300, 500, 1000],  # 1000

               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_  # 'colsample_bytree': 1, 'learning_rate': 0.01, 'n_estimators': 300}

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
round(cv_results['test_accuracy'].mean(), 4)  # 0.8571
round(cv_results['test_roc_auc'].mean(), 4)  # 0.9273
round(cv_results['test_recall'].mean(), 4)  # 0.8981
round(cv_results['test_precision'].mean(), 4)  # 0.8300
round(cv_results['test_f1'].mean(), 4)  # 0.8626

########## LightGBM ########## Hyperparametre ile karşılaştırma
# Accuracy: 0.8515 -->0.8571
# Auc:      0.9263-->0.9273
# Recall:   0.8855-->0.8981
# Precision:0.8295-->0.8300
# F1:       0.8563-->0.8626
########## LightGBM ##########

################################################
# SVM
################################################

SVM_model = SVC(random_state=17)

SVM_params = {"C": [0.1, 1, 10],  # Ceza parametresi
              "kernel": ["linear", "rbf", "sigmoid"]}  # Çekirdek türü

SVM_best_grid = GridSearchCV(SVM_model, SVM_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

SVM_best_grid.best_params_  # {'C': 0.1, 'kernel': 'linear'}

SVM_final = SVM_model.set_params(**SVM_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(SVM_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
round(cv_results['test_accuracy'].mean(), 4)  # 0.8513
round(cv_results['test_roc_auc'].mean(), 4)  # 0.9252
round(cv_results['test_recall'].mean(), 4)  # 0.8855
round(cv_results['test_precision'].mean(), 4)  # 0.8291
round(cv_results['test_f1'].mean(), 4)  # 0.8561


########## SVM ########## Hyperparametre ile karşılaştırma
# Accuracy:  0.8395 --> 0.8499
# Auc:       0.9222--> 0.9248
# Recall:    0.8936 -->0.8809
# Precision: 0.8073--> 0.8297
# F1:        0.8478 -->0.8543
################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final, X, num=15, save=True)
plot_importance(xgboost_final, X, num=15, save=True)
plot_importance(lgbm_final, X, num=15, save=True)

random = X.sample(1, random_state=45)

# rf_final.predict(random)
# xgboost_final.predict(random)
# lgbm_final.predict(random)
# SVM_final.predict(random)

joblib.dump(rf_final, "rf_final.pkl")
joblib.dump(xgboost_final, "xgboost_final.pkl")
joblib.dump(lgbm_final, "lgbm_final.pkl")
joblib.dump(SVM_final, "SVM_final.pkl")

SVM_final_model_from_disc = joblib.load("voting_clf.pkl")
df = pd.read_csv("C:/Users/mehmet kupeli/PycharmProjects/pythonProject/datasets/data.csv", engine='python', sep=None)
random = X.sample(1, random_state=45)

SVM_final._model_from_disc.predict(random)

# BONUS
"""
# Başlangıç zamanını kaydet
start_time = time.time()

# Modeli kaydet
joblib.dump(SVM_final, "SVM_final.pkl")

# Bitiş zamanını kaydet
end_time = time.time()

# İşlem süresini hesapla
process_time = end_time - start_time
print(f"Model kaydetme süresi: {process_time} saniye") """
