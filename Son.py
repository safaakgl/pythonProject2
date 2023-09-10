##############################
# ÖĞRENCİLERİN OKUL TERKİNİ VE AKADEMİK BAŞARISINI TAHMİN EDİN
##############################

##############################
# Veri Seti Hakkında
##############################

# Bir yükseköğretim kurumundan (birkaç ayrık veri tabanından elde edilen) tarım, tasarım, eğitim, hemşirelik, gazetecilik, yönetim,
# sosyal hizmet ve teknolojiler gibi farklı lisans derecelerine kayıtlı öğrencilerle ilgili oluşturulan bir veri kümesi.
# Veri kümesi, öğrenci kaydı sırasında bilinen bilgileri (akademik yol, demografik özellikler ve sosyal-ekonomik faktörler)
# ve öğrencilerin birinci ve ikinci dönem sonundaki akademik performanslarını içermektedir. Veriler, öğrencilerin okulu bırakma
# ve akademik başarılarını tahmin etmek üzere sınıflandırma modelleri oluşturmak için kullanılmaktadır.
# Problem, sınıflardan birine karşı güçlü bir dengesizliğin olduğu üç kategorili bir sınıflandırma görevi olarak formüle edilmiştir.

##############################
# Veri seti hangi amaçla oluşturuldu?
##############################

# Veri kümesi, risk altındaki öğrencileri akademik yollarının erken bir aşamasında tespit etmek için makine öğrenimi
# tekniklerini kullanarak yükseköğretimde akademik terk ve başarısızlığın azaltılmasına katkıda bulunmayı amaçlayan bir
# projede oluşturulmuştur, böylece onları destekleyecek stratejiler uygulamaya konulabilir.
# Veri kümesi, öğrenci kaydı sırasında bilinen akademik yol, demografik özellikler ve sosyal-ekonomik faktörler gibi
# bilgileri içermektedir.
# Problem, kursun normal süresinin sonunda üç kategorili bir sınıflandırma görevi (okulu bırakan, kayıtlı ve mezun)
# olarak formüle edilmiştir.

##############################
# Veri seti'ne ait bilgiler!!!
##############################

# 37 Değişken 4424 Gözlem

#'Maritalstatus': Medeni durumu temsil eder.
#'Applicationmode': Başvuru modunu temsil eder.
#'Applicationorder': Başvuru sırasını temsil eder.
#'Course': Alınan kursu temsil eder.
#'Daytime/eveningattendance': Gündüz veya akşam katılımını temsil eder.
#'Previousqualification': Önceki eğitim seviyesini temsil eder.
#'Previousqualification(grade)': Önceki eğitim notunu temsil eder.
#'Nacionality': Uyruğu temsil eder.
#"Mother'squalification": Anne eğitim seviyesini temsil eder.
#"Father'squalification": Baba eğitim seviyesini temsil eder.
#'Mother'soccupation': Anne mesleğini temsil eder.
#'Father'soccupation': Baba mesleğini temsil eder.
#'Admissiongrade': Kabul notunu temsil eder.
#'Displaced': Yerinden edilip edilmediğini temsil eder.
#'Educationalspecialneeds': Özel Eğitime İhtiyacı Var mı ?.
#'Debtor': Borçlu olup olmadığını temsil eder.
#'Tuitionfeesuptodate': Ders ücretlerinin güncel olup olmadığını temsil eder.
#'Gender': Cinsiyeti temsil eder.
#'Scholarshipholder': Burs sahibi olup olmadığını temsil eder.
#'Ageatenrollment': Kayıt anındaki yaşını temsil eder.
#'International': Uluslararası öğrenci olup olmadığını temsil eder.
#'Curricularunits1stsem(credited)': Birinci yarıyılın tamamlandığını temsil eder.
#'Curricularunits1stsem(enrolled)': Birinci yarıyıla kayıt yaptırdığını temsil eder.
#'Curricularunits1stsem(evaluations)': Birinci yarıyılın değerlendirildiğini temsil eder.
#'Curricularunits1stsem(approved)': Birinci yarıyılın onaylandığını temsil eder.
#'Curricularunits1stsem(grade)': Birinci yarıyıl notunu temsil eder.
#'Curricularunits1stsem(withoutevaluations)': Birinci yarıyılın değerlendirilmeden tamamlandığını temsil eder.
#'Curricularunits2ndsem(credited)': İkinci yarıyılın tamamlandığını temsil eder.
#'Curricularunits2ndsem(enrolled)': İkinci yarıyıla kayıt yaptırdığını temsil eder.
#'Curricularunits2ndsem(evaluations)': İkinci yarıyılın değerlendirildiğini temsil eder.
#'Curricularunits2ndsem(approved)': İkinci yarıyılın onaylandığını temsil eder.
#'Curricularunits2ndsem(grade)': İkinci yarıyıl notunu temsil eder.
#'Curricularunits2ndsem(withoutevaluations)': İkinci yarıyılın değerlendirilmeden tamamlandığını temsil eder.
#'Unemploymentrate': İşsizlik oranını
#'Inflationrate': Enflasyon oranını
#'GDP': Gayri Safi Yurtiçi Hasıla
#'Target': Hedef değişken

# Target : object Bu problem, bir kursun normal süresinin sonunda (belirtilen sürenin bitiminde) üç kategoriye ayrılan bir sınıflandırma görevi olarak formüle edilmiştir:
# ayrılan süre içinde bırakanlar (dropout), kayıtlı olanlar (enrolled) ve mezun olanlar (graduate)."


##############################
#Kaynak
##############################

# M.V.Martins, D. Tolledo, J. Machado, L. M.T. Baptista, V.Realinho. (2021)
# "Yükseköğretimde öğrenci performansının erken tahmini: bir vaka çalışması"
# Trends and Applications in Information Systems and Technologies, vol.1, Advances in Intelligent Systems and Computing serisi içinde.
# Springer. DOI: 10.1007/978-3-030-72657-7_16

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score,confusion_matrix
import warnings

warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df=pd.read_csv("Datasets/data.csv",sep=None,engine='python')
df["Target"].value_counts()
df['Target'] = df['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)

##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

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

df.info()
##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=15, car_th=20):
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
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
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

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car

##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=False)

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

##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Target", col)

##################################
# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Target", col)

##################################
# KORELASYON
##################################

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

df.corrwith(df["Target"]).sort_values(ascending=False)

##################################
# EKSİK DEĞER ANALİZİ
##################################

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

##################################
# BASE MODEL KURULUMU
##################################

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Target"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["Target"]
X = dff.drop(["Target"], axis=1)

#####################
# Başarı Değerlendirme (LogisticRegression Modeli için)
#####################
log_model=LogisticRegression().fit(X,y)
y_pred=log_model.predict(X)
print(classification_report(y,y_pred))
y_prob=log_model.predict_proba(X)[:,1]
roc_auc_score(y,y_prob)
def plot_confusion_matrix(y,y_pred):
    acc=round(accuracy_score(y,y_pred),2)
    cm=confusion_matrix(y,y_pred)
    sns.heatmap(cm,annot=True,fmt=".0f")
    plt.xlabel("y_pred")
    plt.xlabel("y")
    plt.title("Accuary Score:{0}".format(acc),size=10)
    plt.show()

plot_confusion_matrix(y,y_pred)

cv_results=cross_validate(log_model,
                          X,y,
                          cv=5,
                          scoring=["accuracy","precision","recall","f1","roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_precision"].mean()
cv_results["test_recall"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

#####################
# Holdout yöntemi Başarı Değerlendirme (LogisticRegression Modeli için)
#####################


X_train,X_test,y_train,y_test=train_test_split(X,
                                               y,
                                               test_size=0.2)
rf_model=RandomForestClassifier().fit(X_train,y_train)
y_pred=rf_model.predict(X_test)
y_prob=rf_model.predict_proba(X_test)[:,1]
print(classification_report(y_test,y_pred))
roc_auc_score(y_test,y_prob)

plot_confusion_matrix(y_test,y_pred)

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(rf_model,X,15)


##################################
# GÖREV 2: Data Preprocessing & FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ
##################################

df.isnull().sum()
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "COUNT": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Target", na_columns)

##################################
# VERİ TİPİ DÖNÜŞÜMLER
##################################

df.columns

cat_cols=['﻿Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance\t', 'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Unemployment rate', 'Inflation rate','GDP']
df[cat_cols]
df[cat_cols]=df[cat_cols].astype(object)

df.info()

def grab_col_names(dataframe, cat_th=15, car_th=20):
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
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
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

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# KATEGORİK DEĞİŞKENLERİN ANALİZİ

for col in cat_cols:
    cat_summary(df, col,False)

# NUMERİK DEĞİŞKENLERİN ANALİZİ

for col in num_cols:
    num_summary(df, col, False)

##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

##################################
# RARE ENCODER
##################################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Target", cat_cols)
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


rare_encoder(df,0.01)

##################################
# YÜKSEK KORELASYON KALDIRMA
##################################

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

df.corrwith(df["Target"]).sort_values(ascending=False)
def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=False)

##################################
# FEATURE EXTRACTİON
##################################

df.columns
df["Admission grade"].describe().T
df["NEW_Admission grade"]=pd.qcut(df["Admission grade"],q=5,labels=[1,2,3,4,5])
df["Previous qualification (grade)"].describe().T
df["NEW_Previous qualification (grade)"]=pd.qcut(df["Previous qualification (grade)"],q=5,labels=[1,2,3,4,5])
df["NEW_Curricular units 1st sem (grade)"] = pd.qcut(df["Curricular units 1st sem (grade)"], q=5,
                                                     labels=["1", "2", "3", "4", "5"])
df["Curricular units 2nd sem (grade)"] = pd.qcut(df["Curricular units 2nd sem (grade)"], q=5,
                                                     labels=["1", "2", "3", "4", "5"])

df.loc[(df['Age at enrollment'] < 19), 'NEW_Age at enrollment'] = '0'
df.loc[(df['Age at enrollment'] >= 19) & (df['Age at enrollment'] < 25), 'NEW_Age at enrollment'] = '1'
df.loc[(df['Age at enrollment'] >= 25), 'NEW_Age at enrollment'] = '2'


##################################
# BASE MODEL KURULUMU
##################################

cat_cols=['﻿Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance\t', 'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Unemployment rate', 'Inflation rate','GDP']
df[cat_cols]
df[cat_cols]=df[cat_cols].astype(object)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Target"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["Target"]
X = dff.drop(["Target"], axis=1)

#####################
# Başarı Değerlendirme (LogisticRegression Modeli için)
#####################

log_model=LogisticRegression().fit(X,y)
y_pred=log_model.predict(X)
print(classification_report(y,y_pred))
y_prob=log_model.predict_proba(X)[:,1]
roc_auc_score(y,y_prob)
def plot_confusion_matrix(y,y_pred):
    acc=round(accuracy_score(y,y_pred),2)
    cm=confusion_matrix(y,y_pred)
    sns.heatmap(cm,annot=True,fmt=".0f")
    plt.xlabel("y_pred")
    plt.xlabel("y")
    plt.title("Accuary Score:{0}".format(acc),size=10)
    plt.show()

plot_confusion_matrix(y,y_pred)

cv_results=cross_validate(log_model,
                          X,y,
                          cv=5,
                          scoring=["accuracy","precision","recall","f1","roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_precision"].mean()
cv_results["test_recall"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

#####################
# Holdout yöntemi Başarı Değerlendirme (LogisticRegression Modeli için)
#####################


X_train,X_test,y_train,y_test=train_test_split(X,
                                               y,
                                               test_size=0.2)
rf_model=RandomForestClassifier().fit(X_train,y_train)
y_pred=rf_model.predict(X_test)
y_prob=rf_model.predict_proba(X_test)[:,1]
print(classification_report(y_test,y_pred))
roc_auc_score(y_test,y_prob)

plot_confusion_matrix(y_test,y_pred)

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(rf_model, X,15)

#####################
# CV ile Başarı Değerlendirme
#####################


def base_models(X,y,scoring="roc_auc"):
    print("Base Models......")
    classifiers=[("LR",LogisticRegression()),
                 ("KNN",KNeighborsClassifier()),
                 ("CART",DecisionTreeClassifier()),
                 ("RF",RandomForestClassifier()),
                 ("Adaboost",AdaBoostClassifier()),
                 ]
    for name,classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
        print(f"########## {name} ##########")
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

base_models(X, y)

knn_params = {"n_neighbors": range(2, 20)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params)]
def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_best_grid.best_score_

rf_final = rf_model.set_params(**rf_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
round(cv_results['test_accuracy'].mean(), 4)  #
round(cv_results['test_roc_auc'].mean(), 4)
round(cv_results['test_recall'].mean(), 4)
round(cv_results['test_precision'].mean(), 4)
round(cv_results['test_f1'].mean(), 4)

plot_importance(rf_final, X, num=15, save=False)


random = X.sample(1, random_state=45)

rf_final.predict(random)

#joblib.dump(rf_final, "rf_final_dropout.pkl")

#rf_final_model_from_disc = joblib.load("rf_final_dropout.pkl")

# df = pd.read_csv("Datasets/data.csv", engine='python', sep=None)
# random = X.sample(1)

# rf_final_model_from_disc.predict(random)