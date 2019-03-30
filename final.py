import pandas as pd
from urllib import request
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pyecharts import Map
import statsmodels.api as sm
import pylab
from sklearn import linear_model
from numpy import random
station = pd.read_csv('station.csv',encoding='gbk')
price = pd.read_csv('price.csv',encoding='gbk')
distance = pd.read_csv('distance.csv',encoding='gbk')
distance['lat'] = ''
distance['lng'] = ''
for i in range(distance.shape[0]):
    sta = distance.iat[i, 1]
    lat = ''
    lng = ''
    for j in range(station.shape[0]):
        if sta == station.iat[j,0]:
            latlng = station.iat[j,2].split(',')
            lat = latlng[-1]
            lng = latlng[0]
            break
    distance.at[i,'lat'] = lat
    distance.at[i,'lng'] = lng
def get_total_result(Latitude,Longitude,year):
    id = 'UCLMNMBNOSE40D4FPNQMCDR3JHNV1ENATZOF23ZV0JRVTSVH'
    pw = 'C2HEYAZGPROYQH51F0Y344XSQBIIB2ZCJ0OJ1XAC5SK2KXMA'
    url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}0201'.\
        format(id, pw, Latitude, Longitude, year)
    rq = request.Request(url)
    res = request.urlopen(rq)
    respoen = res.read()
    result = str(respoen, encoding="utf-8")
    js = json.loads(result)
    return js['response']['totalResults']
distance['total_result'] = ''
distance['result_before'] = ''
distance['total_result'] = distance.apply(
    lambda row: get_total_result(row['lat'],row['lng'], '2019'), axis=1)
distance['result_before'] = distance.apply(
    lambda row: get_total_result(row['lat'],row['lng'], '2015'), axis=1)
database = distance[['county', 'distance', 'total_result', 'result_before']]
database['price'] = price['ave price']
land = pd.read_csv('land.csv',encoding='gbk')
database['city'] = ''
database['prov'] = ''
for i in range(database.shape[0]):
    county = database.at[i,'county'][:-2]
    city = ''
    prov = ''
    for j in range(land.shape[0]):
        if county == land.at[j,'县']:
            city = land.at[j,'市']
            prov = land.at[j,'省']
            break
    database.at[i,'city'] = city
    database.at[i,'prov'] = prov
database = pd.read_csv('database.csv',encoding='gbk')
database = database.drop(['1'], axis=1)

# 进行cluster算法，将省进行聚类
X = database[['price', 'total_result', 'distance']]
X= preprocessing.StandardScaler().fit(X).transform(X)
k_means = KMeans(init = "k-means++", n_clusters = 10, n_init = 120)
k_means.fit(X)
k_means_labels = k_means.labels_
database['label'] = ''
database1 = database
for i in range(database.shape[0]):
    database.at[i,'label'] = k_means_labels[i]
lab = 0
prove = []
labels = []
for p in range(database1.shape[0]):
    prov = database1.at[p, 'prov']
    if prov == '[]':
        continue
    else:
        lab = database1.at[p, 'label']
        if prov == '内蒙古自治区':
            prov1 = '内蒙古'
        elif prov == '黑龙江省':
            prov1 = '黑龙江'
        else:
            prov1 = prov[:2]
        prove.append(prov1)
        if p == database1.shape[0] - 2:
            break
        else:
            for j in range(p + 1, database1.shape[0]):
                if database1.at[j, 'prov'] == prov:
                    lab = lab + database1.at[j, 'label']
                    database1.at[j, 'prov'] = '[]'
            labels.append(lab)
map2 = Map("中国地图",'中国', width=1200, height=600)
map2.add('中国', prove, labels,
         visual_range=[min(labels), max(labels)],
         maptype='china',
         is_visualmap=True,
         visual_range_color=['#f8f8ff', '#ffd700', '#d94e5d'],
         # is_label_show= True,
         formatter='value')
map2.show_config()
map2.render(path="中国地图.html")

# 利用cluster算法对浙江省进行聚类
db = database[database['prov'] == '浙江省']
X = database[['price', 'total_result', 'distance']]
X= preprocessing.StandardScaler().fit(X).transform(X)
k_means = KMeans(init = "k-means++", n_clusters = 7, n_init = 120)
k_means.fit(X)
k_means_labels = k_means.labels_
db['label'] = ''
db['label'] = ''
db['index'] = range(db.shape[0])
db = db.set_index('index')
db = db.dropna()
for i in range(db.shape[0]):
    db.at[i,'label'] = k_means_labels[i]

cities = []
labels = []
for p in range(db.shape[0]):
    city = db.at[p, 'city']
    if city == '[]':
        continue
    else:
        lab = db.at[p, 'label']
        cities.append(city)
        if p == db.shape[0] - 2:
            break
        else:
            for j in range(p + 1, db.shape[0]):
                if db.at[j, 'city'] == city:
                    lab = lab + db.at[j, 'label']
                    db.at[j, 'city'] = '[]'
            labels.append(lab)
map2 = Map("浙江省地图",'浙江', width=1200, height=600)
map2.add('浙江省', cities, labels,
         visual_range=[min(labels), max(labels)],
         maptype='浙江',
         is_visualmap=True,
         visual_range_color=['#f8f8ff', '#ffd700', '#d94e5d'],
         # is_label_show= True,
         formatter='value')
map2.show_config()
map2.render(path="浙江省地图.html")

# 输出qqplot
price = database['price']
sm.qqplot(price,line='s')
pylab.show()

# 输出取log之后的qqlot
price = database['price']
price = np.log(price)
sm.qqplot(price,line='s')
pylab.show()

# datacleaning
price = database['price']
price = np.log(price)
minprice = min(price)
database['log price'] = price
database = database[database['log price'] > minprice]
price = database['log price']
sm.qqplot(price,line='s')
pylab.show()
plt.hist(database['log price'],bottom=100,bins=20)
plt.show()

# regression
price = database['price']
for i in range(database.shape[0]):
    if database.at[i,'distance'] > 30:
        database.at[i,'station'] = 0
    else:
        database.at[i, 'station'] = 1
regr = linear_model.LinearRegression()
x = np.asanyarray(database[['distance', 'total_result']])
y = np.asanyarray(price)
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

msk = np.random.rand(database.shape[0]) < 0.8
train = database[msk]
test = database[~msk]
x = np.asanyarray(train[['distance', 'total_result', 'station']])
y = np.asanyarray(train[['log price']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

y_hat= regr.predict(test[['distance', 'total_result', 'station']])
x = np.asanyarray(test[['distance', 'total_result', 'station']])
y = np.asanyarray(test[['log price']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , y) )
X = regr.coef_[0][0]*database['distance'] +regr.coef_[0][1]*database['total_result']

plt.scatter(X,X-random.rand(database.shape[0])*database['station']*regr.coef_[0][2])
plt.show()
