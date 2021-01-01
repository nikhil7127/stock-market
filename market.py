import pandas as p
import numpy as n
import matplotlib.pyplot as plt

df = p.read_csv("ITC_stock.csv")

df.drop(df[df["Open"]=="-"].index,inplace=True)

x = n.array(range(0,len(df["Price"])))
y = df["Price"].values

"""#Plotting Data"""

plt.plot(x,y)
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(range(0,len(df["Price"]),20))
plt.savefig("images/Orginal.png")

"""#Train and Test Data"""

x = x.reshape((len(y),1))
y = y.reshape((len(y),1))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

"""#Linear Regression"""

from sklearn.linear_model import LinearRegression
linear = LinearRegression()

linear.fit(x_train,y_train)

pred = linear.predict(x_test)
plt.clf()
plt.scatter(x_test,y_test,color="blue")
plt.scatter(x_test,pred,color="yellow")
plt.savefig("images/Linear.png")

"""# Polynomial"""

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=7)

xPolyTrain = poly.fit_transform(x_train)
xpolyTest = poly.fit_transform(x_test)

linearPloy = LinearRegression()

linearPloy.fit(xPolyTrain,y_train)

pred = linearPloy.predict(xpolyTest)
plt.clf()
plt.scatter(x_test,y_test,color="blue")
plt.scatter(x_test,pred,color="yellow")
plt.savefig("images/polynomial.png")

"""# Support Vector Machine"""

from sklearn.svm import SVR
svr = SVR()

from sklearn.preprocessing import StandardScaler

x_stand = StandardScaler()
y_stand = StandardScaler()
xTrain = x_stand.fit_transform(x_train)
yTrain = y_stand.fit_transform(y_train)
xTest = x_stand.fit_transform(x_test)
yTest = y_stand.fit_transform(y_test)

svr.fit(xTrain,yTrain)

pred = svr.predict(xTest)
plt.clf()
plt.scatter(x_stand.inverse_transform(xTest),y_stand.inverse_transform(yTest),color="blue")
plt.scatter(x_stand.inverse_transform(xTest),y_stand.inverse_transform(pred),color="yellow")
plt.savefig("images/svr.png")

"""#Decison Tree Regression"""

from sklearn.tree import DecisionTreeRegressor
dis = DecisionTreeRegressor()

dis.fit(x_train,y_train)
pred = dis.predict(x_test)
plt.clf()
plt.scatter(x_test,y_test,color="blue")
plt.scatter(x_test,pred,color="yellow")
plt.savefig("images/decision.png")

"""# Random Forest"""

from sklearn.ensemble import RandomForestRegressor
rand = RandomForestRegressor()

rand.fit(x_train,y_train)
pred = rand.predict(x_test)
plt.clf()
plt.scatter(x_test,y_test,color="blue")
plt.scatter(x_test,pred,color="yellow")
plt.savefig("images/random.png")

