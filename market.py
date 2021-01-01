import pandas as p
import numpy as n
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from programs.linear import linearRegression
from programs.polynomial import polynomialRegression
from programs.svr import supportVectorRegression
from programs.decision import decisionTreeRegression
from programs.random import randomForestRegression
from  getStockData import Stock_info

df = p.read_csv("ITC_stock.csv")
df.drop(df[df["Open"]=="-"].index,inplace=True)

x = n.array(range(0,len(df["Price"])))
y = df["Price"].values

#Plotting Data
plt.plot(x,y)
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(range(0,len(df["Price"]),20))
plt.savefig("images/Orginal.png")

#Train and Test Data
x = x.reshape((len(y),1))
y = y.reshape((len(y),1))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

#Linear Regression
linearAcc=linearRegression(x_train,y_train,x_test,y_test)

# Polynomial
polyAcc = polynomialRegression(x_train,y_train,x_test,y_test)

# Support Vector Machine
svrAcc = supportVectorRegression(x_train,y_train,x_test,y_test)

#Decison Tree Regression
decAcc = decisionTreeRegression(x_train,y_train,x_test,y_test)

# Random Forest"
randomAcc = randomForestRegression(x_train,y_train,x_test,y_test)

print(f"Linear Regression        :   {round(linearAcc,3)}")
print(f"Polynomial Regression    :   {round(polyAcc,3)}")
print(f"Support Vector Machine   :   {round(svrAcc,3)}")
print(f"Decision Tree Regression :   {round(decAcc,3)}")
print(f"Random Forest            :   {round(randomAcc,3)}")

