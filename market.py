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
total = len(df["Price"])
x = n.array(range(0,total))
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
linearAcc=linearRegression(x_train,y_train,x_test,y_test,total+1)

# Polynomial
polyAcc = polynomialRegression(x_train,y_train,x_test,y_test,total+1)

# Support Vector Machine
svrAcc = supportVectorRegression(x_train,y_train,x_test,y_test,total+1)

#Decison Tree Regression
decAcc = decisionTreeRegression(x_train,y_train,x_test,y_test,total+1)

# Random Forest"
randomAcc = randomForestRegression(x_train,y_train,x_test,y_test,total+1)

print(f"Linear Regression        :   {round(linearAcc[0],3)} {linearAcc[1][0][0]}")
print(f"Polynomial Regression    :   {round(polyAcc[0],3)} {polyAcc[1][0][0]}")
print(f"Support Vector Machine   :   {round(svrAcc[0],3)} {svrAcc[1][0]}")
print(f"Decision Tree Regression :   {round(decAcc[0],3)} {decAcc[1][0]}")
print(f"Random Forest            :   {round(randomAcc[0],3)} {randomAcc[1][0]}")

