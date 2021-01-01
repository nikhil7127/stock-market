from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as n

def decisionTreeRegression(x_train,y_train,x_test,y_test,total):
    dis = DecisionTreeRegressor()
    dis.fit(x_train,y_train)
    pred = dis.predict(x_test)
    plt.clf()
    plt.scatter(x_test,y_test,color="blue")
    plt.scatter(x_test,pred,color="yellow")
    plt.savefig("images/decision.png")
    return [r2_score(y_test,pred)*100,dis.predict([[total]])]
