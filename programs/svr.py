from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def supportVectorRegression(x_train,y_train,x_test,y_test):
    svr = SVR()
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
    return r2_score(y_test,y_stand.inverse_transform(pred))*100
