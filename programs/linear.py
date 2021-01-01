from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def linearRegression(x_train,y_train,x_test,y_test):
    linear = LinearRegression()
    linear.fit(x_train,y_train)
    pred = linear.predict(x_test)
    plt.clf()
    print(type(x_test))
    plt.scatter(x_test,y_test,color="blue")
    plt.scatter(x_test,pred,color="yellow")
    plt.savefig("images/Linear.png")
    return r2_score(y_test,pred)*100