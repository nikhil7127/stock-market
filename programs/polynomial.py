from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

def polynomialRegression(x_train,y_train,x_test,y_test):
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
    return r2_score(y_test,pred)*100