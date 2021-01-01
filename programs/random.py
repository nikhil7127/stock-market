from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def randomForestRegression(x_train,y_train,x_test,y_test):
    rand = RandomForestRegressor()
    rand.fit(x_train,y_train)
    pred = rand.predict(x_test)
    plt.clf()
    plt.scatter(x_test,y_test,color="blue")
    plt.scatter(x_test,pred,color="yellow")
    plt.savefig("images/random.png")
    return r2_score(y_test,pred)*100
