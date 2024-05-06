from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from scipy.sparse import random

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

model = GaussianNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Gauss Naive Bayes Model Doğruluğu (% Olarak) : ", metrics.accuracy_score(y_test, y_pred)*100)
