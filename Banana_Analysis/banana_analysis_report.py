import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv("banana_quality.csv")

#Checking data
print(data.head())
print(data.info())
print(data.value_counts(subset=['Quality']))
print("Data is fine, no need to remove Nulls")

#Checking out features
sns.pairplot(data=data, hue="Quality")
plt.show()

#Splitting Training and Test Data
Xdata = data.drop(['Quality'], axis=1)
Ydata = data['Quality']

X_train, X_test, Y_train, Y_test = train_test_split(Xdata, Ydata, test_size=0.2, stratify=Ydata, random_state=0)

#Logistic Regression
print("Classifying Via Logistic Regression")
model_lgr = LogisticRegression(random_state=0)
model_lgr.fit(X_train, Y_train)

Y_pred = model_lgr.predict(X_test)

accuray = accuracy_score(Y_pred, Y_test)

print("Accuracy:", accuray)
print(classification_report(Y_test, Y_pred, target_names=['Good', 'Bad']))

labels = ["Good", "Bad"]
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()

print("Displaying first 10 values of Test data (Prediction, Calculated Probability, and Actual)")
print(pd.Series(model_lgr.predict(X_test)[0:10]))
print(model_lgr.predict_proba(X_test)[:10])
print(Y_test[:10])

#Nearest Neighbour
print("Classifying Via K-Nearest Neighbours")

test_accuracy = []

neighbourOptions = range(1,51)
for neighbour in neighbourOptions:
    cltree = KNeighborsClassifier(n_neighbors=neighbour)
    cltree.fit(X_train, Y_train)
    Y_TestPred = accuracy_score(Y_test, cltree.predict(X_test))

    test_accuracy.append(Y_TestPred)

#Plot of test accuracies vs number of neighbours
plt.plot(neighbourOptions,test_accuracy,'bo--')
plt.legend(['Test Accuracy'])
plt.xlabel('Neighbours')
plt.ylabel('Classifier Accuracy')
plt.show()

#A max number of neighbours of 16 was determined to maximize accuracy without suffering from noise or other groups
model_knn = KNeighborsClassifier(n_neighbors=16)
model_knn.fit(X_train, Y_train)
Y_pred = model_knn.predict(X_test)

accuray = accuracy_score(Y_pred, Y_test)

print("Accuracy:", accuray)
print(classification_report(Y_test, Y_pred, target_names=['Good', 'Bad']))

labels = ["Good", "Bad"]
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()

print("Displaying first 10 values of Test data (Prediction, Calculated Probability, and Actual)")
print(pd.Series(model_knn.predict(X_test)[0:10]))
print(model_knn.predict_proba(X_test)[0:10])
print(Y_test[0:10])

#Naive Bayes
print("Classifying Via Naive Bayes")

#Using a gaussian classifier for the data
model_NB = GaussianNB()
model_NB.fit(X_train, Y_train)

Y_pred = model_NB.predict(X_test)

accuray = accuracy_score(Y_pred, Y_test)

print("Accuracy:", accuray)
print(classification_report(Y_test, Y_pred, target_names=['Good', 'Bad']))

labels = ["Good", "Bad"]
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()

print("Displaying first 10 values of Test data (Prediction, Calculated Probability, and Actual)")
print(pd.Series(model_NB.predict(X_test)[0:10]))
print(model_NB.predict_proba(X_test)[0:10])
print(Y_test[0:10])

#Decision Tree
print("Classifying Via Decision Tree")

train_accuracy = []
test_accuracy = []

depthOptions = range(1,51)
for depth in depthOptions:
    cltree = tree.DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=0)
    cltree.fit(X_train, Y_train)
    Y_TrainPred = accuracy_score(Y_train, cltree.predict(X_train))
    Y_TestPred = accuracy_score(Y_test, cltree.predict(X_test))

    train_accuracy.append(Y_TrainPred)
    test_accuracy.append(Y_TestPred)
    
#Plot of training and test accuracies vs the tree depths  
plt.plot(depthOptions,train_accuracy,'rv-',depthOptions,test_accuracy,'bo--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Tree Depth')
plt.ylabel('Classifier Accuracy')
plt.show()

#A max depth of 10 was determined to maximize depth without overfitting
model_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=0)
model_tree.fit(X_train, Y_train)
Y_pred = model_tree.predict(X_test)

accuray = accuracy_score(Y_pred, Y_test)

print("Accuracy:", accuray)
print(classification_report(Y_test, Y_pred, target_names=['Good', 'Bad']))

labels = ["Good", "Bad"]
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()

tree.plot_tree(model_tree, feature_names=list(Xdata.columns), filled=True, fontsize=7)
plt.show()

#Random Forest
print("Classifying Via Random Forest")

train_accuracy = []
test_accuracy = []

estimators = range(1,51)
for estimator in estimators:
    cltree = RandomForestClassifier(criterion='gini', n_estimators=estimator , random_state=0)
    cltree.fit(X_train, Y_train)
    Y_TrainPred = accuracy_score(Y_train, cltree.predict(X_train))
    Y_TestPred = accuracy_score(Y_test, cltree.predict(X_test))
    train_accuracy.append(Y_TrainPred)
    test_accuracy.append(Y_TestPred)
    
#Plot of training and test accuracies vs estimators
plt.plot(depthOptions,train_accuracy,'rv-',depthOptions,test_accuracy,'bo--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Number of Estimators')
plt.ylabel('Classifier Accuracy')
plt.show()

#A max depth of 44 was determined to maximize depth without overfitting
model_forest = RandomForestClassifier(criterion='gini', n_estimators=44, random_state=0)
model_forest.fit(X_train, Y_train)
Y_pred = model_forest.predict(X_test)

accuray = accuracy_score(Y_pred, Y_test)

print("Accuracy:", accuray)
print(classification_report(Y_test, Y_pred, target_names=['Good', 'Bad']))

labels = ["Good", "Bad"]
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()

print("Displaying first 10 values of Test data (Prediction, Calculated Probability, and Actual)")
print(pd.Series(model_forest.predict(X_test)[0:10]))
print(model_forest.predict_proba(X_test)[0:10])
print(Y_test[0:10])

#Feature importance
coefficients = model_lgr.coef_[0]
feature_importance = pd.DataFrame({'Feature': Xdata.columns, 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', title="Logistic Regression")
plt.show()

plt.pie(model_tree.feature_importances_, labels=Xdata.columns, autopct='%1.1f%%')
plt.title("Decision Tree")
plt.show()

plt.pie(model_forest.feature_importances_, labels=Xdata.columns, autopct='%1.1f%%')
plt.title("Random Forest")
plt.show()