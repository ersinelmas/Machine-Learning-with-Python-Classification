from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score
import matplotlib.pyplot as plt
from pandas import read_csv 
from sklearn.model_selection import cross_val_score
import seaborn as sb

url = "dataR2.csv"
dataset = read_csv(url)
array = dataset.values
x = array[:,0:9]
y = array[:,9]
clfDT = DecisionTreeClassifier(random_state = 0)
clfRF = RandomForestClassifier(max_depth=17, random_state=0)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.7, test_size=0.3, random_state=0, stratify=y)


print("**********************************\n")
print("Decision Tree Classifier:\n")
clfDT.fit(x_train, y_train)
test_sonucDT = clfDT.predict(x_test)

print (test_sonucDT)
print("")

print("-----------------------\n")

cmDT = confusion_matrix(y_test, test_sonucDT)
print("Confusion Matrix of Decision Tree: \n\n",cmDT)
print("")
plt.matshow(cmDT)
sb.heatmap(cmDT, annot=True, cmap = "Blues")
plt.title('Decision Tree')
plt.show()

print("-----------------------\n")

print("10 fold cross validation scores of Decision Tree: \n")
kfoldDT = cross_val_score(clfDT, x, y, cv=10)
print(kfoldDT)
print("")
print("Accuracy: %0.2f" % (kfoldDT.mean()))
print("Recall Score: " + str(recall_score(test_sonucDT,y_test)))
print("Precision Score: " + str(precision_score(test_sonucDT, y_test)))
print("")

print("-----------------------\n")
print("DT with data from user:\n")
res1 = clfDT.predict([[45,31.12457844,106,6.420,1.12457808,34.155,7.156753,9.45678,574.197]])
print("Sample input: [45,31.12457844,106,6.420,1.12457808,34.155,7.156753,9.45678,574.197]\n")
print("Result: ",res1)

print("\n**********************************\n")
print("Random Forest Classifier:\n")
clfRF.fit(x_train, y_train)
test_sonucRF = clfRF.predict(x_test)

print (test_sonucRF)
print("")

print("-----------------------\n")

cmRF = confusion_matrix(y_test, test_sonucRF)
print("Confusion Matrix of Random Forest: \n\n",cmRF)
print("")
plt.matshow(cmRF)
sb.heatmap(cmRF, annot=True, cmap = "Blues")
plt.title('Random Forest')
plt.show()

print("-----------------------\n")

print("10 fold cross validation scores of Random Forest:\n")
kfoldRF = cross_val_score(clfRF, x, y, cv=10)
print(kfoldRF)
print("")
print("Accuracy: %0.2f" % (kfoldRF.mean()))
print("Recall Score: " + str(recall_score(test_sonucRF,y_test)))
print("Precision Score: " + str(precision_score(test_sonucRF, y_test)))
print("")

print("-----------------------\n")
print("RF with data from user:\n")
res1 = clfRF.predict([[35,21.12457844,98,7.420,1.12457808,28.155,6.156753,9.45678,774.197]])
print("Sample input: [35,21.12457844,98,7.420,1.12457808,28.155,6.156753,9.45678,774.197]\n")
print("Result:" , res1)
print("\n**********************************\n")