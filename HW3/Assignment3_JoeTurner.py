import os
import cv2
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report


def load(Dataset_path):
    data = []
    labels = []
    class_folders = os.listdir(Dataset_path)
    for class_name in class_folders:
        image_list = os.listdir(Dataset_path+class_name)
        for image_name in image_list:
            image = cv2.imread(Dataset_path+class_name+'/'+image_name)
            image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
            data.append(image)
            labels.append(class_name)
        print(class_name + " folder done")
    return (np.array(data), np.array(labels))


Dataset_path = "animals/"
data, labels = load(Dataset_path)
data = data.reshape((data.shape[0], 3072))
print(data.shape)
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)
X_train, X_tv, y_train, y_tv = train_test_split(
    data, labels, test_size=0.30, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(
    X_tv, y_tv, test_size=0.33, random_state=42)
print('train test validate split done')

# k nearest neigbors
K = [3, 5, 7]
# distance (1 = manhattan, 2 = euclidian)
L = [1, 2]

BestK = 0
BestL = 0
BestAccuracy = 0

for l in L:
    for k in K:
        model = KNeighborsClassifier(n_neighbors=k, p=l)
        model.fit(X_train, y_train)
        # print(classification_report(y_test, model.predict(
        #    X_test), target_names=le.classes_))
        accuracy = metrics.accuracy_score(y_valid, model.predict(
            X_valid))
        print("L" + str(l) + ", " + "k=" + str(k) +
              ", Accuracy=" + str(accuracy))
        if accuracy > BestAccuracy:
            BestAccuracy = accuracy
            BestK = k
            BestL = l

print("Best: L" + str(BestL) + ", k=" +
      str(BestK) + ", Accuracy=" + str(BestAccuracy))

model = KNeighborsClassifier(n_neighbors=BestK, p=BestL)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(
    X_test), target_names=le.classes_))
