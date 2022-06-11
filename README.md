# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Popuri.Siva Naga Nithin.
RegisterNumber:  212221240037.
*/
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/content/Spam.csv',encoding='latin-1')
df = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df.head()

df.info()

df.isnull().sum()

x=df["v1"].values
y=df["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
![img1](https://user-images.githubusercontent.com/94154780/173191660-f45c845b-95b0-4fb3-998d-4ba2a927ba10.jpg)
![img2](https://user-images.githubusercontent.com/94154780/173191663-547dbcaf-d635-4287-89ed-d5d7ec48f191.jpg)
![img3](https://user-images.githubusercontent.com/94154780/173191666-340c2eec-3703-4f5f-a36a-34f42e545b35.jpg)
![img4](https://user-images.githubusercontent.com/94154780/173191670-d78a2bb1-62b2-4924-8824-c18c7b2ebe54.jpg)
![img5](https://user-images.githubusercontent.com/94154780/173191682-72ec446f-032f-4906-a373-a20bcf93eab2.jpg)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
