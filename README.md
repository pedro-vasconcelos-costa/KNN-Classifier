# KNN-Classifier

## 01 - PROJECT & DATA
While there are several other Machine Learning techniques capable of solving much more complex problems and tackling grater dimensions of data and number of parameters involved, choosing some of them for this specific task could be considered an overkill. The use of AI must follow a sustainable proposition and justify the amount of time and resources dedicated to its implementation. The KNN model bears plenty of capability to handle the classification of the Iris Flower dataset. 

### Project flowchart:
![image](https://user-images.githubusercontent.com/130906484/232326414-9da41ff5-3fe4-4e29-af80-eb1ef2d4919b.png](https://github.com/pedro-vasconcelos-costa/KNN-Classifier/blob/main/img_%20flowchart.png)

Also known as Fisher’s dataset (Fisher, R., 1936) the Iris Flower set comprehends 150 instances of data containing selected characteristics of three sub-species of the Iris Flower (Vatshayan, S., 2019). It contains individual measurements for each specimen describing their respective petal length and width as well as sepal length and width. There are 50 instances of each species, iris-setosa, iris-virginica and iris-versicolor, and the pattern generated by similar measurements on each of the different species allow for the algorithm to identify unseen examples.
  
![img_ Iris Species](https://github.com/pedro-vasconcelos-costa/KNN-Classifier/blob/main/img_%20Iris%20Species.png)

## 02 - APPROACH & IMPLEMENTATION DETAILS 
	
The algorithm used in this Project was programmed in python. PyCharm was chosen as the programming environment.
Libraries used: 

-> Numpy: allows for operations with multi-dimensional arrays of data
-> Pandas: process and analyse data in various formats
-> Seaborn: generates the heat-map 
-> Matplotlib: plots data into graphs 
-> scikit-learn: library dedicated to machine learning algorithms 

![image](https://user-images.githubusercontent.com/130906484/232326339-0649b988-6d77-4dbc-bab9-76e8a82691c8.png)

	  
Import and prepare the data:

Remove the index labels and separate it into a variable x with the parameter values and variable y with the classes. Encode the class variable into numerical values 0-setosa, 1-versicolor, and 3-virginica and fit them accordingly, the x instances with their respective y class with sklearn’s functions LabelEncoder & fit_transform.

![image](https://user-images.githubusercontent.com/130906484/232326482-b1e9a8d6-8c49-4827-b49d-3340d674b418.png)

Once variables are defined and encoded, the data must be randomized and divided into training and test sets. This process can be achieved through the train_test_split function. Initially, I arbitrarily configured the data split to 20% test and 80% training, and experimented variations to verify changes in performance.
 
![image](https://user-images.githubusercontent.com/130906484/232326529-f5424bef-9056-4b75-a571-86fa6dfbc1e5.png)

![image](https://user-images.githubusercontent.com/130906484/232326538-b0d458d7-dbf0-43f8-89a7-7baea5c34a4a.png)
