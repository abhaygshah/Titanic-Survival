# Titanic Machine:  Learning from Disaster
This is a solution to the Titanic Challenge on Kaggle (written in R)

```
train = read.csv("Desktop/Titanic/train.csv")
test = read.csv("Desktop/Titanic/test.csv")
```

After importing the files, lets see what the names in the train-files are using

```
names(train)
```

which then gives me

```
[1] "PassengerId" "Survived"    "Pclass"      "Name"       
[5] "Sex"         "Age"         "SibSp"       "Parch"      
[9] "Ticket"      "Fare"        "Cabin"       "Embarked"
````

We can also look at the data in the train-file using

```
View(train)
```
which gives us

![Titanic_4_github](https://user-images.githubusercontent.com/50455967/57469186-eb212b80-723a-11e9-87f6-6499f6b368dc.png)
