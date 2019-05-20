# Titanic: Machine Learning from Disaster 
**Challenge:** The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# Solution using R

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

Let me first tell you what each column is
(1) PassengerId is just the row number. Thankfully, I am using **R** which means I don't have to worry about the off-by-one issue - whew!
(2) Survived is just 0 or 1 (for those who survived).
(3) Pclass tells you if its first, second or third class.
(4-6) Name, Sex and Age are pretty much self-explanatory.
(7) SibSp gives the total number of siblings and spouses that particular passenger has. 
(8) Parch gives the total number of parents and children that particular passenger has.
(9) Ticket is the Ticket number.
(10) Fare is self-explanatory.
(11) Cabin is the cabin-number.
(12) Embarked tells us where the passenger embarked from Cherbourg, Queenstown or Southampton. Psst - I was in Southampton for 4 years! Its a nice place. 

We can also look at the data in the train-file using

```
View(train)
```
which gives us


![Titanic_4_github](https://user-images.githubusercontent.com/50455967/57469186-eb212b80-723a-11e9-87f6-6499f6b368dc.png)


Lets check how the different numerical predictors (Age, SibSp, Parch, Fare) go-in-hand with the response (Survived). To do that, we can make a heatmap using the following:

First lets just make a numerical sub-data using

```
train.num = na.omit( train[,c( "Survived" , "Age"  , "Parch" , "SibSp" , "Fare" )] )
```
where I have omitted the NA's (Not Availbles) as having them wouldn't give us the correlation matrix.

Next, I find the correlation matrix using
```
train.num.cor = cor( train.num )
```

Now, to make a heatmap, I use green to show a correlation of +1 and red to show -1. To attain this, I make a pallette using
```
library(RColorBrewer) # Loads the package
my_palette <- colorRampPalette(c("red", "yellow", "green"))(n = 299)
```

On to the heatmap using
```
library(gplots) # Loads the package
heatmap.2( train.num.cor , col = my_palette, Rowv = NULL , Colv = NULL , dendrogram = "none" , trace="none" , density.info="none" , cellnote = round(train.num.cor , 2) , notecol="black" )
```
which gave me 

![Heatmap](https://user-images.githubusercontent.com/50455967/57475053-6b9a5900-7248-11e9-90cc-9741a15c02a2.png)

In case you are wondering how I achieved this, I used this tutorial on https://sebastianraschka.com/Articles/heatmaps_in_r.html .

This is interesting - a positive correlation for Fare and Survived: Greater the Fare, larger the chance of Survival! Hmm... 


