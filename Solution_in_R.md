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

Let see if being male or female increases/decreases the chance of survival. Also, what about Pclass and where your boarded the ship from? Lets check those details. Lets re-do things a bit more neatly this time:
```
clean.train = na.omit(train)
clean.train = clean.train[,-c(1,4,9,11)] # Removing non-numerical ID, Name, Ticket and Cabin
names(clean.train)
```
gave me 
```
[1] "Survived" "Pclass"   "Sex"      "Age"      "SibSp"   
[6] "Parch"    "Fare"     "Embarked"
```
Now I am going to make a new column called NumSex as follow:
```
clean.train$NumSex = 0
clean.train$NumSex[clean.train$Sex=="male"] = 1
clean.train = clean.train[,-3] # Removing the non-numerical "Sex" column
```
A positive correlation will tell me that being a male increases the chances of survivial and a negative will tell me having a penis hurts the chances of survival.

Lets change the Embarked predictor from alphabetic to numerical as follows:
```
clean.train$NumEmbarked = 1
clean.train$NumEmbarked[clean.train$Embarked=="Q"] = 2
clean.train$NumEmbarked[clean.train$Embarked=="S"] = 3
clean.train = clean.train[,-7] # Removing the non-numerical "Embarked" column
```
So what did I do here? What this tells me is the following: A positive correlation tells me that departing from a city whose first letter is further down the series of alphabets increases the chance of surivial! Just boarding from Southampton is better than departing from Cherbourg.
The more informative heatmap is then given by 
```
heatmap.2( cor(clean.train) , col = my_palette, Rowv = NULL , Colv = NULL , dendrogram = "none" , trace="none" , density.info="none" , cellnote = round(cor(clean.train) , 2) , notecol="black" )
```
which gives us

![Heatmap3](https://user-images.githubusercontent.com/50455967/58042620-0835e400-7af0-11e9-9a76-74d21cd79467.jpeg)

As you can see being a male is really bad! And so is having a lower class ticket (3rd) or lower fare. Hence being a male with a 3rd class ticket which was bought with lower fare (fare and class have strong correlation) signficantly reduces the chances of survival!

**Be careful:** this assumption can be incorrect because there were higher number of people with a 3rd class ticket. If that is true we just encountered the concept of **Confounding**.

So lets go see some plots to understand this clearly:
```
S3 = sum(train$Pclass==3 & train$Survived==1)
S2 = sum(train$Pclass==2 & train$Survived==1)
S1 = sum(train$Pclass==1 & train$Survived==1)
D3 = sum(train$Pclass==3 & train$Survived==1)
D3 = sum(train$Pclass==3 & train$Survived==0)
D2 = sum(train$Pclass==2 & train$Survived==0)
D1 = sum(train$Pclass==1 & train$Survived==0)
tab1.Surv = Surv
tab1.Pclass = c(1,1,2,2,3,3)
tab1.nos = c(D1,S1,D2,S2,D3,S3)
tab1 = cbind(tab1.Surv, tab1.Pclass, tab1.nos)
tab1 = data.frame(tab1)
names(tab1)[1] = "Surv"
names(tab1)[2] = "Pclass" # If you are thinking why I didn't do this earlier, it was to eliminate unnecessary confusion
names(tab1)[3] = "Nos"
library(ggplot2)
ggplot(data=tab1, aes(x=Pclass, y=Nos, fill=Surv)) + geom_bar(stat="identity")
```
which gave us
![Pclass_Surv](https://user-images.githubusercontent.com/50455967/58108836-67076600-7ba1-11e9-8980-5188ddde6842.jpeg)

As one can see, the number of people in 3rd class were much higher. Lets just take one look at the % of people who survived in each class - this should be quick.
```
P1 = sum(train$Pclass==1 & train$Survived==1)/sum(train$Pclass==1)
P2 = sum(train$Pclass==2 & train$Survived==1)/sum(train$Pclass==2)
P3 = sum(train$Pclass==3 & train$Survived==1)/sum(train$Pclass==3)
P = c( P1 , P2 , P3 )
barplot(P*100, col = c("green" , "orange" , "red"), xlab = "Pclass" , ylab = "% survived in each Pclass", names.arg = c("1","2","3"), ylim = c(0,70))
```
which then gave me
![pclass-percent](https://user-images.githubusercontent.com/50455967/58109211-0a587b00-7ba2-11e9-8bb1-6289ca901b0f.jpeg)

Aah, so you can see the "probability" of surviving with each class. Alright, that is enough analysis for Pclass and Survival rate. Just to be clear - **confounding** didn't occur here. % of people who survived in Pclass=3 is indeed less.

Lets look at gender. Heatmap clearly showed how males are less favorable. How true is that?
```
G = c(sum(train$Sex=="male" & train$Survived==1)/sum(train$Sex=="male") , sum(train$Sex=="female" & train$Survived==1)/sum(train$Sex=="female"))
barplot(G*100, col = c("cyan" , "pink"), xlab = "Gender" , ylab = "% survived", names.arg = c("male","female"), ylim = c(0,80))
```

![percent_surv_gender](https://user-images.githubusercontent.com/50455967/58113099-b0f44a00-7ba9-11e9-9d7a-34cccdfdcae7.jpeg)

Holy moly! That is nuts! Less than 20% of males survived.

Lets look at the number of males and females and compare those things with a stacked barplot. 
```
SM = sum(train$Sex=="male" & train$Survived==1)
DM = sum(train$Sex=="male" & train$Survived==0)
SF = sum(train$Sex=="female" & train$Survived==1)
DF = sum(train$Sex=="female" & train$Survived==0)
tab2.Surv = c(0,1,0,1)
tab2.Gender = c("Male", "Male" , "Female" , "Female")
tab2.Nos = c(DM, SM , DF , SF)
tab2 = data.frame( cbind( tab2.Surv , tab2.Gender , tab2.Nos ) )
names(tab2)[1] = "Surv"
names(tab2)[2] = "Gender"
names(tab2)[3] = "Nos"
tab2$Nos = tab2.Nos # I had to do it since it thought "Nos" was a factor! Weird huh!
ggplot(data=tab2, aes(x=Gender, y=Nos, fill=Surv)) + geom_bar(stat="identity")
```
which then gave me
![Gender_nos](https://user-images.githubusercontent.com/50455967/58114843-899f7c00-7bad-11e9-895f-04f7b2631bcf.jpeg)

This really tells us something, huh!

Lets look at Fare - now Fare is a continuous variable (a real number greater than or equal to zero in this case). So, to study it what I am going to do is the following: lets look at the median or mean of the fare and assign people to two groups, one whose fare was less than the median or mean, and another whose fare was greater than the median or the mean. Here is a possible issue: what should I chose? The median or the mean? I am going to use the median since it "symmetrizes" the population into two *well-defined* halves. By the way, the median of the fare is £14.45 and the mean is £32.20. This might not seem a lot but remember that we are looking at the 1912. To convert it to 2019 money, *multiply it with 25*. Still not that bad for a trans-atlantic journey huh! Alright, anyhow, let me not go off-tangent to economics and continue my analysis of the fare vs survived.
```
DFL = sum(train$Fare<=median(train$Fare) & train$Survived==0)
DFG = sum(train$Fare>median(train$Fare) & train$Survived==0)
SFG = sum(train$Fare>median(train$Fare) & train$Survived==1)
SFL = sum(train$Fare<=median(train$Fare) & train$Survived==1)
tab3.Surv = c(0,1,0,1)
tab3.Fare = c("Less","Less","More","More")
tab3.Nos = c(DFL, SFL , DFG , SFG)
tab3 = data.frame(cbind(tab3.Surv , tab3.Fare , tab3.Nos))
tab3$Nos = tab3.Nos
ggplot(data=tab3, aes(x=Fare, y=Nos, fill=Surv)) + geom_bar(stat="identity")
```

