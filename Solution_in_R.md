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
A positive correlation will tell me that being a male increases the chances of survivial and a negative will tell me that being a male hurts the chances of survival. And the opposite for a female passenger.

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
![Fare_vs_Nos](https://user-images.githubusercontent.com/50455967/58116386-02540780-7bb1-11e9-8f7b-b61da79a96c2.jpeg)

Alright, this looks really bad for those with lesser fare. Remember the "Less" and "More" in the above diagram is for those who paid fare less or more than the median of the fare (whose distribution is really skewed). 

Lets do one last check and see how the chances of survival are if one departs from Cherbourg, Queenstown or Southampton.
```
DC = sum(train$Survived==0 & train$Embarked=="C")
SC = sum(train$Survived==1 & train$Embarked=="C")
DQ = sum(train$Survived==0 & train$Embarked=="Q")
SQ = sum(train$Survived==1 & train$Embarked=="Q")
DS = sum(train$Survived==0 & train$Embarked=="S")
SS = sum(train$Survived==1 & train$Embarked=="S")
tab4.Surv = rep(c(0,1),3)
tab4.Emb = c("C" , "C" , "Q" , "Q" , "S" , "S" )
tab4.Nos = c(DS, SC , DQ , SQ , DS , SS)
tab4 = data.frame(cbind(tab4.Surv , tab4.Emb , tab4.Nos))
names(tab4) = c("Surv" , "Emb" , "Nos")
tab4$Nos = tab4.Nos # This is annoying!
ggplot(data=tab4, aes(x=Emb, y=Nos, fill=Surv)) + geom_bar(stat="identity")
```
which then give us

![Emb_Vs_Surv](https://user-images.githubusercontent.com/50455967/58117194-e18cb180-7bb2-11e9-8dbf-8a7a508db9aa.jpeg)

You can clearly see that a large % of people who embarked on this journey from Cherbourg didn't survive. This is also evident from the heatmap above. Alright, I think this is enough analysis for now... Lets move on to fill some missing values, shall we? I can clearly see that the Age predictor has 20% of its values missing. I am going to fill it up using the titles that each person hold. The titles are given in the Name predictor. See, everything comes in some use. 
Note: This idea is indeed superb but I cannot take credit for it - I borrowed this idea from Yassine Ghouzam who has shown some fine work on this competition. 

Alright so, let me go ahead and work on it. These little lines extracts the title (Mr., Mrs, Master, etc)
```
SecondPart = colsplit(train$Name, ",", c("1","2") )[,2]
RemoveSpace = substring( SecondPart , 2 )
train$Titles = colsplit( RemoveSpace ," ",c("1","2") )[,1]
train$Titles = as.factor(train$Titles)
levels(train$Titles) # gave me
    [1] "Capt."     "Col."      "Don."      "Dr."      
    [5] "Jonkheer." "Lady."     "Major."    "Master."  
    [9] "Miss."     "Mlle."     "Mme."      "Mr."      
    [13] "Mrs."      "Ms."       "Rev."      "Sir."     
    [17] "the" 
```
The "the" comes from the 760th row whose name is "Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)"
So, 17 new titles and how many people have them?
```
tab5.Titles = levels(train$Titles)
tab5.Nos = rep(0,17)
for(i in 1:17){tab5.Nos[i] = sum(train$Titles==levels(train$Titles)[i]) }
tab5 = data.frame(tab5.Titles , tab5.Nos)
names(tab5) = c("Titles" , "Nos")
tab5 # gives us
      Titles Nos
    1      Capt.   1
    2       Col.   2
    3       Don.   1
    4        Dr.   7
    5  Jonkheer.   1
    6      Lady.   1
    7     Major.   2
    8    Master.  40
    9      Miss. 182
    10     Mlle.   2
    11      Mme.   1
    12       Mr. 517
    13      Mrs. 125
    14       Ms.   1
    15      Rev.   6
    16      Sir.   1
    17       the   1
```
What I will now do is make a new title called *"Fancy"* and shove "Capt.", "Col.", "Don.", "Dr.", "Jonkheer.", "Lady.", "Major.", "Rev.", "Sir.", and "the". 

I will group "Ms." and "Mlle." into "Miss.", and

group "Mme." into "Mrs.". 

So, we should then have "Fancy", "Miss." (female child), "Master." (male child), "Mrs." (adult/married female) and "Mr." (adult/married male). **We should now be able to guess the ages from them.**

But before I do that I think I made a blunder. This was just the training dataset. I can borrow info from the test dataset about these ages too. So let me go ahead and merge the train and test datasets appropriately. 
```
names(train)
    [1] "PassengerId" "Survived"    "Pclass"      "Name"       
    [5] "Sex"         "Age"         "SibSp"       "Parch"      
    [9] "Ticket"      "Fare"        "Cabin"       "Embarked"   
    [13] "Titles"  
names(test)
    [1] "PassengerId" "Pclass"      "Name"        "Sex"        
    [5] "Age"         "SibSp"       "Parch"       "Ticket"     
    [9] "Fare"        "Cabin"       "Embarked"
```
```
mtrain = train[,3:12]
mtest = test[,2:11]
mfull = rbind(mtrain, mtest)
```
Now, lets extract the title again and do some quick analysis:
```
SecondPart = colsplit(mfull$Name, ",", c("1","2") )[,2]
RemoveSpace = substring( SecondPart , 2 )
mfull$Titles = colsplit( RemoveSpace ," ",c("1","2") )[,1]
mfull$Titles = as.factor(mfull$Titles)
levels(mfull$Titles)
    [1] "Capt."     "Col."      "Don."      "Dona."    
    [5] "Dr."       "Jonkheer." "Lady."     "Major."   
    [9] "Master."   "Miss."     "Mlle."     "Mme."     
    [13] "Mr."       "Mrs."      "Ms."       "Rev."     
    [17] "Sir."      "the"
```
Alright 18 titles. I am going to do the same grouping as before with "Dona." also going under "Fancy". 
```
tab5.Nos = rep(0,18)
for(i in 1:18){
                tab5.Nos[i] = sum(mfull$Titles==levels(mfull$Titles)[i]) 
                }
tab5.Titles = levels(mfull$Titles)
tab5 = data.frame(cbind(tab5.Titles , tab5.Nos))
names(tab5) = c("Titles" , "Nos")
tab5
          Titles Nos
    1      Capt.   1
    2       Col.   4
    3       Don.   1
    4      Dona.   1
    5        Dr.   8
    6  Jonkheer.   1
    7      Lady.   1
    8     Major.   2
    9    Master.  61
    10     Miss. 260
    11     Mlle.   2
    12      Mme.   1
    13       Mr. 757
    14      Mrs. 197
    15       Ms.   2
    16      Rev.   8
    17      Sir.   1
    18       the   1
```

```
axe = which(mfull$Titles == "Capt.")
mfull$Titles[axe] = "Fancy"
axe = which(mfull$Titles == "Col.")
mfull$Titles[axe] = "Fancy"
axe = which(mfull$Titles == "Don.")
mfull$Titles[axe] = "Fancy"
axe = which(mfull$Titles == "Dona.")
mfull$Titles[axe] = "Fancy"
axe = which(mfull$Titles == "Dr.")
mfull$Titles[axe] = "Fancy"
axe = which(mfull$Titles == "Jonkheer.")
mfull$Titles[axe] = "Fancy"
axe = which(mfull$Titles == "Lady.")
mfull$Titles[axe] = "Fancy"
axe = which(mfull$Titles == "Major.")
mfull$Titles[axe] = "Fancy"
axe = which(mfull$Titles == "Rev.")
mfull$Titles[axe] = "Fancy"
axe = which(mfull$Titles == "Sir.")
mfull$Titles[axe] = "Fancy"
axe = which(mfull$Titles == "the")
mfull$Titles[axe] = "Fancy"

axe = which(mfull$Titles == "Ms.")
mfull$Titles[axe] = "Miss."
axe = which(mfull$Titles == "Mlle.")
mfull$Titles[axe] = "Miss."
axe = which(mfull$Titles == "Mme.")
mfull$Titles[axe] = "Mrs."
```
With these changes, lets see what I get:
```
tab5.Nos = rep(0,18)
for(i in 1:18){
                tab5.Nos[i] = sum(mfull$Titles==levels(mfull$Titles)[i]) 
                }
tab5.Titles = levels(mfull$Titles)
tab5 = data.frame(cbind(tab5.Titles , tab5.Nos))
names(tab5) = c("Titles" , "Nos")
tab5
          Titles Nos
    1      Capt.   0
    2       Col.   0
    3       Don.   0
    4      Dona.   0
    5        Dr.   0
    6  Jonkheer.   0
    7      Lady.   0
    8     Major.   0
    9    Master.  61
    10     Miss. 264
    11     Mlle.   0
    12      Mme.   0
    13       Mr. 757
    14      Mrs. 198
    15       Ms.   0
    16      Rev.   0
    17      Sir.   0
    18       the   0
    19     Fancy  29
```
Lets find rows which belong to these 5 titles:
```
row.Master = which( mfull$Titles == "Master." )
row.Mr = which( mfull$Titles == "Mr." )
row.Mrs = which( mfull$Titles == "Mrs." )
row.Miss = which( mfull$Titles == "Miss." )
row.Fancy = which( mfull$Titles == "Fancy" )
```
Lets find the mean age in each of these
```
age.Master = mean(na.omit(mfull$Age[row.Master])) # 5.48
age.Mr = mean(na.omit(mfull$Age[row.Mr])) # 32.25
age.Mrs = mean(na.omit(mfull$Age[row.Mrs])) # 36.92
age.Miss = mean(na.omit(mfull$Age[row.Miss])) # 21.82
age.Fancy = mean(na.omit(mfull$Age[row.Fancy])) # 45.18
```
Lets find the rows with missing ages as follows:
```
rows.Masters.ageless = which(mfull$Titles == "Master." & is.na(mfull$Age) == T)
rows.Mr.ageless = which(mfull$Titles == "Mr." & is.na(mfull$Age) == T)
rows.Miss.ageless = which(mfull$Titles == "Miss." & is.na(mfull$Age) == T)
rows.Mrs.ageless = which(mfull$Titles == "Mrs." & is.na(mfull$Age) == T)
rows.Fancy.ageless = which(mfull$Titles == "Fancy" & is.na(mfull$Age) == T)
```
Assigning the ages as follows:
```
mfull$Age[rows.Fancy.ageless] = age.Fancy
mfull$Age[rows.Masters.ageless] = age.Master
mfull$Age[rows.Miss.ageless] = age.Miss
mfull$Age[rows.Mr.ageless] = age.Mr
mfull$Age[rows.Mrs.ageless] = age.Mrs
```
Lets check if we have any more NA's.
```
sum(is.na(mfull)) 
    1
```
Uh oh, there is still one left. What would it be? Aah, its Fare on the 1044-th row in the full dataset or 153rd in the test set. Mr Thomas Storey. Well what shall we do? 
I think we can fill it up with the mean of the fares for those who embarked from Southampton, have a Pclass=3 and an age of greater than 50. Shall we do that? It makes more sense to do this than filling it up with mean of all the fares. 
Well there are 9 people (a total of 10 including Mr Storey) who satisfy these criteria.
```
mean(na.omit(mfull$Fare[mfull$Embarked=="S" & mfull$Pclass==3 & mfull$Age > 50]))
    7.719275
```
```
mfull$Fare[1044] = 7.719275
```


So, I have finally filled up all the missing values. Its time for some robust analysis. Since I began with (stupidly) assigning the names "train" and "test" to the datasets, its time to ammend these names:
```
train.data = train
test.data = test
train = 1:891
mfull.Survived = c(train.data$Survived , rep(NA,418))
mfull$Survived = mfull.Survived
mfull$Pclass = as.factor(mfull$Pclass)
```
I do a linear fit as follows:
```
glm.fit = glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Titles , subset = train , data = mfull, family = binomial)

summary(glm.fit)

##### RESULT BELOW THIS #####

glm(formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + 
    Fare + Embarked + Titles, family = binomial, data = mfull, 
    subset = train)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.3876  -0.5576  -0.3830   0.5404   2.5939  

Coefficients:
              Estimate Std. Error z value Pr(>|z|)    
(Intercept)  3.305e+01  1.323e+03   0.025 0.980073    
Pclass2     -1.084e+00  3.235e-01  -3.351 0.000804 ***
Pclass3     -2.170e+00  3.193e-01  -6.796 1.08e-11 ***
Sexmale     -1.621e+01  8.326e+02  -0.019 0.984465    
Age         -2.964e-02  9.616e-03  -3.082 0.002055 ** 
SibSp       -5.557e-01  1.260e-01  -4.411 1.03e-05 ***
Parch       -3.574e-01  1.347e-01  -2.653 0.007988 ** 
Fare         3.432e-03  2.648e-03   1.296 0.195050    
EmbarkedC   -1.250e+01  1.029e+03  -0.012 0.990302    
EmbarkedQ   -1.262e+01  1.029e+03  -0.012 0.990207    
EmbarkedS   -1.293e+01  1.029e+03  -0.013 0.989972    
TitlesMiss. -1.671e+01  8.326e+02  -0.020 0.983987    
TitlesMr.   -3.389e+00  5.490e-01  -6.172 6.74e-10 ***
TitlesMrs.  -1.584e+01  8.326e+02  -0.019 0.984818    
TitlesFancy -3.373e+00  7.929e-01  -4.254 2.10e-05 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 1186.66  on 890  degrees of freedom
Residual deviance:  723.63  on 876  degrees of freedom
AIC: 753.63

Number of Fisher Scoring iterations: 14
```

WOW - look at that! Fare and Embarked don't play a role in a linear-fit. Amazing! Or is it? Does my assumption that Fare is uniformly disributed hold here? Lets see... I need to find the skewness of the distribution using library e1071.
```
library(e1071)
skewness(mfull$Fare)
4.359
```
Lets fix this continuous variable's skewness using Box-Cox Transformation as follows:
```
v1 = seq(-5,5,0.05)
v2 = rep(NA,201)
for(i in 1:201){
    if(v1[i] != 0){
    v2[i] = skewness( ((mfull$Fare+1)^v1[i] - 1)/v1[i] )}
    else { v2[i] = skewness(log(mfull$Fare+1)) } }
plot(v1,abs(v2))
which.min(abs(v2))
    98
```
The plot gives us

![FareSkewness](https://user-images.githubusercontent.com/50455967/58187773-49530300-7c6c-11e9-8d85-85acf61f9178.jpeg)

This clearly shows that using v1[98] = -0.15 (the lambda for Box-Cox Transformation) gives the best zero skewness. 
```
mfull$Fare.Std = scale( ( (mfull$Fare+1)^(-0.15) - 1 )/(-0.15) )
```
Age's skewness is not that bad - its 0.37. Lets try to bring it closer to zero. 
```
v4 = rep(NA,201)
> for(i in 1:201){
    if(v1[i] != 0){
    v4[i] = skewness( ((mfull$Age+1)^v1[i] - 1)/v1[i] )}
    else { v4[i] = skewness(log(mfull$Age+1)) } }
which.min(abs(v4))
    117
v1[117]
    0.8
v2[117]
    -0.002134134
mfull$Age.Std = scale( ( (mfull$Age+1)^(0.8) - 1 )/(0.8) )
```
Though Parch and SibSp are discrete variables (positive integers), their skewness is pretty high. We should fix them too. And once we fix them, we have to start worrying about **Outliers**. And do something with them!
