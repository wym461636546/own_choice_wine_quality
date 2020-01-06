#Set work directory
rm(list=ls())
gc()
my_dir<-choose.dir()
setwd(my_dir)
print(paste("The director will be used is",my_dir),sep=":")
options(digits=5)
#install & load packages
packages_used<-c("tidyverse","scales","data.table","caret","rpart",
                 "rpart.plot","corrplot")
for (i in packages_used){
  if (!requireNamespace(i)){
    install.packages(i)
  }
}

library(tidyverse)
library(scales)
library(data.table)
library(caret)
library(rpart)
library(rpart.plot)
library(corrplot)

##download the data that will be used in this study
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
              destfile = dl, mode = "wb")

wine_white<-fread(dl,sep=";")
#head(wine_white)
setnames(wine_white,c("fixed_acidity","volatile_acidity","citric_acid",
                      "residual_sugar","chlorides","free_sulfur_dioxide",
                      "total_sulfur_dioxide","density","PH","sulphates",
                      "alcohol","quality"))
####save the data set just in case will use them next time
save.image("wine_quality.rda")
#load our data
#load("wine_quality.rda")
#####
#create train & test set, the 10% of the data will be our test set
set.seed(7,sample.kind = "Rounding") # set.seed(7) if R version is 3.5 or before
index<-createDataPartition(wine_white$quality,times = 1,
                           p=0.1,list = FALSE)
train_white<-wine_white%>%
  slice(-index)
test_white<-wine_white%>%
  slice(index)
# Several algorithms will be introduced to see which one has better performance
#let's check the distribution of the quality scores
wine_white%>%
  ggplot(aes(quality))+geom_bar(position = "dodge")+
  scale_x_continuous(limits=c(1,10),breaks = seq(1,10,1))

#Most score of white wine is 6 
# First a simple regression, let's have a quick look at the linear relations between variables
corrplot(cor(wine_white),addCoef.col = "grey",rect.col = "blue",
         method="number",number.cex = 0.8,
         diag = FALSE, tl.pos = "lt", cl.pos = "n",
         tl.cex = 0.8,tl.col = "black")

#first we build a linear model that used all variables
white_lm_all<-lm(quality~.,data = train_white)
#then use step regresion to select independent variables
white_lm<-step(white_lm_all)
#Let's see the RMSE of each models
RMSE(predict(white_lm,test_white),test_white$quality)
RMSE(predict(white_lm_all,test_white),test_white$quality)

print(paste("The baseline RMSE of white wine is",
            format(RMSE(predict(white_lm,test_white),test_white$quality),digits = 5),sep=":"))

# Now let's try other methods

set.seed(1,sample.kind = "Rounding") 
models <- c("glm", "svmLinear",
            "knn", "gamLoess", "rf","rpart")    
fits <- lapply(models, function(model){ 
  print(model)
  train(quality ~ ., method = model,trControl=trainControl(method = "cv"),
        data = train_white)
}) 

names(fits) <- models
my_predict<-function(x){predict(x,test_white)}
predict<-sapply(fits,my_predict)
ensemble_pred<-rowMeans(predict)
predict<-as.data.frame(predict)%>%
  mutate(ensemble=ensemble_pred)
auu<-function(x){RMSE(x,test_white$quality)}   
rmses<-apply(predict, 2, auu)
rmses<-as.data.frame(rmses)
rmses$method<-row.names(rmses) 
names(rmses)<-c("RMSE","method")

print(paste("The best performed algorithm is random forest",
            format(rmses[5,1],5),sep=":"))

#head(predict)
#we could see the prediction more clear in image below

residual<-predict-test_white$quality
residual%>%
  pivot_longer(cols=1:7,names_to = "models",values_to = "residual")%>%
  ggplot(aes(models,residual))+geom_boxplot()

#let's tune the rf by ourseleves

set.seed(3,sample.kind = "Rounding")
sqtmtry<- round(sqrt(ncol(train_white) - 1))
rfGrid <- expand.grid(mtry = c(round(sqtmtry / 2), sqtmtry, 2 * sqtmtry))
#below code may take a while
rf_fit<-train(quality ~ ., method = "rf", data = train_white,
              tuneGrid = rfGrid,
              nodesize=5,importance=T)

rf_fit$results
plot(rf_fit)
predict_rf<-predict(rf_fit,test_white)
RMSE(predict_rf,test_white$quality)

imp<-varImp(rf_fit)
imp$importance%>%
  as.data.frame()%>%
  mutate(var=rownames(imp$importance))%>%
  arrange(desc(Overall))
