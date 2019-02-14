############################################ INSTALL PACKAGES ##################################
install.packages("pacman")
pacman::p_load("readr", "ggplot2", "rpart", "rpart.plot", "caret","dplyr", "Hmisc", 
               "MASS", "mlr", "plotly", "corrplot",
               "party", "ipred", "LearnBayes")


############################################# IMPORT DATA #######################################

setwd("C:/Users/usuario/Desktop/UBIQUM/Project 4 - Product Profitability in R (Multiple Regression)/Product-Profitability")
existing <- read.csv("./DataSets/existing.csv")
new <- read.csv("./DataSets/new.csv")
options(digits = 3)


# First data set overview
summary(existing)


# Filter concerning Product Types only
needed.categories <- filter(
  existing, ProductType %in% c(
    "PC","Laptop","Smartphone","Netbook"))


needed.categories1 <- filter(
  new, ProductType %in% c(
    "PC","Laptop","Smartphone","Netbook"))


######################################## PRE-PROCESSING #################################################

# MISSING VALUES & DUPLICATES
summary(is.na(existing)) # NA check -> 15 missing values in BestSellersRank
existing$BestSellersRank <- NULL
duplicated(existing)  # duplicates check -> no duplicates



# DATA TYPES & CLASSES TREATMENT
sapply(existing, class) # Check attributes data class -> No attributes need treatment



# Dummify product types to use the attribute in regression
existing.dummified <- dummyVars(" ~ .", data = existing) 
existing.final <- data.frame(predict(existing.dummified, 
                                     newdata = existing)) # new df with dummified Product Type




# OUTLIERS DETECTION & TREATMENT

# Plot data to evaluate the distribution by Product Type
plotly.volume.box <- plot_ly(existing, x = existing$Volume,
                             y= existing$ProductType, type = "box")
plotly.volume.box # Accessories, Extended Warranty and printer have volume outliers 


outliersVol <- boxplot(existing.final$Volume)$out # Boxplot to detect overall outliers 
boxplot(existing.final$Volume)$out # Outliers values detected: 11204 and  7036

existing.final <- existing.final[-which(existing$Volume %in% 
                                          outliersVol),] # Remove outliers in Volume


# Extended warranty has way too many records and
#   it is not a product category we need to focus on, as our main categories
#   are "PC","Laptop","Smartphone","Netbook"

existing.final[34:41,"Price"] <- mean(
  existing.final[34:41,"Price"]
)
existing.final <- existing.final[-c(35:41),] 



########################################## ATTRIBUTE SELECTION ######################################

# Correlation 
corr_analysis <- cor(existing.final)
corrplot <- corrplot(corr_analysis, 
                     method = "number", 
                     tl.cex= 0.55, number.cex = 0.53) 
# 4 and 3 stars reviews are the most correlated variables to volume (followed by Positive)
# however, as 4 and 3 stars are strongly correlated to each other, 
# 4 stars is kept as its correlation to Volume is higher.

# Moreover, 5 star Reviews is excluded as it might be flawed data (correlation = 1)
existing$x5StarReviews <- NULL
existing.final$x5StarReviews <- NULL



#Decision Tree to visualize correlation results
set.seed(123)
control.tree <- ctree_control(maxdepth = 10)
Decision.tree <- ctree (Volume ~ ., data=existing.final, 
                        controls = control.tree)
plot(Decision.tree) # The decision tree verifies that the most correlated variables to volume 
                    # are 4 stars and Positive reviews




#Multiple Linear Regression

set.seed(123)
fit0 <- lm(Volume ~ ., 
           data = existing.final) # fit0 performance (all variables in the linear model):
                                  # R-squared:  0.863;	Adjusted R^2:  0.787 
summary (fit0)                    


fit1 <- lm(Volume ~ PositiveServiceReview + x4StarReviews, # fit1 performance (only Positive & 4 stars in the lm)
           data = existing.final)                          # R-squared:  0.677,	Adjusted R^2:  0.667
summary (fit2)



fit2 <- lm(Volume ~ PositiveServiceReview + x4StarReviews # fit2 performance (fit1 but treating the intercept)
           +0, data = existing.final)                     # R-squared:  0.773,	Adjusted R^2:  0.767 
summary (fit2)


fit3 <- lm(Volume ~ PositiveServiceReview + # fit 3 performance (fit2 adding Negative reviews & product depth)
             NegativeServiceReview +        # R-squared:  0.852,	Adjusted R^2:  0.843 -> best lm results
             x4StarReviews + 
             ProductDepth
           +0, data = existing.final)
summary (fit3)



#Standardization of variables to check if it helps the model perform better 

set.seed(123)
existing.final$vol.standardised <- scale(existing.final$Volume, 
                                         center = TRUE, scale = TRUE)
existing.final$x4.standardised <- scale(existing.final$x4StarReviews, 
                                        center = TRUE, scale = TRUE)
existing.final$pos.standardised <- scale(existing.final$PositiveServiceReview, 
                                         center = TRUE, scale = TRUE)
existing.final$depth.standardised <- scale(existing.final$ProductDepth, 
                                           center = TRUE, scale = TRUE)

# lm applied to standarized attributes
set.seed(123)

# After the standarization, the models tried before perform worse, for instance:
fit3_standarized <- lm(vol.standardised ~ pos.standardised + x4.standardised + 
               depth.standardised
             +0, data = existing.final) 

summary(fit3_standarized) # Best performance among standarised models, 
                          # although it performs worse than not-standarised lm's
                          # R-squared:  0.774,	Adjusted R-squared:  0.765 


########################################### TRAINING ##############################################

# Create Data partition to train and test models

set.seed(123)
existing.partition <- createDataPartition(existing.final$Volume, p = .75, list = FALSE) # partition 75/25
training <- existing.final[existing.partition,]
testing <- existing.final[-existing.partition,]

#LINEAR MODEL

set.seed(123)
model1 <- lm(Volume ~ PositiveServiceReview + x4StarReviews + ProductDepth
             +0, data = training)
summary (model1)
plot(model1)

model2 <- lm(Volume ~ pos.standardised + x4.standardised + depth.standardised
             +0, data = training)
summary (model2)
plot(model2)

model3 <- lm(vol.standardised ~ pos.standardised + x4.standardised + depth.standardised
             +0, data = training)
summary (model3)
plot(model3)

model4 <- lm(vol.standardised ~ PositiveServiceReview + x4StarReviews + ProductDepth
             +0, data = training)
summary (model4)
plot(model4)






##################################### MODELLING #######################################

#### CROSS-VALIDATION ####

fitControl <- trainControl(
  method = "repeatedcv",
  predictionBounds = c(0,NA),
  number = 10,
  repeats = 3)



#RANDOM FOREST

# set.seed(123)
# modelRF <- caret::train(Volume~ PositiveServiceReview + x4StarReviews + ProductDepth, 
#                         data = training, method = "rf", trControl=fitControl, 
#                         tuneLength = 2)
# saveRDS(modelRF, "./Models/modelRF.rds")
modelRF <- readRDS("./Models/modelRF.rds")


#GBT

# set.seed(123)
# modelGBT <- caret::train(Volume~ PositiveServiceReview + x4StarReviews + ProductDepth
#                          +0 , data = training, trControl= fitControl, method = "gbm")
# saveRDS(modelGBT, "./Models/modelGBT.rds")
modelGBT <- readRDS("./Models/modelGBT.rds")

########################################### TESTING #################################################

#LINEAR MODEL (no linearity)

applymodel1 <- predict(model1, 
                       newdata = testing)
summary (applymodel1)

testing$lmpredictions <- applymodel1
testing$absolute.errorlm <- abs(testing$Volume - 
                                  testing$lmpredictions)
testing$relative.errorlm <- testing$absolute.errorlm/testing$Volume

Errors.LM <- ggplot(data = testing, aes(x =Volume, y = absolute.errorlm))+
  geom_smooth()+geom_point()+ggtitle("Absolute Errors in LM")
Errors.LM

Metrics.LM <- postResample(pred = testing$lmpredictions, obs = testing$Volume)
Metrics.LM


#RANDOM FOREST

applymodelRF <- Predict(modelRF, newdata= testing)
summary(applymodelRF)
applymodelRF

testing$RFpredictions <- applymodelRF
testing$absolute.errorRF <- abs(testing$Volume - testing$RFpredictions)
testing$relative.errorRF <- testing$absolute.errorRF/testing$Volume

Errors.RF <- ggplot(data = testing, aes(x =Volume, y = absolute.errorRF))+
  geom_smooth()+geom_point()+ggtitle("Absolute Errors in RF")
Errors.RF

Metrics.RF <- postResample(pred = testing$RFpredictions, obs = testing$Volume)
Metrics.RF

#GBT 

applymodelGBT <- Predict(modelGBT, newdata= testing)
summary(applymodelGBT)
applymodelGBT

testing$GBTpredictions <- applymodelGBT
testing$absolute.errorGBT <- abs(testing$Volume - testing$GBTpredictions)
testing$relative.errorGBT <- testing$absolute.errorGBT/testing$Volume

Errors.GBT <- ggplot(data = testing, aes(x =Volume, y = absolute.errorGBT))+
  geom_smooth()+geom_point()+ggtitle("Absolute Errors in GBT")
Errors.GBT

Metrics.GBT <- postResample(pred = testing$GBTpredictions, obs = testing$Volume)
Metrics.GBT

########################################## ERROR ANALYSIS TO COMPARE RESULTS ######################################

Metrics.LM.df <- data.frame(Metrics.LM)
Metrics.GBT.df <- data.frame(Metrics.GBT)
Metrics.RF.df <- data.frame(Metrics.RF)
Modelresults <- cbind(Metrics.LM.df, Metrics.GBT.df, Metrics.RF.df)

#Absolute Errors Comparison

dfLM=data.frame(x=testing$Volume,y=testing$absolute.errorlm)
dfRF=data.frame(x=testing$Volume,y=testing$absolute.errorRF)
dfGBT=data.frame(x=testing$Volume,y=testing$absolute.errorGBT)
dfLM$model <- "Linear M"
dfRF$model <- "RF"
dfGBT$model <- "GBT"
df.absolute.errors <- rbind(dfLM, dfRF, dfGBT)

df.absolute.errors$ProductType <- 0

df.absolute.errors$ProductTypePC <- testing$ProductType.PC
df.absolute.errors[which(testing$ProductType.PC == 1),]$ProductType <- "PC"

df.absolute.errors$ProductType.Laptop <- testing$ProductType.Laptop
df.absolute.errors[which(testing$ProductType.Laptop == 1),]$ProductType <- "Laptop"

df.absolute.errors$ProductType.Netbook <- testing$ProductType.Netbook
df.absolute.errors[which(testing$ProductType.Netbook == 1),]$ProductType <- "Netbooks"

df.absolute.errors$ProductType.Smartphones <- testing$ProductType.Smartphone
df.absolute.errors[which(testing$ProductType.Smartphone == 1),]$ProductType <- "Smartphones"

df.absolute.errors$ProductType <- factor(df.absolute.errors$ProductType)



g.abs.error.comp <- ggplot(df.absolute.errors, aes(x, y, group=model,colour=ProductType)) +
  geom_point(size=5) +
  stat_smooth(aes(col=model))+ggtitle("Absolute Error Comparison")+ylab("Absolute Error")+
  xlab("Volume")
g.abs.error.comp

#Relative Errors Comparison

dfLMRe=data.frame(x=testing$Volume,y=testing$relative.errorlm)
dfRFRe=data.frame(x=testing$Volume,y=testing$relative.errorRF)
dfGBTRe=data.frame(x=testing$Volume,y=testing$relative.errorGBT)
dfLMRe$model <- "Linear M"
dfRFRe$model <- "RF"
dfGBTRe$model <- "GBT"
df.relative.errors <- rbind(dfLMRe, dfRFRe, dfGBTRe)

g.rela.error.comp <- ggplot(df.relative.errors, aes(x, y, group=model)) + geom_point() + 
  stat_smooth(aes(col=model))+ggtitle("Relative Error Comparison")+ylab("Relative Error")+
  xlab("Volume")
g.rela.error.comp

############################################# NEW DATA ##############################################

#Reclassify variables

sapply(new, class)
id = 2:18
new[id] = data.matrix(new[id])
sapply(existing, class)


#Exclude BestsellerRank, 5Stars 
new$BestSellersRank <- NULL
new$x5StarReviews <- NULL
new.final$BestSellersRank <- NULL
new.final$x5StarReviews <- NULL


############################################ PREDICT ##############################################

#RF Predict

FinalPredictionRF <- predict(modelRF, newdata = new.final) 
FinalPredictionRF
new.final$predicted.VolumeRF <- FinalPredictionRF

#GBT Predict

FinalPredictionGBT <- predict(modelGBT, newdata = new.final) 
FinalPredictionGBT
new.final$predicted.VolumeGBT <- FinalPredictionGBT

#LM Predict

FinalPredictionLM <- predict(model1, newdata = new.final) 
FinalPredictionLM
new.final$predicted.VolumeLM <- FinalPredictionLM


#create new file

output <- new
output$PredictionsRF <- FinalPredictionRF
output$PredictionsGBT <- FinalPredictionGBT
output$PredicrionsLM <- FinalPredictionLM
output
write.csv(output, 'outputfile.csv')


#final predictions
new$x4StarReviews <- NULL
new$PositiveServiceReview <- NULL
new$ProductDepth <- NULL
new$PredictedVolumeGBT <- NULL
new$PredictedVolumeLM <- NULL

new$PredictedVolumeLM <- FinalPredictionLM
new$PredictedVolumeRF <- FinalPredictionRF
new$PredictedVolumeGBT <- FinalPredictionGBT
new$Profitability <- new$Price * new$ProfitMargin * new$PredictedVolumeRF


plot.final.predictions <- ggplot(new[new$ProductType == "PC" | new$ProductType == "Laptop" | new$ProductType == "Netbook" | new$ProductType == "Smartphone",],
  aes(x = ProductType, y = PredictedVolumeRF, fill= as.character(ProductNum)))+ 
  geom_col() + ggtitle("Sales Volume for selected Product Categories") +ylab("Sales Volume")+
  xlab("Product Type")
plot.final.predictions

plot.final.predictions.profitability <- ggplot(new[new$ProductType == "PC" | new$ProductType == "Laptop" | new$ProductType == "Netbook" | new$ProductType == "Smartphone",],
  aes(x = ProductType, y = Profitability, fill= as.character(ProductNum)))+ 
  geom_col() + ggtitle("Profitability for selected Product Categories") + ylab("Profitability")+
  xlab("Product Type")
plot.final.predictions.profitability

plot.final.predictions.total <- ggplot(new, aes(x = ProductType, y = PredictedVolumeRF, fill= as.character(ProductNum)))+ 
  geom_col() + ggtitle("Sales Volume for all  Product Categories") +ylab("Sales Volume")+
  xlab("Product Type")
plot.final.predictions.total

plot.final.predictions.total.profitability <- ggplot(new,
  aes(x = ProductType, y = Profitability, fill= as.character(ProductNum)))+ 
  geom_col() + ggtitle("Profitability for all Product Categories") +ylab("Profitability")+
  xlab("Product Type")
plot.final.predictions.total.profitability

plot.final.prof <- ggplot(
  new[new$ProductType == "PC" | new$ProductType == "Laptop" | new$ProductType == "Netbook" | new$ProductType == "Smartphone",],
  aes(x = ProductType, y = Profitability, fill= as.character(ProductNum)))+ 
  geom_col() + ggtitle ("Product Profitability") + ylab ("Profitability")+
  xlab("Product Type")
plot.final.prof


write.csv(output, file="ProductProfitability.csv", row.names = TRUE)

