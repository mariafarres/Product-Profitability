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


# CATEGORIES TREATMENT

# Extended warranty has way too many records and
#   it is not a product category we need to focus on, as our main categories
#   are "PC","Laptop","Smartphone","Netbook"

existing.final[34:41,"Price"] <- mean(     # mean of all prices for Extended Warranty (rows 34:41)
  existing.final[34:41,"Price"]
)
existing.final <- existing.final[-c(35:41),] # delete the rest of rows for Extended Warranty



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




# Carry out a Multiple Linear Regression to study correlation attributes further 
set.seed(123)

# fit including ONLY the most related attributes to Volume (4 stars and positive reviews)
fit1 <- lm(Volume ~ x4StarReviews + PositiveServiceReview+ # initial fit performance -> R-squared:  0.677,	Adjusted R^2:  0.667
             0, data = existing.final) # Intercept detected! (Intercept)  101.801     47.717    2.13    0.036 * 
# model performance increases without Intercept -> R-squared:  0.773,	Adjusted R^2:  0.767 


# Attributes tandardization to check if it helps the model perform better 
set.seed(123)
existing.final$vol.standardised <- scale(existing.final$Volume, 
                                         center = TRUE, scale = TRUE)
existing.final$x4.standardised <- scale(existing.final$x4StarReviews, 
                                        center = TRUE, scale = TRUE)
existing.final$pos.standardised <- scale(existing.final$PositiveServiceReview, 
                                         center = TRUE, scale = TRUE)



# lm applied to standarized attributes
set.seed(123)
fit1_standarized <- lm(vol.standardised ~  # After the standarization, the models tried before perform worse
                         pos.standardised +  # R-squared:  0.677,	Adjusted R-squared:  0.667
                         x4.standardised+0,
                       data = existing.final) 




################################# DATA PARTITION & CROSS-VALIDATION ##############################################

# Create Data partition to train and test models

set.seed(123)
existing.partition <- createDataPartition(existing.final$Volume, p = .75, list = FALSE) # partition 75/25
training <- existing.final[existing.partition,]
testing <- existing.final[-existing.partition,]


#### CROSS-VALIDATION preparation to control models' fit ####

fitControl <- trainControl(
  method = "repeatedcv",
  predictionBounds = c(0,NA),
  number = 10,
  repeats = 3)


##################################### MODELLING #######################################

#LINEAR MODEL

# set.seed(123)
# model1_lm <- lm(Volume ~ PositiveServiceReview 
#            + x4StarReviews + ProductDepth+0, 
#            data = training)
# saveRDS(model1_lm, "./Models/model1_lm.rds")

model1_lm <- readRDS("./Models/model1_lm.rds") # R-squared:  0.885,	Adjusted R-squared:  0.878 


#GBT

# set.seed(123)
# modelGBT <- caret::train(Volume~ PositiveServiceReview + x4StarReviews + ProductDepth+0,
#                          data = training, trControl= fitControl, method = "gbm")
# saveRDS(modelGBT, "./Models/modelGBT.rds")
modelGBT <- readRDS("./Models/modelGBT.rds")  # RMSE 304 ;  R-squared 0.849 ; MAE 220



#RANDOM FOREST

# set.seed(123)
# modelRF <- caret::train(Volume~ PositiveServiceReview + x4StarReviews + ProductDepth,
#                         data = training, method = "rf", trControl=fitControl,
#                         tuneLength = 2)
# saveRDS(modelRF, "./Models/modelRF.rds")
modelRF <- readRDS("./Models/modelRF.rds") # BEST METRICS (lowest error & best fit:
                                           #  RMSE 226 ; R-squared 0.928 ; MAE  129


########################################### TESTING #################################################


#LINEAR MODEL -> no linearity detected (model disregarded)
  # Metrics: RMSE     Rsquared      MAE 
  #        420.188    0.544         226.588 

applymodel1 <- predict(model1_lm, 
                       newdata = testing)

testing$lmpredictions <- applymodel1
testing$absolute.errorlm <- abs(testing$Volume - 
                                  testing$lmpredictions)
testing$relative.errorlm <- testing$absolute.errorlm/testing$Volume

Errors.LM <- ggplot(data = testing, aes(x =Volume, y = absolute.errorlm))+
  geom_smooth()+geom_point()+ggtitle("Absolute Errors in LM")
Errors.LM

Metrics.LM <- postResample(pred = testing$lmpredictions, obs = testing$Volume)




#GRADIENT BOOSTING TREES
# Metrics:   RMSE       Rsquared      MAE 
#           331.947      0.853       205.262 

applymodelGBT <- Predict(modelGBT, newdata= testing)

testing$GBTpredictions <- applymodelGBT
testing$absolute.errorGBT <- abs(testing$Volume - testing$GBTpredictions)
testing$relative.errorGBT <- testing$absolute.errorGBT/testing$Volume

Errors.GBT <- ggplot(data = testing, aes(x =Volume, y = absolute.errorGBT))+
  geom_smooth()+geom_point()+ggtitle("Absolute Errors in GBT")
Errors.GBT

Metrics.GBT <- postResample(pred = testing$GBTpredictions, obs = testing$Volume)


#RANDOM FOREST
# Metrics:  RMSE       Rsquared      MAE 
#           184.208    0.967        96.929  ->   RF metrics in testing are also the best!

applymodelRF <- Predict(modelRF, newdata= testing)

testing$RFpredictions <- applymodelRF
testing$absolute.errorRF <- abs(testing$Volume - testing$RFpredictions)
testing$relative.errorRF <- testing$absolute.errorRF/testing$Volume

Errors.RF <- ggplot(data = testing, aes(x =Volume, y = absolute.errorRF))+
  geom_smooth()+geom_point()+ggtitle("Absolute Errors in RF")
Errors.RF

Metrics.RF <- postResample(pred = testing$RFpredictions, obs = testing$Volume)


########################################## ERROR ANALYSIS TO COMPARE RESULTS ######################################

# Table with all metrics together (visualization purposes)
Metrics.LM.df <- data.frame(Metrics.LM)
Metrics.GBT.df <- data.frame(Metrics.GBT)
Metrics.RF.df <- data.frame(Metrics.RF)
Modelresults <- cbind(Metrics.LM.df, Metrics.GBT.df, Metrics.RF.df)

# Table with Absolute Errors Comparison (visualization purposes)

dfLM=data.frame(x=testing$Volume,y=testing$absolute.errorlm)
dfRF=data.frame(x=testing$Volume,y=testing$absolute.errorRF)
dfGBT=data.frame(x=testing$Volume,y=testing$absolute.errorGBT)
dfLM$model <- "LM abs.error"
dfRF$model <- "RF abs.error"
dfGBT$model <- "GBT abs.error"
df.absolute.errors <- rbind(dfLM, dfRF, dfGBT)


g.abs.error.comp <- ggplot(df.absolute.errors, aes(x, y, group=model)) +
    stat_smooth(aes(col=model))+ggtitle("Absolute Error Comparison")+ylab("Absolute Error")+
  xlab("Volume")
g.abs.error.comp # RF is the one with the lowest error; 
                 # however, we need to bare in mind that it predicts high sales volumes worse


# Table with Relative Errors Comparison (visualization purposes)

dfLMRe=data.frame(x=testing$Volume,y=testing$relative.errorlm)
dfRFRe=data.frame(x=testing$Volume,y=testing$relative.errorRF)
dfGBTRe=data.frame(x=testing$Volume,y=testing$relative.errorGBT)
dfLMRe$model <- "LM rel.error"
dfRFRe$model <- "RF rel.error"
dfGBTRe$model <- "GBT rel.error"
df.relative.errors <- rbind(dfLMRe, dfRFRe, dfGBTRe)

g.rela.error.comp <- ggplot(df.relative.errors, aes(x, y, group=model)) + geom_point() + 
  stat_smooth(aes(col=model))+ggtitle("Relative Error Comparison")+ylab("Relative Error")+
  xlab("Volume")
g.rela.error.comp  # RF provides us with a really low relative error




############################################# VALIDATION SET ##############################################

#### PRE-PROCESSING ####

# DATA TYPES & CLASS TREATMENT

sapply(new, class)

#Exclude BestsellerRank & 5Stars as we did in "existing
new$BestSellersRank <- NULL
new$x5StarReviews <- NULL


############################################ PREDICT ##############################################


#RF Prediction <- chosen method to predict as it presents the lowest error metrics

FinalPredictionRF <- predict(modelRF, newdata = new) 
new$PredictedVolumeRF <- FinalPredictionRF #Sales volume prediction with Random Forest

# Profitability calculation from Sales volume prediction*price*profit margin
new$Profitability <- new$Price * new$ProfitMargin * new$PredictedVolumeRF



######################################## VISUALIZATION ######################################

# plot Sales by each concerning Product Category
plot.final.predictions <- ggplot(new[new$ProductType == "PC" | 
                                       new$ProductType == "Laptop" | 
                                       new$ProductType == "Netbook" | 
                                       new$ProductType == "Smartphone",],
  aes(x = ProductType, y = PredictedVolumeRF, fill= as.character(ProductNum)))+ 
  geom_col() + 
  ggtitle("Sales by each concerning Product Category") +
  ylab("Sales Volume")+
  xlab("Product Type")+
  guides(fill=guide_legend(title="Product Number"))

plot.final.predictions


# plot concerning categories to show the profitability predicted & empower
# stores to keep promoting them, as they will achieve good results
plot.final.predictions.profitability <- ggplot(new[new$ProductType == "PC" | 
                                                     new$ProductType == "Laptop" | 
                                                     new$ProductType == "Netbook" | 
                                                     new$ProductType == "Smartphone",],
  aes(x = ProductType, y = Profitability, fill= as.character(ProductNum)))+ 
  geom_col() + 
  ggtitle("Profitability by each concerning Product Category") + 
  ylab("Profitability")+
  xlab("Product Type")+ 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  guides(fill=guide_legend(title="Product Number"))

plot.final.predictions.profitability


# plot sales volume by Product type & number to identify most powerful categories & products
# in terms of sales volume
plot.final.predictions.total <- ggplot(new, aes(x = ProductType, y = PredictedVolumeRF, 
                                                fill= as.character(ProductNum))) + 
  geom_col() + 
  ggtitle("Sales Volume by Product type & Product number") +
  ylab("Sales Volume") +
  xlab("Product Type") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  guides(fill=guide_legend(title="Product Number"))

plot.final.predictions.total


# plot profitability (all categories included) <- we conclude that the most profitable
# categories are Game consoles and Tablets, followed by PCs.
plot.final.predictions.total.profitability <- ggplot(new, aes(x = ProductType, 
                                                              y = Profitability, 
                                                              fill= as.character(ProductNum)))+ 
  geom_col() + 
  ggtitle("Profitability by Product Category") +
  ylab("Profitability")+
  xlab("Product Type")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
plot.final.predictions.total.profitability


# export final CSV 
write.csv(new, file="ProductProfitability.csv", row.names = TRUE)

