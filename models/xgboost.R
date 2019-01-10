library(tidyverse)
library(rjson)
library(rpart)
library(rpart.plot)
library(caret)
library(Matrix)
library(MatrixModels)
library(randomForest)
library(xgboost)
library(Metrics)


train <- read_csv("data/train.csv") %>%
  mutate(AdoptionSpeed = as.factor(AdoptionSpeed))

test <- read_csv("data/test/test.csv")

###############################################################################
# The below function to read in the sentiment files comes from
# Greg Murray's kernel: https://www.kaggle.com/gregmurray30/ordinal-logit/code
###############################################################################

# Extract sentiment scores for train and test

filenames_train <- list.files("data/train_sentiment", full.names=TRUE)
filenames_test <- list.files("data/test_sentiment", full.names=TRUE)

get_scores <- function(fnames, n_char) {
  sent_json <- list(length(fnames))
  for( i in (1:length(fnames))){
    temp_json <- fromJSON(file=fnames[i])
    petid <- unlist(strsplit(substring(fnames[i], n_char), ".json"))
    temp_pair <- list(petid, temp_json[4][[1]][[1]], temp_json[4][[1]][[2]])
    sent_json[[i]] <- temp_pair
  }
  sent_df <- data.frame(matrix(unlist(sent_json), nrow=length(sent_json), byrow=T))
  return(sent_df)
}

train_sent_df <- get_scores(filenames_train, 22)
test_sent_df <- get_scores(filenames_test, 21)

colnames(train_sent_df) <- c("PetID", "score", "magnitude")
colnames(test_sent_df) <- c("PetID", "score", "magnitude")

###############################################################################

# create a variable to make splitting easier later on
train$df <- "train"
test$df <- "test"

train_index <- 1:nrow(train)
test_index <- 1:nrow(test)

train_test <- bind_rows(train, test) %>%
  mutate(AdoptionSpeed = as.factor(AdoptionSpeed),
         Type = ifelse(Type == 1, "Dog", "Cat"),
         Breed2 = ifelse(Breed2 == Breed1, 0, Breed2),
         BreedType = ifelse(Breed1 == 307 | Breed2 == 307, "Mixed", ifelse(Breed1 == 0 | Breed2 == 0, "Pure", "Cross")),
         HasName = as.numeric(ifelse(is.na(Name), FALSE, TRUE)),
         DescriptionLength = str_length(Description),
         HealthRating = (Vaccinated + Dewormed + Sterilized + Health),
         HealthVFM = Fee / HealthRating)


train_test <- train_test %>%
  mutate(DescriptionCharacterLength = str_length(Description),
         DescriptionSentencesCount = str_count(Description, "[[:alnum:] ][.!?]"),
         DescriptionWordCount = str_count(Description, "[[:alpha:][-]]+"),
         DescriptionCapitalsCount = str_count(Description, "[A-Z]"),
         DescriptionLettersCount = str_count(Description, "[A-Za-z]"),
         DescriptionPunctuationCount = str_count(Description, "[[:punct:]]"),
         DescriptionExclamationCount = str_count(Description, fixed("!")),
         DescriptionQuestionCount = str_count(Description, fixed("?")),
         DescriptionDigitsCount = str_count(Description, "[[:digit:]]"),
         DescriptionDistinctWordsCount = lengths(lapply(strsplit(Description, split = ' '), unique)),
         DescriptionLexicalDensity = DescriptionDistinctWordsCount / DescriptionWordCount)



train_test <- train_test %>%
  rowwise() %>%
  mutate(NumColours = sum(Color1 !=0, Color2 !=0, Color3 != 0)) %>% ungroup()


train_test_sent <- rbind(train_sent_df, test_sent_df)

train_test_sent$PetID <- as.character(train_test_sent$PetID)

train_test <- train_test %>%
  left_join(train_test_sent, by = "PetID") %>%
  mutate(score = as.numeric(as.character(score)),
         magnitude = as.numeric(as.character(magnitude)),
         score = ifelse(is.na(score), 0, score),
         magnitude = ifelse(is.na(magnitude), 0, magnitude))


# train_test <- train_test %>%
#   mutate(score = log(score + 1) * magnitude) %>%
#   select(-magnitude)

rescuer_count <- train_test %>%
  group_by(RescuerID) %>%
  summarise(RescuerCount = n()) %>% ungroup()

train_test <- train_test %>%
  left_join(rescuer_count, by = "RescuerID")


train_petID <- train$PetID
test_petID <- test$PetID

AdoptionSpeed_labels <- train$AdoptionSpeed

train_test$AdoptionSpeed <- NULL

train_test$PetID <- NULL


# # drop variables that wont be used in the model
# train_test <- train_test %>%
#   select(-Name, -Description, -RescuerID, -df)




# one-hot encode categorical variables and then join back to numeric training variables
train_test_num <- train_test %>%
  select_if(is.numeric)

train_test_type <- model.matrix(~ Type-1, train_test)

breed_type_train_test <- model.matrix(~ BreedType-1, train_test)

train_test_matrix <- cbind(train_test_num, train_test_type, breed_type_train_test) %>% as.matrix()


# multiply all columns by each other, first adding 1 to each column before multiplying
a <- train_test_matrix %>% data.frame()

res <- cbind(a^2, do.call(cbind,combn(colnames(a), 2, 
                                      FUN= function(x) list(a[x[1]] * a[x[2]]))))
colnames(res)[-(seq_len(ncol(a)))] <-  combn(colnames(a), 2, 
                                             FUN = paste, collapse=":")

res <- as.matrix(res)

# Join back to the full matrix for splitting and training
train_test_matrix <- cbind(train_test_matrix, res)

dim(train_test_matrix)

# split back out to train and test sets
train_matrix <- train_test_matrix[train_index,]

test_matrix <- train_test_matrix[-train_index,]

set.seed(1356)

# get the 70/30 training test split
numberOfTrainingSamples <- round(nrow(train_matrix) * .7)

# training data
train_data <- train_matrix[1:numberOfTrainingSamples,]
train_labels <- AdoptionSpeed_labels[1:numberOfTrainingSamples]

# testing data
test_data <- train_matrix[-(1:numberOfTrainingSamples),]
test_labels <- AdoptionSpeed_labels[-(1:numberOfTrainingSamples)]

# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = train_data, label= train_labels)
dtest <- xgb.DMatrix(data = test_data, label= test_labels)

# function to use as metric while training model
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- ScoreQuadraticWeightedKappa(labels,round(preds))
  return(list(metric = "kappa", value = err))}

#~~~~~~~~~~~~~~~~~~~~~~~~~
# Model Training
#~~~~~~~~~~~~~~~~~~~~~~~~~
model <- xgboost(data = dtrain, # the data   
                 max.depth = 3, # the maximum depth of each decision tree
                 nround = 1000, # number of boosting rounds
                 early_stopping_rounds = 30, # if we dont see an improvement in this many rounds, stop # max number of boosting iterations
                 objective = "multi:softmax",
                 eval_metric = evalerror,
                 num_class = 6,
                 gamma = 1, maximize = TRUE, print_every_n = 25)  # the objective function


pred <- predict(model, dtest) -1

evalerror(pred, dtest)


importance_matrix <- xgb.importance(colnames(train_matrix), model = model)

xgb.plot.importance(importance_matrix[1:30])

## eval metric: 0.3045839, kaggle leaderboard score: 0.340

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Prepare Test data for prediction
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

test_for_sub <- xgb.DMatrix(data = test_matrix)

sub_pred <- predict(model, test_for_sub) -1


as.data.frame(cbind(PetID = test_petID, AdoptionSpeed = sub_pred)) %>%
  write.csv("submission.csv", row.names = F)


