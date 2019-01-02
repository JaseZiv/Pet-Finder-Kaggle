source("src/feature engineering.R")


test_dog_id <- test %>%
  filter(Type == 1) %>% select(PetID) %>% pull()

test_cat_id <- test %>%
  filter(Type == 2) %>% select(PetID) %>% pull()

# split train and test into cat and dog
train_dog <- train %>%
  filter(Type == 1)  %>%
  select(-df, -Name, -RescuerID, -PetID, -Description)

test_dog <- test %>%
  filter(Type == 1) %>%
  select(-df, -Name, -RescuerID, -PetID, -Description, -AdoptionSpeed)


train_cat <- train %>%
  filter(Type == 2) %>%
  select(-df, -Name, -RescuerID, -PetID, -Description)

test_cat <- test %>%
  filter(Type == 2) %>%
  select(-df, -Name, -RescuerID, -PetID, -Description, -AdoptionSpeed)


#~~~~~~~~~~~~~~~
# Dogs
#~~~~~~~~~~~~~~~

train_label_dog <- train_dog$AdoptionSpeed

# Total number of rows in the joined_train data frame
n_dog <- nrow(train_dog)

# Number of rows for the joined_train training set (80% of the dataset)
n_train_dog <- round(0.80 * n_dog) 

# Create a vector of indices which is an 80% random sample
set.seed(123)
train_indices_dog <- sample(1:n_dog, n_train_dog)

# Subset the joined_train frame to training indices only
train_tr_dog <- train_dog[train_indices_dog, ]  

# Exclude the training indices to create the joined_train test set
train_te_dog <- train_dog[-train_indices_dog, ] 

train_te_dog_outcome <- train_te_dog$AdoptionSpeed

train_te_dog$AdoptionSpeed <- NULL


######################################
#---------- Model Training ----------#
######################################

model_rf_1_dog <- train(
  AdoptionSpeed ~ .,
  tuneLength = 1,
  data = train_tr_dog, method = "ranger",
  trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE)
)


predict_train_dog <- predict(model_rf_1_dog, train_te_dog)

confusionMatrix(train_label_dog[-train_indices_dog], predict_train_dog)

test_dog_AdoptionSpeed <- predict(model_rf_1_dog, test_dog)


#~~~~~~~~~~~~~~~
# Cats
#~~~~~~~~~~~~~~~

train_label_cat <- train_cat$AdoptionSpeed

# Total number of rows in the joined_train data frame
n_cat <- nrow(train_cat)

# Number of rows for the joined_train training set (80% of the dataset)
n_train_cat <- round(0.80 * n_cat) 

# Create a vector of indices which is an 80% random sample
set.seed(123)
train_indices_cat <- sample(1:n_cat, n_train_cat)

# Subset the joined_train frame to training indices only
train_tr_cat <- train_cat[train_indices_cat, ]  

# Exclude the training indices to create the joined_train test set
train_te_cat <- train_cat[-train_indices_cat, ] 

train_te_cat_outcome <- train_te_cat$AdoptionSpeed

train_te_cat$AdoptionSpeed <- NULL


######################################
#---------- Model Training ----------#
######################################

model_rf_1_cat <- train(
  AdoptionSpeed ~ .,
  tuneLength = 1,
  data = train_tr_cat, method = "ranger",
  trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE)
)


predict_train_cat <- predict(model_rf_1_cat, train_te_cat)

confusionMatrix(train_label_cat[-train_indices_cat], predict_train_cat)

test_cat_AdoptionSpeed <- predict(model_rf_1_cat, test_cat)


dog_sub <- cbind(test_dog_id, test_dog_AdoptionSpeed) %>% data.frame()

cat_sub <- cbind(test_cat_id, test_cat_AdoptionSpeed) %>% data.frame()

submission <- read_csv("../input/test/sample_submission.csv")


colnames(dog_sub) <- c("PetID", "AdoptionSpeed")

colnames(cat_sub) <- c("PetID", "AdoptionSpeed")


sub1 <- rbind(dog_sub, cat_sub)

submission <- submission %>%
  select(-AdoptionSpeed) %>%
  left_join(sub1, by = "PetID") %>%
  mutate(AdoptionSpeed = as.numeric(AdoptionSpeed) - 1)

write.csv(submission, "submissions/submission_rf_with_sentiment_calc_and_log_trans.csv", row.names = F)
