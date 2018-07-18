
library(dplyr)
library(caret)
library(rstan)

raw_data <- read.csv("law_data.csv")
law <- dplyr::select(raw_data, race, sex, LSAT, UGPA, region_first, ZFYA, sander_index, first_pf) 
law <- law[law$region_first != "PO",]
law$region_first <- factor(law$region_first)

law$amerind <- as.numeric(law$race == "Amerindian")
law$asian   <- as.numeric(law$race == "Asian")
law$black   <- as.numeric(law$race == "Black")
law$hisp    <- as.numeric(law$race == "Hispanic")
law$mexican <- as.numeric(law$race == "Mexican")
law$other   <- as.numeric(law$race == "Other")
law$puerto  <- as.numeric(law$race == "Puertorican")
law$white   <- as.numeric(law$race == "White")

law$female    <- as.numeric(law$sex == 1)
law$male      <- as.numeric(law$sex == 2)

sense_cols <- c("amerind", "asian", "black", "hisp", "mexican", "other", "puerto", "white", "male", "female")

set.seed(0)
trainIndex <- createDataPartition(law$first_pf, p = .8, 
                                  list = FALSE, 
                                  times = 1)
lawTrain <- law[trainIndex,]
lawTest  <- law[-trainIndex,]

#n <- nrow(df2)
n <- nrow(lawTrain)
ne <- nrow(lawTest)


lawTrain$LSAT <- round(lawTrain$LSAT)
lawTest$LSAT <- round(lawTest$LSAT)

# don't fit model transductively
# ------------------------------
law_stan_train <- list(N = n, K = length(sense_cols), a = data.matrix(lawTrain[,sense_cols]), 
                          ugpa = lawTrain[,c("UGPA")], lsat = lawTrain[,c("LSAT")], zfya = lawTrain[,c("ZFYA")])


fit_law_train <- stan(file = 'law_school_train.stan', data = law_stan_train, iter = 2000, chains = 1, verbose = TRUE)
# Extract information

la_law_train <- extract(fit_law_train, permuted = TRUE)
#u_te_samp <- colMeans(la_law_train$u_TE)
U_TRAIN   <- colMeans(la_law_train$u)

save(la_law_train,file='law_school_l_stan_train.Rdata')

ugpa0      <- mean(la_law_train$ugpa0)
eta_u_ugpa <- mean(la_law_train$eta_u_ugpa)
eta_a_ugpa <- colMeans(la_law_train$eta_a_ugpa)

lsat0      <- mean(la_law_train$lsat0)
eta_u_lsat <- mean(la_law_train$eta_u_lsat)
eta_a_lsat <- colMeans(la_law_train$eta_a_lsat)

SIGMA_G <- mean(la_law_train$sigma_g)
#----------------------------------
law_stan_test <- list(N = ne, K = length(sense_cols), a = data.matrix(lawTest[,sense_cols]),
                      ugpa = lawTest[,c("UGPA")], lsat = lawTest[,c("LSAT")],
                      ugpa0 = ugpa0, eta_u_ugpa = eta_u_ugpa, eta_a_ugpa = eta_a_ugpa,
                      lsat0 = lsat0, eta_u_lsat = eta_u_lsat, eta_a_lsat = eta_a_lsat,
                      sigma_g = SIGMA_G)


fit_law_test <- stan(file = 'law_school_only_u.stan', data = law_stan_test, iter = 2000, chains = 1, verbose = TRUE)
la_law_test <- extract(fit_law_test, permuted = TRUE)
#u_te_samp <- colMeans(la_law_train$u_TE)
U_TEST   <- colMeans(la_law_test$u)


save(la_law_test,file='law_school_l_stan_test.Rdata')


# Classifiers on data
# -------------------

# all features (unfair)
X_U <- as.data.frame(data.matrix(lawTrain[,sense_cols]))
X_U$ZFYA <- lawTrain$ZFYA
X_U$LSAT <- lawTrain$LSAT
X_U$UGPA <- lawTrain$UGPA

X_U_TE <- as.data.frame(data.matrix(lawTest[,sense_cols]))
X_U_TE$ZFYA <- lawTest$ZFYA
X_U_TE$LSAT <- lawTest$LSAT
X_U_TE$UGPA <- lawTest$UGPA

model_u <- lm(ZFYA ~ LSAT + UGPA + amerind + asian + black + hisp + mexican + other + puerto + white + male + female + 1, data=X_U)
pred_u <- predict.glm(model_u)
pred_u_te <- predict(model_u, newdata=X_U_TE)
rmse_u_te <- sqrt( sum( (pred_u_te - X_U_TE$ZFYA)^2 ) / nrow(X_U_TE) ) # test error
print('unfair full model:')
print(rmse_u_te)


# unaware (unfair)
model_un <- lm(ZFYA ~ LSAT + UGPA + 1, data=X_U)
pred_un <- predict.glm(model_un)
pred_un_te <- predict(model_un, newdata=X_U_TE)
rmse_un_te <- sqrt( sum( (pred_un_te - X_U_TE$ZFYA)^2 ) / nrow(X_U_TE) ) # test error
print('unfair unaware model:')
print(rmse_un_te)


# fair, non-deterministic model
X_F <- data.frame(u=U_TRAIN, ZFYA=lawTrain$ZFYA)
X_F_TE <- data.frame(u=U_TEST, ZFYA=lawTest$ZFYA)

model_f <- lm(ZFYA ~ u + 1, data=X_F)

pred_f <- predict.glm(model_f)
pred_f_te <- predict.glm(model_f, newdata=X_F_TE)
rmse_f_te <- sqrt( sum( (pred_f_te - X_F_TE$ZFYA)^2 ) / nrow(X_F_TE) ) # test error
print('fair non-deterministic model:')
print(rmse_f_te)


# fair, deterministic model
model_ugpa <- lm(UGPA ~ amerind + asian + black + hisp + mexican + other + puerto + white + male + female + 1, data=lawTrain) # regress UGPA on race, sex
model_lsat <- lm(LSAT ~ amerind + asian + black + hisp + mexican + other + puerto + white + male + female + 1, data=lawTrain) # regress LSAT on race, sex

#resid
lawTrain$resid_UGPA = lawTrain$UGPA - predict(model_ugpa, newdata=lawTrain)
lawTrain$resid_LSAT = lawTrain$LSAT - predict(model_lsat, newdata=lawTrain)

model_det <- lm(ZFYA ~ resid_UGPA + resid_LSAT + 1, data=lawTrain)

model_ugpa_te <- lm(UGPA ~ amerind + asian + black + hisp + mexican + other + puerto + white + male + female + 1, data=lawTest)
model_lsat_te <- lm(LSAT ~ amerind + asian + black + hisp + mexican + other + puerto + white + male + female + 1, data=lawTest)

#resid
lawTest$resid_UGPA = lawTest$UGPA - predict(model_ugpa_te, newdata=lawTest)
lawTest$resid_LSAT = lawTest$LSAT - predict(model_lsat_te, newdata=lawTest)


pred_det_te <- predict(model_det, newdata=lawTest)
rmse_det_te <- sqrt( sum( (pred_det_te - lawTest$ZFYA)^2 ) / nrow(lawTest) ) # test error
print('fair deterministic model:')
print(rmse_det_te)