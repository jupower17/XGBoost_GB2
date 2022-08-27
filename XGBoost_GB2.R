# Adjust a GAM to obtain initial parameters
keep <- c(...) # Variables to keep to train the GAM model
sub <- dat[,names(dat) %in% keep]
gam_gb2 <- gamlss(INC_AB ~ ., family = GB2(), data = sub, n.cyc = 100000)

# Loss function
ll_gb2 <- function(preds, dtrain, ss = gam_gb2$sigma.fv[1], nn = gam_gb2$nu.fv[1], 
                   tt = gam_gb2$tau.fv[1])
{
  labels <- getinfo(dtrain, "label")
  ee <- exp(-ss*preds)
  grad <- 2*ss*(nn - (nn+tt)*(labels^ss*ee)/(1+labels^ss*ee))
  hess <- 2*(ss^2*(nn+tt)*labels^ss*ee)/(1+labels^ss*ee)^2
  return(list(grad = grad, hess = hess))
}

# Evaluation function
evalerror <- function(preds, dtrain, ss = gam_gb2$sigma.fv[1], nn = gam_gb2$nu.fv[1],
                      tt = gam_gb2$tau.fv[1]) 
{
  labels <- getinfo(dtrain, "label")
  err <- sum(ss*nn*preds + (nn+tt)*log(1+(labels/exp(preds))^ss))
  return(list(metric = "Negll_GB2", value = err))
}

# Optimization of hyperparameters
# Optimization of depth and gamma, with larger learning rate
dep <- 1:4
gam <- c(0,0.5,1)

mse <- matrix(nrow = length(dep), ncol = length(gam))

set.seed(54738)
for (i in 1:length(dep))
{
  d <- dep[i]
  for (j in 1:length(gam))
  {
    g <- gam[j]
    mod_xgb_cv1 <- xgb.cv(params = list(eta = 0.1, 
                                        max.depth = d, 
                                        gamma = g),
                          data = dat_xgb_train,
                          nrounds = 10000,
                          obj = ll_gb2,
                          feval = evalerror,
                          subsample = 0.75,
                          colsample_by_tree = 0.75,
                          min_child_weight = 5,
                          nfold = 5,
                          early_stopping_rounds = 100,
                          maximize = FALSE,
                          verbose = TRUE)
    mse[i,j] <- mod_xgb_cv1$evaluation_log$test_Negll_GB2_mean
    [mod_xgb_cv1$best_iteration]
  }
}

best_depth <- dep[which(mse == min(mse), arr.ind=TRUE)[1]]
best_gamma <- gam[which(mse == min(mse), arr.ind=TRUE)[2]]

# Optimization of number of trees with smaller learning rate
set.seed(263256)
mod_xgb_cv2 <- xgb.cv(params = list(eta = 0.01, 
                                    max.depth = best_depth, 
                                    gamma = best_gamma),
                      data = dat_xgb_train,
                      nrounds = 10000,
                      obj = ll_gb2,
                      feval = evalerror,
                      subsample = 0.75,
                      colsample_by_tree = 0.75,
                      min_child_weight = 5,
                      nfold = 5,
                      early_stopping_rounds = 100,
                      maximize = FALSE,
                      verbose = TRUE)
proc.time() - ptm
ntrees <- mod_xgb_cv2$best_ntreelimit

# Training of final model
set.seed(647398)
ptm <- proc.time()
mod_xgb <- xgb.train(data = dat_xgb_train,
                     max.depth = best_depth, 
                     eta = 0.01,
                     gamma = best_gamma,
                     nrounds = ntrees,
                     objective = ll_gb2,
                     subsample = 0.75,
                     colsample_bytree = 0.75,
                     min_child_weight = 5,
                     verbose = TRUE)
proc.time() - ptm

# Estimation of parameters by maximum likelihood
ll <- function(par, dat = dat, mod = mod_xgb)
{
  sparse_xgb <- sparse.model.matrix(Y~.-1, data = dat)
  m <- exp(predict(mod, newdata = sparse_xgb))
  s <- par[1]
  n <- par[2]
  t <- par[3]
  y <- dat$Y
  sum(log(s) + (s*n-1)*log(y) - s*n*log(m) - (n+t)*log(1+(y/m)^s) -
        log(beta(n,t)))*-1
}

op <- optim(par = c(gam_gb2$sigma.fv[1], gam_gb2$nu.fv[1], gam_gb2$tau.fv[1]), 
            fn = ll)
ss_xgb <- op$par[1]
nn_xgb <- op$par[2]
tt_xgb <- op$par[3]

# Prédictions
c2 <- beta(nn_xgb+1/ss_xgb,tt_xgb-1/ss_xgb)/beta(nn_xgb,tt_xgb)
sparse_xgb_train <- sparse.model.matrix(Y~.-1, data = dat)
preds_train <- c2 * exp(predict(mod_xgb, newdata = sparse_xgb_train))0