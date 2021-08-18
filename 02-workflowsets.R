library(tidymodels)
library(workflowsets)

# receitas ----------------------------------------------------------------
receita_base <- recipe(Status ~ ., data = credit_train)

receita_base_impute_jeito1 <- receita_base %>%
  step_mutate(
    Home = factor(ifelse(is.na(Home), "other", as.character(Home))),
    Job = factor(ifelse(is.na(Job), "partime", as.character(Job))),
    Marital = factor(ifelse(is.na(Marital), "widow", as.character(Marital))),
    Assets = ifelse(is.na(Assets), min(Assets, na.rm = TRUE), Assets),
    Income = ifelse(is.na(Income), min(Income, na.rm = TRUE), Income),
    Debt = ifelse(is.na(Debt), 0, Debt)
  ) %>%
  step_normalize(all_numeric()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_novel(all_nominal_predictors())

receita_base_impute_jeito2 <- receita_base %>%
  step_impute_mode(Home, Job, Marital) %>%
  step_impute_mean(Debt, Income, Assets) %>%
  step_normalize(all_numeric()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_novel(all_nominal_predictors())

# modelos -----------------------------------------------------------------

# <mostrar o addin do parsnip>

boost_tree_xgboost_spec <-
  boost_tree(tree_depth = tune(),
             trees = 100,
             learn_rate = tune(),
             min_n = 4,
             loss_reduction = tune(),
             sample_size = tune(),
             stop_iter = 20) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

linear_reg_glmnet_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine('glmnet') %>%
  set_mode('classification')

nearest_neighbor_kknn_spec <- nearest_neighbor(
    neighbors = tune(),
    weight_func = tune(),
    dist_power = tune()
  ) %>%
  set_engine('kknn') %>%
  set_mode('classification')

rand_forest_ranger_spec <-
  rand_forest(mtry = tune(), min_n = tune(), trees = 100) %>%
  set_engine('ranger') %>%
  set_mode('classification')



# workflowset ------------------------------------------------------------
wfset <- workflow_set(
    preproc = list(
      jeito1 = receita_base_impute_jeito1,
      jeito2 = receita_base_impute_jeito2
    ),

    models = list(
      xgb = boost_tree_xgboost_spec,
      linear = linear_reg_glmnet_spec,
      knn = nearest_neighbor_kknn_spec,
      rf = rand_forest_ranger_spec
    )
  )

grid_control <- control_grid(
  allow_par = TRUE,
  parallel_over = "everything",
  save_pred = TRUE,
  save_workflow = TRUE
)

registerDoFuture()
plan(multisession)
tictoc::tic("wfset")
grid_results <- workflow_map(
  wfset,
  seed = 1503,
  resamples = credit_resamples,
  grid = 3,
  control = grid_control
)
tictoc::toc()
plan(sequential)
autoplot(grid_results)
autoplot(
  grid_results,
  rank_metric = "roc_auc",  # <- como ordenar os modelos
  metric = "roc_auc",       # <- qual métrica visualizar
  select_best = TRUE     # <- um ponto por workflow
)

collect_metrics(grid_results)
autoplot(grid_results, id = "jeito1_xgb", metric = "roc_auc")

best_results <- grid_results %>%
  extract_workflow_set_result("jeito1_xgb") %>%
  select_best(metric = "roc_auc")

boosting_test_results <- grid_results %>%
  pull_workflow("jeito1_xgb") %>%
  finalize_workflow(best_results) %>%
  last_fit(split = credit_initial_split)

collect_metrics(boosting_test_results)
