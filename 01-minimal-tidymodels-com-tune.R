# codigo enxuto de tidymodels
library(tidymodels)




# VOCABULÁRIO ##################
# PASSO 1) split
# - initial_split/training/testing
# PASSO 3) dataprep
# - recipe/step_log/step_other/step_other/prep/bake
# PASSO 4) Especificação do modelo
# - decision_tree/set_engine/set_mode
# PASSO 5) Montagem do workflow
# - workflow/add_model/add_recipe
# PASSO 6) Estratégia de reamostragem
# - vfold_cv
# PASSO 7) Tunagem de Hiperparâmetros
# - tune_grid/autoplot/select_best/finalize_workflow
# PASSO 8) Avaliação do desempenho na base de teste
# - last_fit/metric_set/collect_metrics/collect_predictions/mae/rmse/vip
# PASSO 9) Ajuste do modelo final na base inteira
# - fit/predict





# PASSO 0) import ---------------------------------------------------------
data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))

# PASSO 1) split ----------------------------------------------------------
set.seed(123)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)

# PASSO 2) Exploração -----------------------------------------------------

# Pulamos!

# PASSO 3) dataprep -------------------------------------------------------
ames_rec <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01)

ames_rec_preparada <- prep(ames_rec)
bake(ames_rec_preparada, new_data = ames_train)
bake(ames_rec_preparada, new_data = ames_test)

# PASSO 4) Especificação do modelo ----------------------------------------
dt_model <-
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune(),
    min_n = tune()
  ) %>%
  set_engine("rpart") %>%
  set_mode("regression")

# PASSO 5) Montagem do workflow -------------------------------------------
dt_wflow <-
  workflow() %>%
  add_model(dt_model) %>%
  add_recipe(ames_rec)

# PASSO 6) Estratégia de reamostragem -------------------------------------
reamostragens <- vfold_cv(ames_train, v = 5)

# PASSO 7) Tunagem de Hiperparâmetros -------------------------------------
dt_tune <- tune_grid(
  dt_wflow,
  resamples = reamostragens,
  grid = 10
)

autoplot(dt_tune)
dt_best <- select_best(dt_tune, metric = "rmse")
dt_wflow <- dt_wflow %>% finalize_workflow(dt_best)

# PASSO 8) Avaliação do desempenho na base de teste -----------------------
dt_fit_no_treino <- dt_wflow %>% last_fit(ames_split, metrics = metric_set(rmse, mae, mase, mape, rsq))
collect_metrics(dt_fit_no_treino)
obs_vs_pred <- collect_predictions(dt_fit_no_treino)
obs_vs_pred %>% mae(Sale_Price, .pred)
obs_vs_pred %>% rmse(Sale_Price, .pred)
obs_vs_pred %>% rsq(Sale_Price, .pred)
qplot(.pred, Sale_Price, data = obs_vs_pred) + geom_abline(colour = "red", size = 2)
vip(dt_fit_no_treino$.workflow[[1]]$fit$fit)

# PASSO 9) Ajuste do modelo final na base inteira -------------------------
dt_fit <- fit(dt_wflow, ames)
predict(dt_fit, ames)

# PASSO 10) salva tudo ----------------------------------------------------
saveRDS(dt_fit, "dt_fit.rds")
saveRDS(dt_fit_no_treino, "dt_fit_no_treino.rds")


















