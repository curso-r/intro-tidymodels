# https://www.tmwr.org/
# codigo enxuto de tidymodels
library(tidymodels)
data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(123)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)

ames_rec <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

lm_model <-
  linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

lm_wflow <-
  workflow() %>%
  add_model(lm_model) %>%
  add_recipe(ames_rec)

lm_fit <- fit(lm_wflow, ames_train)

predict(lm_fit, ames_test)











##################
# vocabualrio
# - initial_split/training/testing
# - recipe/step_log/step_other/step_other/step_dummy/step_interact/step_ns
# - linear_reg/set_engine/set_mode
# - workflow/add_model/add_recipe
# - fit
# - predict
