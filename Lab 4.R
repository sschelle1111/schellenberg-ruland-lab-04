#### Loading libraries and dataset ####

library(glmmTMB)
library(dplyr)
library(tidyr)
library(ggplot2)
library(broom.mixed)
library(performance)
library(gt)
library(readr)
library(DHARMa)
library(ggeffects)
library(titanic)

titanic <- titanic_train

#### Data Cleaning ####

titanic <- titanic |>
  select(Survived, Sex, Age, Pclass) |>
  drop_na() |>
  mutate(
    Sex.Binary = case_when(
      Sex == "male" ~ 0,
      Sex == "female" ~ 1)) |>
  mutate(
    Survived.Char = case_when(
      Survived == 0 ~ "Died",
      Survived == 1 ~ "Survived")) |>
  mutate(
    Pclass.Char = case_when(
      Pclass == "1" ~ "First Class",
      Pclass == "2" ~ "Second Class",
      Pclass == "3" ~ "Third Class")) |>
  mutate(Pclass = factor(Pclass, levels = c(1, 2, 3))) |>
  select(Survived, Survived.Char, Sex, Sex.Binary, Pclass, Pclass.Char, Age)

n <- nrow(titanic)



#### Explore Data ####

#descriptive statistics for continuous variable

descriptives <- titanic |>
  summarise(
    across(
      Age,
      list(mean = mean, sd = sd, min = min, max = max),
      na.rm = TRUE
    )
  ) |>
  pivot_longer(
    everything(),
    names_to = c("variable", ".value"),
    names_sep = "_"
  )

descriptives |>
  gt() |>
  fmt_number(columns = where(is.numeric), decimals = 2)

#Counts for categorical variables

categorical_counts <- titanic |>
  mutate(
    across(c(Survived, Pclass, Sex), as.character)
  ) |>
  pivot_longer(
    cols = c(Survived, Pclass, Sex),
    names_to = "variable",
    values_to = "category"
  ) |>
  count(variable, category)

categorical_counts |>
  gt()




#### Specify and estimate four models ####


#Intercept only
m0 <- glmmTMB(Survived ~ 1, data = titanic,
                family = binomial(link = "logit"))

#Sex
m1 <- glmmTMB(
  Survived ~ Sex,
  data = titanic,
  family = binomial(link = "logit"))

#Sex + Passenger class
m2 <- glmmTMB(
  Survived ~ Sex + Pclass,
  data = titanic,
  family = binomial(link = "logit"))

#Sex + Passenger class + Age
m3 <- glmmTMB(
  Survived ~ Sex + Pclass + Age,
  data = titanic,
  family = binomial(link = "logit"))

#### Compare Model Performance ####

#compare nested tests
tests <- anova(m0, m1, m2, m3)

#keep the output we're interested in
tests <- tests |>
  select(Chisq, `Chi Df`, `Pr(>Chisq)`) |>
  rename(
    Nested.df = `Chi Df`,
    Nested.Chisq.stat = Chisq,
    Nested.p.value = `Pr(>Chisq)`)

#extract overall model performance
glance0 <- broom.mixed::glance(m0)
glance1 <- broom.mixed::glance(m1)
glance2 <- broom.mixed::glance(m2)
glance3 <- broom.mixed::glance(m3)

#combine results using rbind
mods <- c(
  "m0: Null",
  "m1: Sex",
  "m2: + Passenger Class",
  "m3: + Age")

glances <- rbind(
  glance0,
  glance1,
  glance2,
  glance3)

#combine nested tests and model performance
comparison_table <- cbind(
  Model = mods,
  glances,
  tests)

#display combined model and comparison table

formatted_table <- comparison_table |>
  select(
    Model,
    AIC,
    BIC,
    logLik,
    deviance,
    df.residual,
    Nested.df,
    Nested.Chisq.stat,
    Nested.p.value
  ) |>
  gt() |>
  cols_label(
    Model = "Model",
    AIC = "AIC",
    BIC = "BIC",
    logLik = "Log-likelihood",
    deviance = "Deviance",
    df.residual = "Residual.df",
    Nested.df = "Nested df",
    Nested.Chisq.stat = "Nested χ²",
    Nested.p.value = "Nested p"
  ) |>
  fmt_number(
    columns = c(
      AIC,
      BIC,
      logLik,
      deviance,
      Nested.Chisq.stat
    ),
    decimals = 3
  ) |>
  fmt(
    columns = Nested.p.value,
    fns = function(x) {
      ifelse(
        is.na(x),
        "—",
        ifelse(x < .001, "< .001", sprintf("%.3f", x))
      )
    }
  ) |>
  sub_missing(
    columns = c(Nested.df, Nested.Chisq.stat),
    missing_text = "—"
  )

formatted_table
#### Interpreting the Combined Table ####

#### Selecting a Model ####

#Use BIC for selection
selected_row <- comparison_table |>
  arrange(desc(BIC)) |>
  slice_tail(n = 1)

#creating an object with our selected model's name for dynamic reporting
selected_name <- selected_row$Model

#case matching for dynamic reporting
selected_model <- switch(
  selected_name,
  "m0: Null" = m0,
  "m1: Sex" = m1,
  "m2: + Passenger Class" = m2,
  "m3: + Age" = m3)


#Based on our comparison table, the provisionally selected model is **`r selected_name`**.
#Its performance is:
  
  #AIC = **`r round(AIC(selected_model), 3)`**
  #BIC = **`r round(BIC(selected_model), 3)`**
  
  #We will now examine whether the selected model has important diagnostic problems.



#### Evaluating Regression Diagnostics####

#Linearity on logit scale
set.seed(9041)
res_m3 <- simulateResiduals(m3, n = 500, plot = FALSE)
plot(res_m3)

#colinearity 
check_collinearity(m3)

#VIF
vif_plot <- plot(check_collinearity(m3))

vif_plot

#outliers
testOutliers(res_m3)

#events per predictor
n_event <- sum(titanic$Survived == 1)
n_nonevent <- sum(titanic$Survived == 0)
n_coef <- length(fixef(m3)$cond)

#calculating events per predictor for each group
events_per_predictor_survived <- n_event / n_coef
events_per_perdictor_died <- n_nonevent / n_coef

#table of events per predictor
epp_table <- tibble(
  Group = c(
    "Survivors",
    "Non-survivors"
  ),
  Value = c(
    events_per_predictor_survived,
    events_per_perdictor_died
  )
)

epp_table |>
  gt() |>
  fmt_number(
    columns = Value,
    decimals = 2
  ) |>
  cols_label(
    Group = "Group",
    Value = "Events per Predictor"
  ) |>
  tab_header(
    title = "Events per Predictor"
  )

#Diagnostic paragraph

#### Create Coefficient table####


#selection paragraph

#extract coefficients
coefficient_results <- broom.mixed::tidy(
  selected_model,
  effects = "fixed",
  component = "cond",
  conf.int = TRUE,
  conf.level = 0.95)

#Create coefficient table

coefficient_results |>
  select(
    term,
    estimate,
    std.error,
    conf.low,
    conf.high,
    p.value
  ) |>
  gt() |>
  cols_label(
    term = "Predictor",
    estimate = "Estimate",
    std.error = "SE",
    conf.low = "95% CI lower",
    conf.high = "95% CI upper",
    p.value = "p"
  ) |>
  fmt_number(
    columns = c(
      estimate,
      std.error,
      conf.low,
      conf.high
    ),
    decimals = 3
  ) |>
  fmt(
    columns = p.value,
    fns = function(x) {
      ifelse(
        x < .001,
        "< .001",
        sprintf("%.3f", x)
      )
    }
  )


#### Interpretation####

#log-odds
fixef(m3)

#transforming to odds-ratios
exp(fixef(m3)$cond)

#odds-ratio CIs
or_ci <- exp(confint(m3, parm = "beta_"))
or_ci


####overall model performance####

selected_row |>
  select(
    Model,
    AIC,
    BIC,
    logLik,
    deviance,
    df.residual,
    Nested.df,
    Nested.Chisq.stat,
    Nested.p.value
  ) |>
  gt() |>
  cols_label(
    Model = "Model",
    AIC = "AIC",
    BIC = "BIC",
    logLik = "Log-likelihood",
    deviance = "Deviance",
    df.residual = "Residual df",
    Nested.df = "Nested df",
    Nested.Chisq.stat = "Nested χ²",
    Nested.p.value = "Nested p"
  ) |>
  fmt_number(
    columns = where(is.numeric),
    decimals = 2
  )

#### Visualization ####

#visualizing age and survival

pred_age <- ggpredict(m3, terms = "Age [0:80 by=0.5]")

ggplot(pred_age, aes(x, predicted)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.18) +
  geom_line(linewidth = 1.2) +
  labs(
    x = "Age",
    y = "Predicted Probability of Survival"
  ) +
  theme_classic()


