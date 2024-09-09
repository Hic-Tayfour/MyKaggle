# Carregar pacotes necessários
library(tidymodels)
library(ggplot2)
library(yardstick)

# 1. Carregar os dados
train_data <- read.csv("train.csv")  # Conjunto de treino
test_data <- read.csv("test.csv")    # Conjunto de teste

# 2. Converter a variável Survived para fator (importante para classificação)
train_data$Survived <- as.factor(train_data$Survived)

# 3. Dividir o conjunto de treinamento em treino (80%) e validação (20%)
set.seed(123)
data_split <- initial_split(train_data, prop = 0.8, strata = Survived)

# Criar os conjuntos de treino e validação
train_data_split <- training(data_split)
valid_data_split <- testing(data_split)

# 4. Receita de pré-processamento dos dados
titanic_recipe <- recipe(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
                         data = train_data_split) %>%
  step_impute_median(Age, Fare) %>%     # Imputar valores faltantes em Age e Fare
  step_novel(Embarked) %>%              # Lidar com novos níveis em Embarked
  step_unknown(Embarked) %>%            # Tratar valores desconhecidos em Embarked
  step_dummy(all_nominal_predictors()) %>%  # Converter variáveis categóricas em dummies
  step_zv(all_predictors()) %>%        # Remover colunas com variância zero
  step_normalize(all_numeric_predictors())  # Normalizar variáveis numéricas

# Preparar a receita
prepped_recipe <- prep(titanic_recipe)

# 5. Especificação do modelo de Regressão Logística
logistic_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# 6. Workflow: combinar a receita e o modelo de regressão logística
titanic_workflow <- workflow() %>%
  add_recipe(titanic_recipe) %>%
  add_model(logistic_model)

# 7. Ajuste do modelo ao conjunto de treino
fit_logistic <- fit(titanic_workflow, data = train_data_split)

# 8. Fazer previsões no conjunto de validação
valid_class_preds <- predict(fit_logistic, new_data = valid_data_split) %>%
  bind_cols(valid_data_split)

# Previsões de probabilidade para a Curva ROC
valid_prob_preds <- predict(fit_logistic, new_data = valid_data_split, type = "prob") %>%
  bind_cols(valid_data_split)

# 9. Avaliar a acurácia
accuracy_result <- accuracy(valid_class_preds, truth = Survived, estimate = .pred_class)
print(accuracy_result)

# Avaliar o AUC (Área Sob a Curva ROC)
auc_result <- roc_auc(valid_prob_preds, truth = Survived, .pred_1, event_level = "second")
print(auc_result)

# 10. Gerar a Curva ROC
roc_curve_data <- roc_curve(valid_prob_preds, truth = Survived, .pred_1, event_level = "second")

# Plotar a Curva ROC
ggplot(roc_curve_data, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "blue") +
  geom_abline(lty = 2) +  # Linha de referência
  labs(title = "Curva ROC - Regressão Logística no Titanic", x = "1 - Especificidade", y = "Sensibilidade") +
  theme_minimal()

# 11. Previsões no conjunto de teste para submissão
test_class_preds <- predict(fit_logistic, new_data = test_data) %>%
  bind_cols(test_data)

# 12. Preparar o arquivo de submissão
submission <- test_class_preds %>%
  select(PassengerId, Survived = .pred_class)

# Converter as previsões para formato binário
submission$Survived <- as.integer(as.character(submission$Survived))

# 13. Escrever o arquivo de submissão
write.csv(submission, "submission.csv", row.names = FALSE)
