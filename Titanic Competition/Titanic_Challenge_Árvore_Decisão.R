# Carregar pacotes necessários
library(tidymodels)
library(ggplot2)
library(rpart.plot)  # Pacote necessário para visualizar a árvore de decisão

# 1. Carregar os dados
train_data <- read.csv("train.csv")  # Conjunto de treino
test_data <- read.csv("test.csv")    # Conjunto de teste

# 2. Converter a variável Survived para fator (importante para classificação)
train_data$Survived <- as.factor(train_data$Survived)

# 3. Dividir o conjunto de treinamento em treino (80%) e validação (20%)
set.seed(123)
data_split <- initial_split(train_data, prop = 0.8, strata = Survived)
train_data_split <- training(data_split)
valid_data_split <- testing(data_split)

# 4. Receita de pré-processamento dos dados
titanic_recipe <- recipe(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
                         data = train_data_split) %>%
  step_impute_median(Age, Fare) %>%     # Imputar valores faltantes em Age e Fare
  step_dummy(all_nominal_predictors()) %>%  # Converter variáveis categóricas em dummies
  step_zv(all_predictors())             # Remover colunas com variância zero

# Preparar a receita
prepped_recipe <- prep(titanic_recipe)

# 5. Especificação do modelo de Árvore de Decisão (com model=TRUE)
decision_tree_model <- decision_tree() %>%
  set_engine("rpart", model = TRUE) %>%
  set_mode("classification")

# 6. Workflow: combinar a receita e o modelo de árvore de decisão
titanic_workflow <- workflow() %>%
  add_recipe(titanic_recipe) %>%
  add_model(decision_tree_model)

# 7. Ajustar o modelo ao conjunto de treino
fit_decision_tree <- fit(titanic_workflow, data = train_data_split)

# 8. Fazer previsões de probabilidade no conjunto de validação
valid_prob_preds <- predict(fit_decision_tree, new_data = valid_data_split, type = "prob") %>%
  bind_cols(valid_data_split)

# Previsões de classe
valid_class_preds <- predict(fit_decision_tree, new_data = valid_data_split) %>%
  bind_cols(valid_data_split)

# 9. Avaliar o AUC (Área Sob a Curva ROC)
auc_result <- roc_auc(valid_prob_preds, truth = Survived, .pred_1, event_level = "second")
print(paste("AUC: ", auc_result))

# 10. Gerar e Plotar a Curva ROC
roc_curve_data <- roc_curve(valid_prob_preds, truth = Survived, .pred_1, event_level = "second")

ggplot(roc_curve_data, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "blue") +
  geom_abline(lty = 2) +  # Linha de referência
  labs(title = "Curva ROC - Árvore de Decisão no Titanic", x = "1 - Especificidade", y = "Sensibilidade") +
  theme_minimal()

# 11. Exibir a árvore de decisão
final_tree <- extract_fit_engine(fit_decision_tree)  # Extraindo o modelo rpart
rpart.plot(final_tree, type = 3, extra = 101, fallen.leaves = TRUE)

# 12. Previsões no conjunto de teste para submissão
test_class_preds <- predict(fit_decision_tree, new_data = test_data) %>%
  bind_cols(test_data)

# 13. Preparar o arquivo de submissão
submission <- test_class_preds %>%
  select(PassengerId, Survived = .pred_class)

# Converter as previsões para formato binário (0 ou 1)
submission$Survived <- as.integer(as.character(submission$Survived))

# 14. Escrever o arquivo de submissão
write.csv(submission, "submission.csv", row.names = FALSE)
