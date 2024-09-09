# Carregar pacotes necessários
library(tidymodels)
library(ggplot2)
library(rpart.plot)  # Pacote necessário para visualizar a árvore de decisão

# 1. Carregar os dados
train_data <- read.csv("train.csv")  # Conjunto de treino
test_data <- read.csv("test.csv")    # Conjunto de teste (para submissão)

# 2. Converter a variável Survived para fator (importante para classificação)
train_data$Survived <- as.factor(train_data$Survived)

# 3. Receita de pré-processamento dos dados (prever Survived)
titanic_recipe <- recipe(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
                         data = train_data) %>%
  step_impute_median(Age, Fare) %>%     # Imputar valores faltantes em Age e Fare
  step_dummy(all_nominal_predictors()) %>%  # Converter variáveis categóricas em dummies
  step_zv(all_predictors())             # Remover colunas com variância zero

# Preparar a receita
prepped_recipe <- prep(titanic_recipe)

# 4. Especificação do modelo de árvore de decisão (classificação)
decision_tree_model <- decision_tree() %>%
  set_engine("rpart", model = TRUE) %>%
  set_mode("classification")

# 5. Workflow: combinar a receita e o modelo de árvore de decisão
titanic_workflow <- workflow() %>%
  add_recipe(titanic_recipe) %>%
  add_model(decision_tree_model)

# 6. Ajustar o modelo ao conjunto de treino
fit_decision_tree <- fit(titanic_workflow, data = train_data)

# 7. Fazer previsões no conjunto de teste
test_predictions <- predict(fit_decision_tree, new_data = test_data) %>%
  bind_cols(test_data)

# 8. Preparar o arquivo de submissão
submission <- test_predictions %>%
  select(PassengerId, Survived = .pred_class)

# Converter as previsões para formato binário (0 ou 1)
submission$Survived <- as.integer(as.character(submission$Survived))

# 9. Escrever o arquivo de submissão
write.csv(submission, "submission.csv", row.names = FALSE)

# 10. Exibir a árvore de decisão
final_tree <- extract_fit_engine(fit_decision_tree)  # Extraindo o modelo rpart
rpart.plot(final_tree, type = 3, extra = 101, fallen.leaves = TRUE)
