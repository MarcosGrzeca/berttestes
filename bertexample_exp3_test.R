#Source: https://blogs.rstudio.com/ai/posts/2019-09-30-bert-r/

library(zeallot)
library(dplyr)

Sys.setenv(TF_KERAS=1) 
# make sure we use python 3
reticulate::use_python('/usr/bin/python3', required=T)
# to see python version
reticulate::py_config()


pretrained_path = '/var/www/html/uncased_L-12_H-768_A-12'
config_path = file.path(pretrained_path, 'bert_config.json')
checkpoint_path = file.path(pretrained_path, 'bert_model.ckpt')
vocab_path = file.path(pretrained_path, 'vocab.txt')


seq_length = 38L
bch_size = 64
epochs = 3
learning_rate = 1e-4

DATA_COLUMN = 'textEmbedding'
LABEL_COLUMN = 'resposta'

df = data.table::fread('planilhas/exp3.csv')

library("caret")

resultados <- data.frame(matrix(ncol = 4, nrow = 0))
names(resultados) <- c("Baseline", "F1", "Precisão", "Revocação")

addRowAdpater <- function(resultados, baseline, matriz, ...) {
  print(baseline)
  newRes <- data.frame(baseline, matriz$byClass["F1"] * 100, matriz$byClass["Precision"] * 100, matriz$byClass["Recall"] * 100)
  rownames(newRes) <- baseline
  names(newRes) <- c("Baseline", "F1", "Precision", "Recall")
  newdf <- rbind(resultados, newRes)
  return (newdf)
}

set.seed(10)

for (year in 1:10) {
  
  library(reticulate)
  k_bert = import('keras_bert')
  token_dict = k_bert$load_vocabulary(vocab_path)
  tokenizer = k_bert$Tokenizer(token_dict)

  model = k_bert$load_trained_model_from_checkpoint(
  config_path,
  checkpoint_path,
  training=T,
  trainable=T,
  seq_len=seq_length)


  trainIndex <- createDataPartition(df$resposta, p=0.8, list=FALSE)
  train <- df[ trainIndex,]
  test <- df[-trainIndex,]

  tokenize_fun = function(dataset) {
    c(indices, target, segments) %<-% list(list(),list(),list())
    for ( i in 1:nrow(dataset)) {
      c(indices_tok, segments_tok) %<-% tokenizer$encode(dataset[[DATA_COLUMN]][i], max_len=seq_length)
      indices = indices %>% append(list(as.matrix(indices_tok)))
      target = target %>% append(dataset[[LABEL_COLUMN]][i])
      segments = segments %>% append(list(as.matrix(segments_tok)))
    }
    return(list(indices,segments, target))
  }

  dt_data = function(data){
    c(x_train, x_segment, y_train) %<-% tokenize_fun(data)
    return(list(x_train, x_segment, y_train))
  }

  c(x_train,x_segment, y_train) %<-% dt_data(train)

  train = do.call(cbind,x_train) %>% t()
  segments = do.call(cbind,x_segment) %>% t()
  targets = do.call(cbind,y_train) %>% t()
  concat = c(list(train ),list(segments))

  c(x_test,x_segment_test, y_test) %<-% dt_data(test)

  test_validate = do.call(cbind, x_test) %>% t()
  segments_test = do.call(cbind, x_segment_test) %>% t()
  targets_test = do.call(cbind, y_test) %>% t()

  concat_test = c(list(test_validate ),list(segments_test))

  c(decay_steps, warmup_steps) %<-% k_bert$calc_train_steps(
    targets %>% length(),
    batch_size=bch_size,
    epochs=epochs
  )

  library(keras)

  input_1 = get_layer(model,name = 'Input-Token')$input
  input_2 = get_layer(model,name = 'Input-Segment')$input
  inputs = list(input_1,input_2)

  dense = get_layer(model,name = 'NSP-Dense')$output

  outputs = dense %>% layer_dense(units=1L, activation='sigmoid',
                                  kernel_initializer=initializer_truncated_normal(stddev = 0.02),
                                  name = 'output')

  model = keras_model(inputs = inputs,outputs = outputs)

  model %>% compile(
    k_bert$AdamWarmup(decay_steps=decay_steps, 
                      warmup_steps=warmup_steps, lr=learning_rate),
    loss = 'binary_crossentropy',
    metrics = 'accuracy'
  )

  history <- model %>% fit(
    concat,
    targets,
    epochs=epochs,
    batch_size=bch_size, validation_split=0.2)

  # history
  predictions <- model %>% predict(concat_test)
  predictions2 <- round(predictions, 0)

  matriz <- confusionMatrix(data = as.factor(predictions2), as.factor(test$resposta), positive="1")
  resultados <- addRowAdpater(resultados, "MARCOS", matriz)
  View(resultados)
  #print(paste("F1 ", matriz$byClass["F1"] * 100, "Precisao ", matriz$byClass["Precision"] * 100, "Recall ", matriz$byClass["Recall"] * 100, "Acuracia ", matriz$overall["Accuracy"] * 100))
}
