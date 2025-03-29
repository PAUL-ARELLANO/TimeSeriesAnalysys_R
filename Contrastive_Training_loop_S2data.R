# Load the reticulate package for Python integration
reticulate::use_condaenv("SimCLR", required = TRUE)
reticulate::py_config()

library(dplyr)
library(tidyr)
library(recipes)
library(zoo)
library(keras)
library(tensorflow)
library(purrr)

# Load the dataset
load("C:/Users/pa589/NAU/TREE_STRESS/TreeStress_detection/TimeSeriesAnalysys_R/NDVI_TimeSeries_Junipers.RData")
NDVI_Junipers <- TILE95_clean
NDVI_Junipers <- NDVI_Junipers %>%
  select(-Tile) %>%
  mutate(Date = as.Date(Date))

# Normalize data
normalize_data <- function(df) {
  recipe(NDVI ~ ., data = df) %>%
    step_normalize(all_numeric_predictors()) %>%
    prep() %>%
    bake(new_data = NULL)
}

NDVI_Junipers_normalized <- normalize_data(NDVI_Junipers)

# Define parameters
window_size <- 4
embedding_dim <- 128
batch_size <- 32
projection_dim <- 64
learning_rate <- 0.001
num_epochs <- 50

# Adjusted sequence generation function
generate_sequences <- function(df, window_size) {
  df %>%
    group_by(Status) %>%  # Group by Status
    arrange(Date) %>%  # Ensure data is ordered by Date
    mutate(seq_id = row_number()) %>%  # Create sequence IDs
    filter(seq_id >= window_size) %>%  # Ensure sequences are complete
    mutate(
      NDVI_seq = purrr::map(seq_id, function(i) {
        # Generate a window of NDVI values from `i - window_size + 1` to `i`
        df$NDVI[(i - window_size + 1):i]
      })
    ) %>%
    ungroup()  # Ungroup after mutation
}

# Generate sequences
NDVI_sequences <- generate_sequences(NDVI_Junipers_normalized, window_size)

# Convert NDVI_seq (list of vectors) into a matrix for training
NDVI_sequences_matrix <- do.call(rbind, NDVI_sequences$NDVI_seq)

# Flatten the 3rd dimension (if needed)
NDVI_sequences_matrix <- array_reshape(NDVI_sequences_matrix, c(nrow(NDVI_sequences_matrix), 4))

# Print new shape of NDVI_sequences_matrix
cat("New Shape of NDVI_sequences_matrix:", dim(NDVI_sequences_matrix), "\n")

# Reshape the labels into a 2D tensor of shape (1895, 1)
labels <- as.integer(as.factor(NDVI_sequences$Status)) - 1
labels_tensor <- tf$constant(labels, dtype = tf$int32)

# Print the shape of labels_tensor
cat("Shape of labels_tensor (TensorFlow):", as.array(tf$shape(labels_tensor)), "\n")

# Ensure the sequences are converted into a 3D array
NDVI_sequences_matrix <- NDVI_sequences %>%
  ungroup() %>%
  select(NDVI_seq) %>%
  mutate(across(everything(), as.numeric)) %>%
  as.matrix()

# Ensure correct shape (num_samples, window_size, num_features)
num_samples <- nrow(NDVI_sequences) %/% window_size  # Integer division
if (nrow(NDVI_sequences) %% window_size != 0) {
  stop("Error: Data cannot be evenly divided by window size.")
}

# Convert to 3D array
NDVI_sequences_matrix <- array(NDVI_sequences_matrix, dim = c(num_samples, window_size, 1))

# Define the model
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(window_size, 1)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")  # Example for binary classification

# Compile the model
model %>% compile(
  loss = "binary_crossentropy",  # Example for binary classification
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)

# Fit the model to the data (binary classification)
model %>% fit(NDVI_sequences_matrix, labels_tensor, epochs = num_epochs, batch_size = batch_size)

# Check the shape of the tensor to ensure it's as expected
cat("Shape of NDVI_sequences_matrix (TensorFlow):", as.array(tf$shape(NDVI_sequences_matrix_tf)), "\n")
cat("Shape of labels (TensorFlow):", as.array(tf$shape(labels_tf)), "\n")

# Ensure the labels are integers
labels <- as.integer(as.factor(NDVI_sequences$Status)) - 1
labels_tensor <- tf$constant(labels, dtype = tf$int32)

batch <- NDVI_sequences_matrix[batch_indices, , , drop = FALSE]
batch <- tf$convert_to_tensor(batch, dtype = "float32")
batch_labels <- labels[batch_indices]
batch_labels <- tf$convert_to_tensor(batch_labels, dtype = "int32")

cat("Input Shape:", dim(NDVI_sequences_matrix), "\n")
cat("Labels Shape:", dim(labels), "\n")

# Model architecture (same as in your original script)
# Define encoder and projection head
define_encoder <- function(input_shape, embedding_dim) {
  input <- layer_input(shape = input_shape)
  output <- input %>%
    layer_conv_1d(filters = 64, kernel_size = 3, activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_flatten() %>%
    layer_dense(units = embedding_dim, activation = "relu")
  keras_model(inputs = input, outputs = output)
}

define_projection_head <- function(embedding_dim, projection_dim) {
  input <- layer_input(shape = embedding_dim)
  output <- input %>%
    layer_dense(units = projection_dim, activation = "relu") %>%
    layer_dense(units = projection_dim, activation = NULL)
  keras_model(inputs = input, outputs = output)
}

projection_head <- define_projection_head(embedding_dim, projection_dim)
projection_head %>% summary()

encoder <- define_encoder(c(window_size, 1), embedding_dim)
encoder %>% summary()

# Contrastive loss function
contrastive_loss <- function(y_true, y_pred, temperature = 0.1) {
  y_pred <- tf$nn$l2_normalize(y_pred, axis = 1L)
  similarity_matrix <- tf$matmul(y_pred, tf$transpose(y_pred)) / temperature
  labels <- tf$eye(tf$shape(y_pred)[1])
  loss <- tf$nn$sparse_softmax_cross_entropy_with_logits(labels = labels, logits = similarity_matrix)
  tf$reduce_mean(loss)
}

# Training function
train_contrastive_model <- function(encoder, projection_head, NDVI_sequences_matrix, num_epochs, batch_size) {
  optimizer <- optimizer_adam(learning_rate = learning_rate)
  
  for (epoch in 1:num_epochs) {
    batch_indices <- sample(1:dim(NDVI_sequences_matrix)[1], batch_size, replace = TRUE)
    batch <- NDVI_sequences_matrix[batch_indices, , , drop = FALSE]
    batch_labels <- labels[batch_indices]
    
    batch <- tf$convert_to_tensor(batch, dtype = "float32")
    batch_labels <- tf$convert_to_tensor(batch_labels, dtype = "int32")
    
    with(tf$GradientTape() %as% tape, {
      embeddings <- encoder(batch)
      projections <- projection_head(embeddings)
      loss_value <- contrastive_loss(batch_labels, projections)
    })
    
    gradients <- tape$gradient(loss_value, c(encoder$trainable_variables, projection_head$trainable_variables))
    optimizer$apply_gradients(purrr::transpose(list(gradients, c(encoder$trainable_variables, projection_head$trainable_variables))))
    cat("Epoch:", epoch, "Loss:", loss_value$numpy(), "\n")
  }
}

train_contrastive_model(encoder, projection_head, NDVI_sequences_matrix, num_epochs, batch_size)
