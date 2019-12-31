library(ggplot2)
library(readxl)
library(readr)

autism_train <- read_excel("autism_train.xlsx")

# Convert to factors
autism_train$hidden_layer_sizes <- factor(autism_train$hidden_layer_sizes)
autism_train$alpha <- factor(autism_train$alpha)
autism_train$learning_rate_init <- factor(autism_train$learning_rate_init)
autism_train$power_t <- factor(autism_train$power_t)

# Train accuracy distribution
ggplot(autism_train, aes(x=mean_accuracy)) +
  geom_histogram(binwidth=0.01, color="black", fill="light blue") +
  scale_x_continuous(limits=c(0.4,1.1), breaks=seq(0, 1, 0.05)) +
  scale_y_continuous(breaks=seq(0, 8500, 500))
ggsave("autism_train_acc_distribution.png")

# Train accuracy distribution > 95%
top <- autism_train[autism_train$mean_accuracy > 0.95, ]
nrow(top)
ggplot(top, aes(x=mean_accuracy)) +
  geom_histogram(binwidth=0.01, color="black", fill="light blue") +
  scale_x_continuous(limits=c(0.95,1.01), breaks=seq(0, 1, 0.01)) +
  scale_y_continuous(breaks=seq(0, 300, 10))
ggsave("autism_train_acc_distribution_top.png")

# Train accuracy distribution > 99%
top <- autism_train[autism_train$mean_accuracy > 0.99, ]
nrow(top)

# Test accuracies
autism_test <- read_delim("autism_test_accs.csv", delim=";", escape_double=FALSE, trim_ws=TRUE)
ggplot(autism_test, aes(y=acc)) +
  geom_boxplot() +
  scale_y_continuous(limits=c(0.9,1), breaks=seq(0, 1, 0.05))
ggsave("autism_test_accs.png")
