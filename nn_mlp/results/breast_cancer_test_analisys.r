library(ggplot2)
library(readxl)
library(readr)

breast_train <- read_excel("breast_cancer_train.xlsx")

# Convert to factors
breast_train$hidden_layer_sizes <- factor(breast_train$hidden_layer_sizes)
breast_train$alpha <- factor(breast_train$alpha)
breast_train$learning_rate_init <- factor(breast_train$learning_rate_init)
breast_train$power_t <- factor(breast_train$power_t)

# Train accuracy distribution
ggplot(breast_train, aes(x=mean_accuracy)) +
  geom_histogram(binwidth=0.01, color="black", fill="light blue") +
  scale_x_continuous(limits=c(0.4,1.1), breaks=seq(0, 1, 0.1)) +
  scale_y_continuous(breaks=seq(0, 8500, 500))
ggsave("breast_train_acc_distribution.png")

# Train accuracy distribution > 70%
top <- breast_train[breast_train$mean_accuracy > 0.7, ]
nrow(top)
ggplot(top, aes(x=mean_accuracy)) +
  geom_histogram(binwidth=0.01, color="black", fill="light blue") +
  scale_x_continuous(limits=c(0.7,0.76), breaks=seq(0, 1, 0.01)) +
  scale_y_continuous(breaks=seq(0, 300, 10))
ggsave("breast_train_acc_distribution_top.png")

# Train accuracy distribution > 99%
top <- breast_train[breast_train$mean_accuracy > 0.74, ]
nrow(top)

# Test accuracies
breast_test <- read_delim("breast_test_accs.csv", delim=";", escape_double=FALSE, trim_ws=TRUE)
ggplot(breast_test, aes(y=acc)) +
  geom_boxplot() +
  scale_y_continuous(limits=c(0.5,0.8), breaks=seq(0, 1, 0.05))
ggsave("breast_test_accs.png")
