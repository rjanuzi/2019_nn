library(ggplot2)
library(readr)

metro_train <- read_delim("metro_train.csv", delim=";", escape_double=FALSE, trim_ws=TRUE)

# Convert to factors
metro_train$hidden_layer_sizes <- factor(metro_train$hidden_layer_sizes)
metro_train$alpha <- factor(metro_train$alpha)
metro_train$learning_rate_init <- factor(metro_train$learning_rate_init)
metro_train$power_t <- factor(metro_train$power_t)

# Train R2 distribution
ggplot(metro_train, aes(x=mean_r2)) +
  geom_histogram(binwidth=0.01, color="black", fill="light blue") +
  scale_x_continuous(limits=c(-3, 1.1), breaks=seq(-3, 1.1, 0.3)) +
  scale_y_continuous(breaks=seq(0, 30, 1))
ggsave("metro_train_acc_distribution.png")

# Train R2 distribution > 0.5
top <- metro_train[metro_train$mean_r2 > 0.5, ]
nrow(top)
ggplot(top, aes(x=mean_r2)) +
  geom_histogram(binwidth=0.01, color="black", fill="light blue") +
  scale_x_continuous(limits=c(0.5, 0.6), breaks=seq(-3, 1, 0.02)) +
  scale_y_continuous(breaks=seq(0, 30, 1))
ggsave("metro_train_acc_distribution_top.png")

# Test accuracies
autism_test <- read_delim("metro_test_r2s.csv", delim=";", escape_double=FALSE, trim_ws=TRUE)
ggplot(autism_test, aes(y=r2)) +
  geom_boxplot() +
  scale_y_continuous(limits=c(0.45,0.53), breaks=seq(0, 1, 0.01))
ggsave("metro_test_r2s.png")
