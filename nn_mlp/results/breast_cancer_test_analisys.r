library(ggplot2)

# Equilibrados
breast_cancer_results$hidden_layer_sizes <- factor(breast_cancer_results$hidden_layer_sizes)
summary(breast_cancer_results$hidden_layer_sizes)
qplot(data=breast_cancer_results, x=hidden_layer_sizes)

breast_cancer_results$activation <- factor(breast_cancer_results$activation)
qplot(data=breast_cancer_results, x=activation)

breast_cancer_results$solver <- factor(breast_cancer_results$solver)
qplot(data=breast_cancer_results, x=solver)

summary(breast_cancer_results$alpha)
qplot(data=breast_cancer_results, x=alpha)

breast_cancer_results$learning_rate <- factor(breast_cancer_results$learning_rate)
summary(breast_cancer_results$learning_rate)
qplot(data=breast_cancer_results, x=learning_rate)

summary(breast_cancer_results$learning_rate_init)
qplot(data=breast_cancer_results, x=learning_rate_init)

summary(breast_cancer_results$power_t)
qplot(data=breast_cancer_results, x=power_t)

summary(breast_cancer_results$max_iter)
ggplot(breast_cancer_results, aes(x=max_iter)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=30)

summary(breast_cancer_results$momentum)
ggplot(breast_cancer_results, aes(x=momentum)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=0.01)

summary(breast_cancer_results$beta_1)
ggplot(breast_cancer_results, aes(x=beta_1)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=0.01)

summary(breast_cancer_results$beta_2)
ggplot(breast_cancer_results, aes(x=beta_2)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=0.01)

summary(breast_cancer_results$epsilon)
ggplot(breast_cancer_results, aes(x=epsilon)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=0.00000000001)

summary(breast_cancer_results$n_iter_no_change)
ggplot(breast_cancer_results, aes(x=n_iter_no_change)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=1)

summary(breast_cancer_results$mean_accuracy)
ggplot(breast_cancer_results, aes(x=mean_accuracy)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=0.01) +
  facet_wrap(~max_iter)

ggplot(breast_cancer_results, aes(x=factor(learning_rate_init), y=mean_accuracy)) +
  geom_boxplot() +
  facet_wrap(~max_iter)

ggplot(breast_cancer_results, aes(x=factor(hidden_layer_sizes), y=mean_accuracy)) +
  geom_boxplot()
ggplot(breast_cancer_results, aes(x=factor(alpha), y=mean_accuracy)) +
  geom_boxplot()
ggplot(breast_cancer_results, aes(x=factor(learning_rate_init), y=mean_accuracy)) +
  geom_boxplot()
ggplot(breast_cancer_results, aes(x=factor(power_t), y=mean_accuracy)) +
  geom_boxplot()
ggplot(breast_cancer_results, aes(x=factor(max_iter), y=mean_accuracy)) +
  geom_boxplot()
ggplot(breast_cancer_results, aes(x=factor(momentum), y=mean_accuracy)) +
  geom_boxplot()
ggplot(breast_cancer_results, aes(x=factor(beta_1), y=mean_accuracy)) +
  geom_boxplot()
ggplot(breast_cancer_results, aes(x=factor(beta_2), y=mean_accuracy)) +
  geom_boxplot()
ggplot(breast_cancer_results, aes(x=factor(epsilon), y=mean_accuracy)) +
  geom_boxplot()
ggplot(breast_cancer_results, aes(x=factor(n_iter_no_change), y=mean_accuracy)) +
  geom_boxplot()
