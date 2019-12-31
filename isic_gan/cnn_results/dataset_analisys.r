library(readr)
library(ggplot2)

dataset_index <- read_delim("index.csv", delim=";", escape_double=FALSE, trim_ws=TRUE)

# Convert to factors
dataset_index$type <- factor(dataset_index$type)
dataset_index$size_x <- factor(dataset_index$size_x)
dataset_index$size_y <- factor(dataset_index$size_y)
dataset_index$benign_malignant <- factor(dataset_index$benign_malignant)
dataset_index$diagnosis <- factor(dataset_index$diagnosis)
dataset_index$diagnosis_confirm_type <- factor(dataset_index$diagnosis_confirm_type)
dataset_index$melanocytic <- factor(dataset_index$melanocytic)
dataset_index$sex <- factor(dataset_index$sex)

# Plot diagnosis distribution
dataset_index <- dataset_index[dataset_index$benign_malignant=="benign" | dataset_index$benign_malignant=="malignant", ]
ggplot(dataset_index, aes(x=benign_malignant)) +
  geom_histogram(stat="count", color="black", fill="light blue") +
  scale_y_continuous(breaks=seq(0, 20000, 1000))
ggsave("dataset_benign_malignant_distribution.png")

summary(dataset_index$size_x)
summary(dataset_index$size_y)
