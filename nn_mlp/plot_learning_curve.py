import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

# MLP
# from sklearn.neural_network import MLPClassifier as mlp
from sklearn.neural_network import MLPRegressor as mlp

# SCORING = 'accuracy' # Classification
# SCORING = 'neg_mean_squared_error' # Regression
# SCORING = 'max_error' # Regression
SCORING = 'r2' # Regression

# Datasets
# from autism_data import get_data
# from breast_cancer_data import get_data
from metro_traffic_data import get_data

def plot_learning_curve(estimator, title, X, y, axes, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    return plt

fig, axes = plt.subplots()

# Looking for only the train data
X, _, y, _ = train_test_split(*get_data(), test_size=0.2, random_state=0)

title = "Learning Curves (MLP - Best)"

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

# # Autism best model
# estimator = mlp(hidden_layer_sizes=(100, 100),
#                     activation='logistic',
#                     solver='adam',
#                     alpha=0,
#                     learning_rate='adaptive',
#                     learning_rate_init=1e-4,
#                     power_t=0.6,
#                     max_iter=3000,
#                     momentum=0.6,
#                     beta_1=0.9,
#                     beta_2=0.5,
#                     epsilon=1e-11,
#                     n_iter_no_change=14)

# # Breast Cancer best model
# estimator = mlp(hidden_layer_sizes=(100, 100),
#                     activation='logistic',
#                     solver='adam',
#                     alpha=1e-6,
#                     learning_rate='adaptive',
#                     learning_rate_init=1e-4,
#                     power_t=0.6,
#                     max_iter=3000,
#                     momentum=0.7,
#                     beta_1=0.8,
#                     beta_2=0.5,
#                     epsilon=1e-12,
#                     n_iter_no_change=12)

# Metro best model
estimator = mlp(hidden_layer_sizes=(100, 100),
                    activation='logistic',
                    solver='adam',
                    alpha=1e-7,
                    learning_rate='adaptive',
                    learning_rate_init=1e-4,
                    power_t=0.6,
                    max_iter=10000,
                    momentum=0.6,
                    beta_1=0.8,
                    beta_2=0.6,
                    epsilon=1e-12,
                    n_iter_no_change=10)

plot_learning_curve(estimator, title, X, y, axes=axes, ylim=(0.5, 1.01), cv=cv, n_jobs=4)

plt.show()
