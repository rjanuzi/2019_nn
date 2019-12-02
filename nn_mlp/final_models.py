# Helper functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# MLP
from sklearn.neural_network import MLPClassifier as mlp
# from sklearn.neural_network import MLPRegressor as mlp

SCORING = 'accuracy' # Classification
# SCORING = 'neg_mean_squared_error' # Regression
# SCORING = 'max_error' # Regression
# SCORING = 'r2' # Regression

# Datasets
from breast_cancer_data import get_data
# from autism_data import get_data
# from metro_traffic_data import get_data

X_train, X_test, y_train, y_test = train_test_split(*get_data(), test_size=0.2, random_state=0)
clf = mlp(hidden_layer_sizes=(100, 100),
                    activation='logistic',
                    solver='adam',
                    alpha=1e-7,
                    learning_rate='adaptive',
                    learning_rate_init=1e-4,
                    power_t=0.6,
                    max_iter=10000,
                    momentum=0.6,
                    beta_1=0.8,
                    beta_2=0.4,
                    epsilon=1e-11,
                    n_iter_no_change=14)

# scores = cross_val_score(clf, X_train, y_train, scoring=SCORING, cv=5, n_jobs=4)
# print(scores.mean())

# Run a partial fit to generate initial coefs and untrained predict
clf.partial_fit(X_test, y_test, np.unique(y_test))
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f %%' % (accuracy*100.0))

# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.imshow(X=conf_matrix, cmap='gray')
# plt.show()

# Print MLP Weigths
fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = clf.coefs_[0].min(), clf.coefs_[0].max()
for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(3, 3), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()

# print(clf.coefs_[0].shape)
# print(clf.coefs_[0][0].shape)
# print(clf.coefs_[0][1].shape)
# print(clf.coefs_[0][1].reshape(10,10))
# plt.imshow(X=clf.coefs_[0][0].reshape(10,10))
# plt.show()

# Fit clf
clf.fit(X_train, y_train)
# plt.imshow(X=clf.coefs_[0][0].reshape(10,10))
# plt.show()
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy: %.2f %%' % (accuracy*100.0))

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = clf.coefs_[0].min(), clf.coefs_[0].max()
for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(3, 3), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()

# # Plot confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.imshow(X=conf_matrix, cmap='gray')
# plt.show()

# Plot MLP params
