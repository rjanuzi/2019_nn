# Helper functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

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

X_train, X_test, y_train, y_test = train_test_split(*get_data(), test_size=0.2, random_state=0)

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

# Breast Cancer best model
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

print('Train: %d samples\nTest: %d samples.' % (len(X_train), len(X_test)))

with open('metro_test_r2s.csv', 'w') as f:
    f.write('run;r2\n')

for ite in range(100):
    # Train model
    estimator.fit(X_train, y_train)

    # Predict
    predicted_ys = estimator.predict(X_test)

    print('Runned %d iterations (%.2f%%)' % (ite, (ite)))

    with open('metro_test_r2s.csv', 'a') as f:
        f.write('%d;%.2f\n' % (ite, r2_score(y_test, predicted_ys)))
