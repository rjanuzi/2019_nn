from random import shuffle
from time import time

# Helper functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from plot_helper import plot_scores_line, plot_scores_bar

# MLP
# from sklearn.neural_network import MLPClassifier as mlp
from sklearn.neural_network import MLPRegressor as mlp

# SCORING = 'accuracy' # Classification
SCORING = 'neg_mean_squared_error' # Regression
# SCORING = 'max_error' # Regression
# SCORING = 'r2' # Regression

# Datasets
# from breast_cancer_data import get_data
# from autism_data import get_data
from metro_traffic_data import get_data

# Plots
# hidden_layer_sizes_opts = [2, 4, 6, 10, 14, 20, 30, 40, 50, 100, 150, 200, 300, 500]
# activation_opts = ['identity', 'logistic', 'tanh', 'relu']
# solver_opts = ['lbfgs', 'sgd', 'adam']
# alpha_opts = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# learning_rate_opts = ['constant', 'invscaling', 'adaptive']
# learning_rate_init_opts = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# power_t_opts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# max_iter_opts = [10, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 20000]
# momentum_opts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# beta_1_opts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# beta_2_opts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
# epsilon_opts = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
# n_iter_no_change_opts = [2, 4, 6, 8, 10, 12, 14, 15, 16]

# Auto test
hidden_layer_sizes_opts = [2, 10, 50, 200]
activation_opts = ['logistic']
solver_opts = ['adam']
alpha_opts = [0, 1e-7, 1e-6]
learning_rate_opts = ['adaptive']
learning_rate_init_opts = [1e-6, 1e-5, 1e-4]
power_t_opts = [0.6, 0.7, 0.8]
max_iter_opts = [500, 3000, 10000]
momentum_opts = [0.5, 0.6, 0.7]
beta_1_opts = [0.8, 0.9]
beta_2_opts = [0.4, 0.5, 0.6, 0.7]
epsilon_opts = [1e-12, 1e-11, 1e-10]
n_iter_no_change_opts = [8, 10, 12, 14]

def copy_dict_list(list_to_copy):
    cpy = []
    for l in list_to_copy:
        cpy.append(l.copy())
    return cpy

def add_params(params, new_params, new_params_name):
    result_params = []
    for n in new_params:
        new_temp_part = copy_dict_list(params)
        for t in new_temp_part:
            t[new_params_name] = n
        result_params += new_temp_part

    return result_params

def gen_params_combination():
    params = []
    for p in hidden_layer_sizes_opts:
        params.append({'hidden_layer_sizes': (int(p/2), int(p/2))})

    params = add_params(params, activation_opts, 'activation')
    params = add_params(params, solver_opts, 'solver')
    params = add_params(params, alpha_opts, 'alpha')
    params = add_params(params, learning_rate_opts, 'learning_rate')
    params = add_params(params, learning_rate_init_opts, 'learning_rate_init')
    params = add_params(params, power_t_opts, 'power_t')
    params = add_params(params, max_iter_opts, 'max_iter')
    params = add_params(params, momentum_opts, 'momentum')
    params = add_params(params, beta_1_opts, 'beta_1')
    params = add_params(params, beta_2_opts, 'beta_2')
    params = add_params(params, epsilon_opts, 'epsilon')
    params = add_params(params, n_iter_no_change_opts, 'n_iter_no_change')

    return params

# PARAMS TESTS
# =======================================================================================
def test_hidden_layer_sizes(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for hls in hidden_layer_sizes_opts:
        classifier = mlp(hidden_layer_sizes=(int(hls/2), int(hls/2, )))
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_line(params=hidden_layer_sizes_opts,
                    scores=score_means,
                    title='hidden_layer_sizes vs %s' % SCORING,
                    xlabel='Hidden layer size (neurons count)',
                    file_name=plot_file_name)

def test_activation(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for act in activation_opts:
        classifier = mlp(activation=act)
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_bar(params=activation_opts,
                    scores=score_means,
                    title='activation vs %s' % SCORING,
                    xlabel='Activation function',
                    file_name=plot_file_name)

def test_solver(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for s in solver_opts:
        classifier = mlp(solver=s)
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_bar(params=solver_opts,
                    scores=score_means,
                    title='solver vs %s' % SCORING,
                    xlabel='Solver Method',
                    file_name=plot_file_name)

def test_alpha(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for alpha in alpha_opts:
        classifier = mlp(alpha=alpha)
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_line(params=alpha_opts,
                    scores=score_means,
                    title='alpha_opts vs %s' % SCORING,
                    xlabel='Alpha',
                    file_name=plot_file_name)

def test_learning_rate(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for l in learning_rate_opts:
        classifier = mlp(learning_rate=l, solver='sgd') # Learning rate is used only with te SGD
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_bar(params=learning_rate_opts,
                    scores=score_means,
                    title='learning_rate vs %s' % SCORING,
                    xlabel='Learning Rate (with SGD)',
                    file_name=plot_file_name)

def test_learning_rate_init(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for l in learning_rate_init_opts:
        classifier = mlp(learning_rate_init=l, solver='adam') # Learning rate is used only with te SGD or ADAM
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_line(params=learning_rate_init_opts,
                    scores=score_means,
                    title='learning_rate_init vs %s' % SCORING,
                    xlabel='Learning Rate Init (with ADAM)',
                    file_name=plot_file_name)

def test_power_t(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for p in power_t_opts:
        classifier = mlp(power_t=p, solver='sgd') # Learning rate is used only with te SGD
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_line(params=power_t_opts,
                    scores=score_means,
                    title='learning_rate vs %s' % SCORING,
                    xlabel='Power T (with SGD)',
                    file_name=plot_file_name)

def test_max_iterations(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for max_iter in max_iter_opts:
        classifier = mlp(max_iter=max_iter)
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_line(params=max_iter_opts,
                    scores=score_means,
                    title='max_iter_opts vs %s' % SCORING,
                    xlabel='Max Iterations',
                    file_name=plot_file_name)

def test_momentum(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for m in momentum_opts:
        classifier = mlp(momentum=m, solver='sgd') # Learning rate is used only with te SGD
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_line(params=momentum_opts,
                    scores=score_means,
                    title='momentum vs %s' % SCORING,
                    xlabel='Momentum (with SGD)',
                    file_name=plot_file_name)

def test_beta_1(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for b in beta_1_opts:
        classifier = mlp(beta_1=b, solver='adam') # Learning rate is used only with the ADAM
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_line(params=beta_1_opts,
                    scores=score_means,
                    title='beta_1 vs %s' % SCORING,
                    xlabel='Beta 1 (with ADAM)',
                    file_name=plot_file_name)

def test_beta_2(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for b in beta_2_opts:
        classifier = mlp(beta_2=b, solver='adam') # Learning rate is used only with the ADAM
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_line(params=beta_2_opts,
                    scores=score_means,
                    title='beta_2 vs %s' % SCORING,
                    xlabel='Beta 2 (with ADAM)',
                    file_name=plot_file_name)

def test_epsilon(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for e in epsilon_opts:
        classifier = mlp(epsilon=e, solver='adam') # Learning rate is used only with the ADAM
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_line(params=epsilon_opts,
                    scores=score_means,
                    title='epsilon vs %s' % SCORING,
                    xlabel='Epsilon (with ADAM)',
                    file_name=plot_file_name)

def test_n_iter_no_change(features, targets, plot_file_name=None):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

    score_means = []
    for n in n_iter_no_change_opts:
        classifier = mlp(n_iter_no_change=n)
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=6)
        score_means.append(scores.mean())

    if plot_file_name:
        plot_scores_line(params=n_iter_no_change_opts,
                    scores=score_means,
                    title='n_iter_no_change_opts vs %s' % SCORING,
                    xlabel='N Iteration no Change',
                    file_name=plot_file_name)

# ========================================================================

def gen_all_plots(file_name_base):
    X_train, X_test, y_train, y_test = train_test_split(*get_data(), test_size=0.2, random_state=0)

    test_hidden_layer_sizes(X_train, y_train, r'figs\%s_hidden_layers_test.png' % file_name_base)
    # test_activation(X_train, y_train, r'figs\%s_activation_test.png' % file_name_base)
    # test_solver(X_train, y_train, r'figs\%s_solver_test.png' % file_name_base)
    # test_alpha(X_train, y_train, r'figs\%s_alpha_test.png' % file_name_base)
    # test_learning_rate(X_train, y_train, r'figs\%s_learning_rate_test.png' % file_name_base)
    # test_learning_rate_init(X_train, y_train, r'figs\%s_learning_rate_init_test.png' % file_name_base)
    # test_power_t(X_train, y_train, r'figs\%s_power_t_test.png' % file_name_base)
    # test_max_iterations(X_train, y_train, r'figs\%s_max_iterations_test.png' % file_name_base)
    # test_momentum(X_train, y_train, r'figs\%s_momentum_test.png' % file_name_base)
    # test_beta_1(X_train, y_train, r'figs\%s_beta_1_test.png' % file_name_base)
    # test_beta_2(X_train, y_train, r'figs\%s_beta_2_test.png' % file_name_base)
    # test_epsilon(X_train, y_train, r'figs\%s_epsilon_test.png' % file_name_base)
    # test_n_iter_no_change(X_train, y_train, r'figs\%s_n_iter_no_change_test.png' % file_name_base)

def auto_experiment(results_file):
    params = gen_params_combination()
    total_combinations = len(params)

    # Random param search
    shuffle(params)

    f = open(results_file, 'w')
    f.write('id;hidden_layer_sizes;activation;solver;alpha;learning_rate;learning_rate_init;power_t;max_iter;momentum;beta_1;beta_2;epsilon;n_iter_no_change;mean_%s\n' % SCORING)
    f.close()

    X_train, X_test, y_train, y_test = train_test_split(*get_data(), test_size=0.2, random_state=0)
    i = 0
    results_lines_buffer = ''
    for p in params:

        print('\n\n=============> Running for params: %s' % p)
        start_time = time()

        classifier = mlp(**p)
        scores = cross_val_score(classifier, X_train, y_train, scoring=SCORING, cv=5, n_jobs=4)

        i += 1
        results_lines_buffer += ('%d;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%.2f\n' % (i, p['hidden_layer_sizes'], p['activation'], p['solver'], p['alpha'], p['learning_rate'], p['learning_rate_init'], p['power_t'], p['max_iter'], p['momentum'], p['beta_1'], p['beta_2'], p['epsilon'], p['n_iter_no_change'], scores.mean()))

        if (i % 5) == 0:
            print('\n\n============> Progress: %d/%d (%.4f %%)\n\n' % (i, total_combinations, ((i/total_combinations) * 100)))
            f = open(results_file, 'a')
            f.write(results_lines_buffer)
            f.close()

            results_lines_buffer = ''

        print('\n\n================>Took %s seconds.' % (time()-start_time))

# gen_all_plots('metro')
auto_experiment(results_file='metro_test.csv')
