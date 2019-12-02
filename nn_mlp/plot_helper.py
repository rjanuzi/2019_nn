import matplotlib.pyplot as plt

def plot_scores_line(params, scores, title, xlabel, ylabel='Accuracy (%)', file_name='plot.png'):
    fig, ax = plt.subplots()

    ax.plot(params, scores)

    ax.set( xlabel=xlabel,
            ylabel=ylabel,
            title=title)
    ax.grid()
    # plt.ylim(bottom=0, top=1)
    ax.axhline(y=0, color='k')

    fig.savefig(file_name)
    plt.show()

def plot_scores_bar(params, scores, title, xlabel, ylabel='Accuracy (%)', file_name='plot.png'):
    plt.rcdefaults()
    fig, ax = plt.subplots()

    ax.bar(range(len(params)), scores, align='center')
    ax.set(xlabel=xlabel,
            ylabel=ylabel,
            title=title)
    ax.grid()

    plt.xticks(range(len(params)), params)
    plt.ylim(bottom=0, top=1)

    fig.savefig(file_name)
    plt.show()
