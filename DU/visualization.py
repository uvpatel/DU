import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column):
    sns.histplot(df[column])
    plt.show()

def correlation_heatmap(df):
    sns.heatmap(df.corr(), annot=True)
    plt.show()