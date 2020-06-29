# from __future__ import print_function
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("res/featuresDF_4.csv")

df = data.iloc[:, 1:]
df['label'] = data.iloc[:, 0].apply(lambda i: "1" if i < 10000 else ("2" if i < 20000 else ("3" if i < 30000 else ("4" if i < 40000 else "5"))))


pca = PCA(n_components=50)
pca_result = pca.fit_transform(df.drop('label', axis=1).to_numpy())
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

df_temp = df
df_temp['pca-one'] = pca_result[:,0]
df_temp['pca-two'] = pca_result[:,1]
df_temp['pca-three'] = pca_result[:,2]


def scatter_plot(df, x, y, hue, output_dir):
    plt.figure(figsize=(25,25))
    sns_plot = sns.scatterplot(
        x=x, y=y, hue=hue,
        palette=sns.color_palette("hls", 5),
        data=df, legend="full", alpha=0.3
    )

    fig = sns_plot.get_figure()
    fig.savefig(output_dir)


# scatter_plot(df_temp, "pca-one", "pca-two", "label", "res/plots/pca-visualization.png")


# from sklearn.manifold import TSNE
from tsnecuda import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.drop('label', axis=1).to_numpy())

df_temp['tsne-2d-one'] = tsne_results[:,0]
df_temp['tsne-2d-two'] = tsne_results[:,1]

scatter_plot(df_temp, "tsne-2d-one", "tsne-2d-two", "label", "res/plots/tsne-visualization2.png")


tsne_results_after_pca = tsne.fit_transform(pca_result)

df_temp['tsne-2d-one-afterPCA'] = tsne_results[:,0]
df_temp['tsne-2d-two-afterPCA'] = tsne_results[:,1]

scatter_plot(df_temp, "tsne-2d-one-afterPCA", "tsne-2d-two-afterPCA", "label", "res/plots/pca-tsne-visualization2.png")
