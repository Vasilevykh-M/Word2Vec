import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from adjustText import adjust_text


def visualize_clustering(hdbscan_results, max_words=200, figsize=(14, 10), output_file='hdbscan_visualization.png'):

    if not hdbscan_results:
        print("Нет данных для визуализации")
        return

    if len(hdbscan_results) > max_words:
        indices = np.random.choice(len(hdbscan_results), max_words, replace=False)
        words = [hdbscan_results[i]["word"] for i in indices]
        centroid = [hdbscan_results[i]["centroid"] for i in indices]
    else:
        words = [hdbscan_results[i]["word"] for i in range(len(hdbscan_results))]
        centroid = [hdbscan_results[i]["centroid"] for i in range(len(hdbscan_results))]

    centroids_array = np.array(centroid)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(centroid) - 1))
    embeddings_2d = tsne.fit_transform(centroids_array)

    plt.figure(figsize=figsize)

    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        alpha=0.7,
        color='blue'
    )

    texts = []
    for i, word in enumerate(words):
        texts.append(plt.text(
            embeddings_2d[i, 0],
            embeddings_2d[i, 1],
            word,
            fontsize=9,
            color='black'
        ))

    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title('Визуализация HDBSCAN кластеризации слов')
    plt.xlabel('t-SNE компонента 1')
    plt.ylabel('t-SNE компонента 2')
    plt.grid(True, alpha=0.3)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
