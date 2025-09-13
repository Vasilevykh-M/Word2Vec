import numpy as np
from sklearn.cluster import KMeans


def cluster_words(model, word_to_ix, n_clusters=10, top_n_words=5):
    words = list(word_to_ix.keys())
    word_indices = list(word_to_ix.values())

    # Извлекаем эмбеддинги из модели
    embeddings = model.embeddings.weight.data.cpu().numpy()
    word_embeddings = embeddings[word_indices]

    # Выполняем K-Means кластеризацию
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(word_embeddings)
    centroids = kmeans.cluster_centers_

    # Для каждого центроида находим ближайшие слова
    centroid_words = {}
    for cluster_id in range(n_clusters):
        # Вычисляем расстояния от всех слов до центроида
        distances = np.linalg.norm(word_embeddings - centroids[cluster_id], axis=1)

        # Находим индексы ближайших слов
        closest_indices = np.argsort(distances)[:top_n_words]

        # Получаем слова
        closest_words = [words[i] for i in closest_indices]
        centroid_words[cluster_id] = closest_words

    # Создаем словарь с результатами
    results = {
        'words': words,
        'embeddings': word_embeddings,
        'cluster_labels': cluster_labels,
        'centroids': centroids,
        'centroid_words': centroid_words,
        'kmeans': kmeans
    }

    return results