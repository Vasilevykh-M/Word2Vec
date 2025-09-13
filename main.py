from model.Model import Model

import nltk
import argparse

def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}' if resource == 'stopwords' else f'taggers/{resource}')
        except LookupError:
            nltk.download(resource)

def get_args():
    parser = argparse.ArgumentParser(description='Векторизация слов при помощи Word2Vec')

    parser.add_argument('--type_model', type=str, choices=['CBOW', 'SG'], default='CBOW',
                        help='Метод векторизации: CBOW (Continuous Bag of Words) или SG (Skip-Gram)')
    parser.add_argument('--input', type=str, required=True,
                        help='Путь к входному TXT файлу с текстами')

    parser.add_argument('--language', type=str, default='russian',
                        help='Язык текста (по умолчанию: russian)')
    parser.add_argument('--use_stopwords', action='store_true',
                        help='Использовать стоп-слова')
    parser.add_argument('--use_stemming', action='store_true',
                        help='Использовать стемминг')
    parser.add_argument('--window_size', type=int, default=2,
                        help='Размер окна контекста')

    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Размерность вектора представления слова')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Число экземпляров в батче при обучении')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Число эпох при обучении модели')
    
    parser.add_argument('--clustering', action='store_true',
                        help='Кластеризация текста')
    parser.add_argument('--draw', action='store_true',
                        help='Визуализировать кластеризацию')
    parser.add_argument('--output', type=str, required=True,
                        help='Путь с результатами кластеризации')

    args = parser.parse_args()

    return args

def main():
    download_nltk_resources()
    args = get_args()
    preproc_cfg = {
        "language": args.language,
        "use_stemming": args.use_stopwords,
        "use_stopwords": args.use_stemming,
        "window_size": args.window_size
    }
    
    model_cfg = {
        "type_model": args.type_model,
        "embedding_dim": args.embedding_dim,
        "batch_size": args.batch_size,
        "epochs": args.epochs
    }

    model = Model(preproc_cfg, model_cfg, args.input)

    if args.clustering:
        model.clustering()
        if args.draw:
            model.draw(args.output)

if __name__ == "__main__":
    main()