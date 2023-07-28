from utils import read_dataset, save_dataset
from fastbm25 import fastbm25
from tqdm import tqdm
import nltk
import argparse

def tokenlizer(text: str, stopwords: list):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words_wo_stopwords = []
    for word in words:
        if word not in stopwords:
            words_wo_stopwords.append(word)
    return words_wo_stopwords

def read_stopwords():
    path = 'datasets/stopwords.txt'
    stopwords = set()
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.add(line.lower().strip())
    return list(stopwords)

def construct_models(dataset, stopwords):
    corpus = []
    for sample in tqdm(dataset):
        if 'question' in sample:
            query = sample['question']
        else:
            query = sample['title']
        corpus.append(tokenlizer(query, stopwords))
    model = fastbm25(corpus)
    return model

def retrieval(dataset, stopwords, model, corpus):
    retrieval_dict = {}
    for sample in tqdm(dataset):
        if 'question' in sample:
            query = sample['question']
        else:
            query = sample['title']
        result = model.top_k_sentence(tokenlizer(query, stopwords), k=20)
        result_tmp = []
        for i, item in enumerate(result):
            sample_case = corpus[item[1]]
            result_tmp.append({
                'id': sample_case['id'],
                'score': item[2]
            })
        retrieval_dict[sample['id']] = result_tmp
        # sample["retrieval"] = result_tmp
    return retrieval_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
                        default='msqa',
                        type=str)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    stopwords = read_stopwords()
    dataset_train = read_dataset(f'datasets/{dataset_name}/initial/train.json')
    dataset_valid = read_dataset(f'datasets/{dataset_name}/initial/valid.json')
    dataset_test = read_dataset(f'datasets/{dataset_name}/initial/test.json')
    model = construct_models(dataset_train, stopwords)
    retrieval_dict = retrieval(dataset_valid, stopwords, model, dataset_train)
    save_dataset(f'datasets/{dataset_name}/bm25/', 'valid.json', retrieval_dict)

    retrieval_dict = retrieval(dataset_test, stopwords, model, dataset_train)
    save_dataset(f'datasets/{dataset_name}/bm25/', 'test.json', retrieval_dict)

