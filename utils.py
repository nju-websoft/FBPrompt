#coding=utf-8
import re
import json
import os

def format_answers(dataset, keyphrase=False):
    dataset_new = []
    for sample in dataset:
        if keyphrase==False:
            answers = sample['answers']
            answers_new = []
            for item in answers:
                if isinstance(item, list):
                    answers_new.append(item[0])
                else:
                    answers_new.append(item)
            sample['answers'] = answers_new
        else:
            trg_data = sample['answers_ori']
            src_data = sample['context_ori']
            src_data = src_data.strip()
            trg_data = trg_data.strip()
            sp = src_data.split('<eos>')
            if len(sp) == 2:
                title = sp[0].strip()
                context = sp[1].strip()
            else:
                title = ""
                context = src_data.strip()
            answers = trg_data.split(';')
            if '<peos>' in answers:
                answers.remove('<peos>')
            answers = [ans for ans in answers]
            sample['title'] = title
            sample['context'] = context
            sample['answers'] = answers
        
        dataset_new.append(sample)
        
        
    return dataset_new

def read_dataset_w_results(path, path_results, path_eval=None):
    if 'drop' in path:
        prediction = {}
        reference = []
        dataset = read_dataset(path)
        if os.path.exists(path_results):
            dataset_pred = read_dataset(path_results)
            dataset_pred_dict = {}
            for sample in dataset_pred:
                dataset_pred_dict[sample['id']] = sample
            dataset_new = []
            for sample in dataset:
                if sample['id'] in dataset_pred_dict.keys() and sample['id'] not in prediction.keys():
                    dataset_new.append(dataset_pred_dict[sample['id']])
                    if 'gpt_pred' in dataset_pred_dict[sample['id']]:
                        prediction[sample['id']] = dataset_pred_dict[sample['id']]['gpt_pred']
                        reference.append(dataset_pred_dict[sample['id']])
                else:
                    dataset_new.append(sample)
            dataset = dataset_new
        print('prediction:', len(prediction))
        assert len(prediction) == len(reference)
        return dataset, reference, prediction
    else:
        reference = {}
        if os.path.exists(path_results):
            dataset = read_dataset(path_results)
            prediction = read_dataset(path_eval)
            for sample in dataset:
                if 'gpt_pred' in sample:
                    reference[sample['id']] = sample['answers']
        else:
            prediction = {}
            dataset = read_dataset(path)
        assert len(reference) == len(prediction)
        print('reference:', len(reference), 'prediction:', len(prediction))
        return dataset, reference, prediction

def read_dataset_w_results_keyphrase(path, result_path):
    src_file, trg_file, pred_file = [], [], []
    if os.path.exists(result_path):
        dataset = read_dataset(result_path)
        for sample in dataset:
            if 'gpt_pred' in sample:
                src_file.append(sample['context_ori'])
                trg_file.append(sample['answers_ori'])
                pred_file.append(sample['gpt_pred'])
    else:
        dataset = read_dataset(path)
    return dataset, src_file, trg_file, pred_file

def read_dataset(path: str):
    if path.endswith('jsonl'):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(json.loads(line))
    else:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    return dataset

def save_dataset(fold_path, file_name, prediction):
    os.makedirs(fold_path, exist_ok=True)
    with open(fold_path + '/' + file_name, 'w', encoding='utf-8') as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)

def remove_answer(string):
    pattern = re.compile(r'Answer\d*:\s*', re.IGNORECASE)
    text = pattern.sub('', string)
    return text.strip()

def parsing(text):
    pattern = r'(?:\d+\.\s*|\-\s*)(.*?)(?=\n(?:\d+\.\s*|\-\s*)|$|\d+\.)'
    matches = re.findall(pattern, text)
    matches = [remove_answer(item) for item in matches]
    if len(matches) > 0:
        return matches
    else:
        return ['']