# coding=utf-8

from mindnlp.models import T5ForConditionalGeneration
from mindnlp.transforms import T5Tokenizer, BartTokenizer
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from tqdm import trange
import os
import random
from utils import save_dataset, read_dataset
import json
import argparse
from eval_scripts.eval_script_msqa import evaluate_msqa
import copy
import numpy as np
import ast

def save_model(output_model_file, model, model_name):
    os.makedirs(output_model_file, exist_ok=True)
    output_model_file += model_name
    mindspore.save_checkpoint(model, output_model_file)

def get_input_feature(features, max_length):
    input_list, answer_list = [], []
    for sample in features:
        context = sample['context']
        question = sample['question']
        answers = sample['answers']
        if isinstance(answers[0], str) is False:
            answers = [ans[0] for ans in answers]
        answers = split_symbol.join(answers)
        input_list.append("Question: " + question + ' Context: ' + context)
        answer_list.append(answers)

    def tokenizer_fun(input_list, max_len):
        encodings = tokenizer.tokenizer.encode_batch(input_list)
        max_len_batch = 0
        ids_b, masks_b = [], []
        for encoding in encodings:
            ids = encoding.ids
            masks = encoding.attention_mask
            if len(ids) > max_len:
                ids = ids[:max_len]
                masks = masks[:max_len]
            if len(ids) > max_len_batch:
                max_len_batch = len(ids)
            ids_b.append(ids)
            masks_b.append(masks)
        for ids, masks in zip(ids_b, masks_b):
            while len(ids) < max_len_batch:
                ids.append(0)
                masks.append(0)
        return ids_b, masks_b

    input_ids, input_masks = tokenizer_fun(input_list, max_length)
    target_ids, _ = tokenizer_fun(answer_list, max_length)
    target_ids = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in
        target_ids
    ]
    input_ids = Tensor(input_ids, mindspore.int32)
    input_masks = Tensor(input_masks, mindspore.int32)
    target_ids = Tensor(target_ids, mindspore.int32)
    return input_ids, input_masks, target_ids


def evaluate(model, test_examples, eval_batch_size, max_len):
    model.eval()
    step_count = len(test_examples) // eval_batch_size
    if step_count * eval_batch_size < len(test_examples):
        step_count += 1
    step_trange = trange(step_count)
    golds, preds = {}, {}
    for step in step_trange:
        beg_index = step * eval_batch_size
        end_index = min((step + 1) * eval_batch_size, len(test_examples))
        batch_example = [example for example in test_examples[beg_index:end_index]]
        input_ids, input_masks, target_ids = get_input_feature(batch_example, max_len)
        # spans_predict = model(input_ids, input_masks)

        t5_output = self.t5_model.generate(
            input_ids=input_ids,
            max_length=self.max_len,
            attention_mask=input_masks,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        output_sequences = t5_output.sequences
        predicts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        spans_predict = [predict.split(split_symbol) for predict in predicts]

        for sample, spans_p in zip(batch_example, spans_predict):
            sample['pred'] = spans_p
            id = sample['id']
            answers = [item[0] for item in sample['answers']]
            golds[id] = answers
            preds[id] = spans_p
            result_score_item = multi_span_evaluate(copy.deepcopy({'1': spans_p}),
                                               copy.deepcopy({"1": answers}))
            sample['em_f1'] = result_score_item['em_f1']
            sample['overlap_f1'] = result_score_item['overlap_f1']

    result_score = multi_span_evaluate(copy.deepcopy(preds), copy.deepcopy(golds))
    result_score = {
        'em_f1': result_score['em_f1'],
        'overlap_f1': result_score['overlap_f1']
    }
    return result_score, test_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        default='t5-small',
                        type=str)
    parser.add_argument("--debug",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--gpu",
                        default="0",
                        type=str)
    parser.add_argument("--dataset_name",
                        default='msqa',
                        type=str)
    parser.add_argument("--results_save_path",
                        default='./results/',
                        type=str)
    parser.add_argument("--train_batch_size",
                        default=24,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--output_dir",
                        default='./outputs/',
                        type=str,
                        help="The output dreader2ctory whretriever the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=False,
                        type=ast.literal_eval,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--max_len",
                        default=512,
                        type=int)
    parser.add_argument("--lr",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num",
                        default=6,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    split_symbol = ' # '
    only_eval = False
    debug = args.debug
    dataset_name = args.dataset_name
    output_model_path = f'./outputs/exercise/'
    path_save_result = f'./datasets/{dataset_name}/exercise/'

    if 'msqa' in dataset_name:
        evaluate_fun = evaluate_msqa

    os.makedirs(path_save_result, exist_ok=True)
    data_path_train = f'datasets/{dataset_name}/inition/train.json'
    if debug:
        dataset_examples = read_dataset(data_path_train)[:10]
    else:
        dataset_examples = read_dataset(data_path_train)
    dataset_size = len(dataset_examples)
    for i in range(2):
        if i == 0:
            train_examples = dataset_examples[:dataset_size // 2]
            test_examples = dataset_examples[dataset_size // 2:]
        else:
            train_examples = dataset_examples[dataset_size // 2:]
            test_examples = dataset_examples[:dataset_size // 2]
        train_size = len(train_examples)
        dev_size = int(train_size * 0.2)
        dev_examples = train_examples[:dev_size]
        train_examples = train_examples[dev_size:]

        train_batch_size = args.train_batch_size
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        print('init tokenizer')
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        print('init model')
        print(json.dumps({"lr": args.lr, "model": args.model_name, "seed": args.seed,
                          "bs": args.train_batch_size,
                          "epoch": args.epoch_num,
                          "train_path": data_path_train,
                          "train_size": len(train_examples),
                          "dev_size": len(dev_examples),
                          "test_size": len(test_examples),
                          'max_len': args.max_len,
                          'path_save_result': path_save_result,
                          'output_model_path': output_model_path,
                          'init_checkpoint': args.init_checkpoint}, indent=2))


        if only_eval:
            args.init_checkpoint = output_model_path + 'model.ckpt'

        if args.init_checkpoint:
            init_checkpoint = f'{output_model_path}/model{i}.ckpt'
            checkpoint = mindspore.load_checkpoint(init_checkpoint)
            print('init from:', init_checkpoint)

        warm_up_ratio = 0.05
        optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)

        step_count, step_all, early_stop = 0, 0, 0
        best_dev_rouge_score, best_test_rouge_score = 0, 0
        best_test_acc = 0
        best_dev_acc = 0
        best_dev_result, best_test_result = None, None
        for epoch in range(args.epoch_num):
            tr_loss, nb_tr_steps = 0, 0.1
            early_stop += 1
            order = list(range(len(train_examples)))
            random.seed(args.seed + epoch)
            random.shuffle(order)
            model.set_train()
            step_count = len(train_examples) // train_batch_size
            if step_count * train_batch_size < len(train_examples):
                step_count += 1
            step_trange = trange(step_count)
            for step in step_trange:
                step_all += 1
                beg_index = step * train_batch_size
                end_index = min((step + 1) * train_batch_size, len(train_examples))
                order_index = order[beg_index:end_index]
                batch_example = [train_examples[index] for index in order_index]
                input_ids, input_masks, target_ids = get_input_feature(batch_example, args.max_len)


                outputs = model.generate(input_ids=input_ids,
                                         attention_mask=input_masks,
                                         max_length=100)
                print(outputs)
                print('outputs:', len(outputs))

                t5_output = model(input_ids=input_ids, attention_mask=input_masks, labels=target_ids)
                loss = t5_output[0]

                def forward_fn(input_ids, input_masks, target_ids):
                    t5_output = model(input_ids=input_ids, attention_mask=input_masks, labels=target_ids)
                    loss = t5_output[0]
                    logits = t5_output[1]
                    return loss, logits

                grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
                (loss, _), grads = grad_fn(input_ids, input_masks, target_ids)
                optimizer(grads)

                tr_loss += loss.asnumpy()
                nb_tr_steps += 1
                loss_show = ' Epoch:' + str(epoch) + " loss:" + str(
                    round(tr_loss / nb_tr_steps, 4))
                step_trange.set_postfix_str(loss_show)

            if epoch > 4:
                result_score_dev, results_dev = evaluate(model, dev_examples, args.eval_batch_size, args.max_len)
                scores = sum([result_score_dev[key] for key in result_score_dev.keys()])
                print(result_score_dev)
                if scores > best_dev_acc:
                    best_dev_result = result_score_dev
                    best_dev_acc = scores
                    save_model(output_model_path, model, f'model{i}.ckpt')
                    print('save new best')
                    result_score_test, results_test = evaluate(model, test_examples, args.eval_batch_size, args.max_len)
                    best_test_result = result_score_test
                    print('test:', result_score_test)
                    save_dataset(path_save_result, f'/test{i}.json', results_test)

        print('best_dev_result:', best_dev_result)
        print('best_test_result:', best_test_result)
        print(path_save_result)

    dataset_labeled0 = read_dataset(path_save_result + f'/test{0}.json')
    dataset_labeled1 = read_dataset(path_save_result + f'/test{1}.json')
    dataset_labeled = dataset_labeled0 + dataset_labeled1
    save_dataset(path_save_result, f'/train.json', dataset_labeled)
