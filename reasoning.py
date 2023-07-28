import argparse
import json
import os
from utils import read_dataset, save_dataset
from tqdm import trange
from prompt_msqa import TaskTemp, ConcatTemp, FeedbackTemp
from utils import parsing
from eval_scripts.eval_script_msqa import evaluate_msqa
from eval_scripts.eval_script_quoref import evaluate_quoref
from eval_scripts.eval_script_drop import evaluate_drop
from eval_scripts.eval_script_keyphrase import evaluate_keyphrase
from copy import deepcopy
import ast
from LLMs import ChatGPT
import time

def list_to_dict(dataset):
    dataset_dict = {}
    for sample in dataset:
        dataset_dict[sample['id']] = sample
    return dataset_dict

def feedback(answers, pred_answers):
    answers_norm = [ans_item.strip().lower() for ans_item in answers]
    pred_norm = [ans_item.strip().lower() for ans_item in pred_answers]

    correct_answers = []
    for ans in pred_answers:
        if ans.strip().lower() in answers_norm:
            correct_answers.append(ans)

    incorrect_answers = []
    for ans in pred_answers:
        if ans.strip().lower() not in answers_norm:
            incorrect_answers.append(ans)
    miss_answers = []
    for ans in answers:
        if ans.strip().lower() not in pred_norm:
            miss_answers.append(ans)
    return correct_answers, incorrect_answers, miss_answers


def question_answering(demo_examples, test_example, exercise_results):
    test_question = test_example['question']
    test_context = test_example['context']
    prompt_test = TaskTemp(test_context, test_question)
    prompt_plus = []
    for demo_example in demo_examples:
        # print('answers0:',answers)
        answers = demo_example['answers']
        if isinstance(answers[0], str) is False:
            answers = [item[0] for item in answers]
        # print('answers1:',answers)
        question = demo_example['question']
        context = demo_example['context']
        prompt_demo_i = TaskTemp(context, question, answers)
        exercise_result = exercise_results[demo_example['id']]
        if isinstance(answers[0], str) is False:
            answers = [item[0] for item in answers]
        correct_answers, incorrect_answers, miss_answers = feedback(answers, exercise_result)
        if use_correct_answer is False:
            correct_answers = None
        if use_incorrect_answer is False:
            incorrect_answers = None
        if use_miss_answer is False:
            miss_answers = None
        prompt_fb_i = FeedbackTemp(correct_answers, incorrect_answers, miss_answers)
        prompt_plus_i = f'{prompt_demo_i}\n{prompt_fb_i}'
        prompt_plus.append(prompt_plus_i)
    prompt = ConcatTemp(prompt_plus, prompt_test)
    output = LLM_fun(prompt)
    pred_answers = parsing(output)
    return pred_answers, prompt, output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
                        default='msqa',
                        type=str)
    parser.add_argument("--llm",
                        default='gpt3',
                        # default='chatgpt',
                        type=str)
    parser.add_argument("--baseline",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--use_correct_answer",
                        default=True,
                        type=ast.literal_eval)
    parser.add_argument("--use_incorrect_answer",
                        default=True,
                        type=ast.literal_eval)
    parser.add_argument("--use_miss_answer",
                        default=True,
                        type=ast.literal_eval)

    parser.add_argument("--retrieval_method",
                        default='bm25',
                        type=str)
    parser.add_argument("--case_num",
                        default=3,
                        type=int)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    retrieval_method = args.retrieval_method
    if args.baseline:
        args.use_correct_answer, args.use_incorrect_answer, args.use_miss_answer = False, False, False
    llm_name = args.llm
    case_num = args.case_num
    result_path = f"results/{dataset_name}/{llm_name}/{args.retrieval_method}"
    use_correct_answer, use_incorrect_answer, use_miss_answer = args.use_correct_answer, args.use_incorrect_answer, args.use_miss_answer
    print(use_correct_answer, use_incorrect_answer, use_miss_answer)
    if use_correct_answer:
        result_path += '_correct'
    if use_incorrect_answer:
        result_path += '_incorrect'
    if use_miss_answer:
        result_path += '_miss'
    if use_correct_answer == False and \
            use_incorrect_answer == False and \
            use_miss_answer == False:
        result_path += '_baseline'
    result_path += '_case' + str(args.case_num)
    print(result_path)
    path_train = f'datasets/{dataset_name}/initial/train.json'
    path_test = f'datasets/{dataset_name}/initial/used.json'
    if dataset_name != 'drop':
        ir_results_path = f'datasets/{dataset_name}/{retrieval_method}/test.json'
    else:
        ir_results_path = f'datasets/{dataset_name}/{retrieval_method}/valid.json'
    exercise_results_path = f'datasets/{dataset_name}/exercise/train.json'

    dataset_train = read_dataset(path_train)
    if os.path.exists(f'{result_path}/prediction.json'):
        dataset_test = read_dataset(f'{result_path}/prediction.json')
    else:
        dataset_test = read_dataset(path_test)
    ir_results = read_dataset(ir_results_path)
    exercise_results = read_dataset(exercise_results_path)

    dataset_train = list_to_dict(dataset_train)
    if dataset_name == 'msqa':
        evaluate_fun = evaluate_msqa
    elif dataset_name == 'quoref':
        evaluate_fun = evaluate_quoref
    elif dataset_name == 'drop':
        evaluate_fun = evaluate_drop
    else:
        evaluate_fun = evaluate_keyphrase

    LLM_fun = ChatGPT

    print(json.dumps({
        'case_num': args.case_num,
        'llm': llm_name,
        'dataset_name': args.dataset_name,
        'use_correct_answer': args.use_correct_answer,
        'use_incorrect_answer': args.use_incorrect_answer,
        'use_miss_answer': args.use_miss_answer,
        'dataset_train': len(dataset_train),
        'dataset_test': len(dataset_test),
        'path_train': path_train,
        'path_test': path_test,
        'path_results': result_path
    }, indent=2))

    step_trange = trange(len(dataset_test))
    preds, golds = {}, {}
    for step in step_trange:
        sample = dataset_test[step]
        q_id = sample['id']
        ir_result = ir_results[q_id][:case_num]
        if 'pred_answers' not in sample:
            demo_examples = []
            for ir_item in ir_result:
                demo_examples.append(dataset_train[ir_item['id']])
            pred_answers, input_prompt, output = question_answering(demo_examples, sample, exercise_results)
            # print(input_prompt)
            # print('pred_answers:',pred_answers)
            # print('-------')
            sample['pred_answers'] = pred_answers
            sample['input_prompt'] = input_prompt
            sample['output'] = output
            preds[q_id] = pred_answers
            if 'answer_ori' in sample:
                golds[q_id] = sample
            else:
                golds[q_id] = sample['answers']
            result_scores = evaluate_fun(deepcopy(preds), deepcopy(golds))
            score = ' '.join([f'{key}: {round(result_scores[key], 2)}' for key in result_scores.keys()])
            step_trange.set_postfix_str(f'{score}')
            save_dataset(result_path, 'prediction.json', dataset_test)
        else:
            preds[q_id] = sample['pred_answers']
            golds[q_id] = sample['answers']
        # try:
        #
        # except Exception as e:
        #     print(e)
    result_scores = evaluate_fun(deepcopy(preds), deepcopy(golds))
    score = ' '.join([f'{key}: {round(result_scores[key], 2)}' for key in result_scores.keys()])
    print(score)