
def TaskTemp(context, question, answers=None):
    prompt = f'Reading the passage: {context}\n' \
             f'Extract spans from the above passage to answer the question:{question}\n' \
             f'Answer as a list e.g.\n1. answer1\n2. answer2\nAnswer:\n'
    if answers is not None:
        answers_str = [str(i + 1) + '. ' + item for i, item in enumerate(answers)]
        answers_str = '\n'.join(answers_str)
        prompt += answers_str + '\n\n'
    return prompt


def FeedbackTemp(correct_answers=None, incorrect_answers=None, miss_answers=None):
    prompt = ""
    if correct_answers is not None:
        if len(correct_answers) == 0:
            correct_answers = "None"
        else:
            correct_answers = [str(i + 1) + '. ' + item for i, item in enumerate(correct_answers)]
            correct_answers = '\n'.join(correct_answers)
        prompt += f"Here are some correct answers responded by other AI model:\n{correct_answers}\n\n"

    if incorrect_answers is not None:
        if len(incorrect_answers) == 0:
            incorrect_answers = "None"
        else:
            incorrect_answers = [str(i + 1) + '. ' + item for i, item in enumerate(incorrect_answers)]
            incorrect_answers = '\n'.join(incorrect_answers)
        prompt += f"Here are some incorrect answers responded by other AI model:\n{incorrect_answers}\n\n"

    if miss_answers is not None:
        if len(miss_answers) == 0:
            miss_answers = "None"
        else:
            miss_answers = [str(i + 1) + '. ' + item for i, item in enumerate(miss_answers)]
            miss_answers = '\n'.join(miss_answers)
        prompt += f"Here are some answers missed by other AI model:\n{miss_answers}\n\n"

    return prompt


def ConcatTemp(demonstrations, test_prompt):
    prompt = ""
    for i, demo in enumerate(demonstrations):
        prompt += F'Example {i+1}:\n{demo}\n'
    prompt += f'Then, answer me a question like the above examples:\n{test_prompt}'
    return prompt

