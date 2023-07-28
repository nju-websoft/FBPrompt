#coding=utf-8
import openai
import time



def ChatGPT(prompt):
    while True:
        try:
            openai.api_key = "your key"
            messages = [
                {"role": "system", "content": "You are a helpful question answering assistant."},
                {"role": "user", "content": prompt},
            ]
            response = openai.ChatCompletion.create(
                # model="gpt-3",
                model="gpt-3.5-turbo-0301",
                messages=messages,
                temperature=0
            )
            # print(response)
            res = response['choices'][0]['message']['content']
            return res
        except Exception as e:
            time.sleep(5)
            continue
