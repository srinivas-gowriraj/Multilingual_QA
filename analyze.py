import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import sys

def regex_split(df):
    pattern = r"<\w+>"
    question_length = []
    context_length = []
    turns = []
    for query in df['query']:
        query = re.split(pattern, query)
        question_length.append(len(query[1].split()))
        context = ' '.join(query[2:])
        context_length.append(len(context.split()))
        turns.append(len(query)-2)
    df['answer_length'] = [len(answer.lstrip("<response>").split()) for answer in df['response']]

        

    df['question_length'] = question_length
    df['context_length'] = context_length
    df['turns'] = turns
    print(len(df))

    #plot seperate histogram for question length, context length, turns, answer length
    #question length
    plt.hist(df['question_length'], bins=15)
    plt.title('Question Length')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.savefig('question_length.png')
    plt.clf()

    #context length
    plt.hist(df['context_length'], bins=15)
    plt.title('Paragraph Length')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.savefig('context_length.png')
    plt.clf()

    #turns
    plt.hist(df['turns'], bins=[0,2,4,6,8,12,16,20])
    plt.title('Turns')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.savefig('turns.png')
    plt.clf()

    #answer length
    plt.hist(df['answer_length'], bins=15)
    plt.title('Answer Length')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.savefig('answer_length.png')
    plt.clf()

    print(df['turns'].describe())
    



def main():
    #load train_frvi.csv
    df = pd.read_csv('train_frvi.csv')
    regex_split(df)


if __name__ == '__main__':
    main()