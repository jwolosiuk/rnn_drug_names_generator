import numpy as np

from utils import encode

def generate_name(start="", model=None, chars=None, next_chars=50, temp=1.):
    word = start

    for i in range(next_chars):
        pred = model(encode(word, end=False))
        next_letter_prob = pred[0, :, -1].softmax(dim=0).cpu().detach().numpy()

        next_letter_prob = next_letter_prob ** (1 / temp)
        next_letter_prob = next_letter_prob / next_letter_prob.sum()

        next_char_id = np.random.choice(len(next_letter_prob), 1, p=next_letter_prob)[0]

        word += chars[next_char_id]

        if chars[next_char_id] == '<':
            break
    return word


def calculate_probability(name, model, char2id, temp=1.0):
    word = ''

    prob = 1
    for i, next_char in enumerate(name):
        pred = model(encode(word, end=False))
        next_letter_prob = pred[0, :, -1].softmax(dim=0).cpu().detach().numpy()
        next_letter_prob = next_letter_prob ** (1 / temp)
        next_letter_prob = next_letter_prob / next_letter_prob.sum()

        next_char_id = char2id[next_char]
        next_char_prob = next_letter_prob[next_char_id][0]

        prob *= next_char_prob
        word += next_char
    return prob


def analyze_name(name, model, dataset, temp=1.0):
    prob = calculate_probability(name, model, temp=temp)
    print(f"{name} ({'NIE' if name not in dataset else ''}ISTNIEJE), probability: {prob}")