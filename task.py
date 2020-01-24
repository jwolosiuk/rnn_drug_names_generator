import numpy as np

from training import encode, char2id, id2char, dataset, device

def generate_name(model=None, start="", next_chars=50, temp=1.):
    word = start

    for i in range(next_chars):
        tensor = encode(word, end=False)
        tensor = tensor.to(device)
        pred = model(tensor)
        next_letter_prob = pred[0, :, -1].softmax(dim=0).cpu().detach().numpy()

        next_letter_prob = next_letter_prob ** (1 / temp)
        next_letter_prob = next_letter_prob / next_letter_prob.sum()

        next_char_id = np.random.choice(len(next_letter_prob), 1, p=next_letter_prob)[0]

        if next_char_id not in id2char:
            break

        word += id2char[next_char_id]
    return word


def calculate_probability(name, model, temp=1.0):
    word = ''

    prob = 1
    for i, next_char in enumerate(name):
        tensor = encode(word, end=False)
        tensor = tensor.to(device)
        pred = model(tensor)
        next_letter_prob = pred[0, :, -1].softmax(dim=0).cpu().detach().numpy()
        next_letter_prob = next_letter_prob ** (1 / temp)
        next_letter_prob = next_letter_prob / next_letter_prob.sum()

        next_char_id = char2id[next_char]
        next_char_prob = next_letter_prob[next_char_id]

        prob *= next_char_prob
        word += next_char
    return prob


def analyze_name(name, model, temp=1.0):
    prob = calculate_probability(name, model, temp=temp)
    print(f"{name} ({'NIE' if name not in dataset else ''}ISTNIEJE), probability: {prob}")