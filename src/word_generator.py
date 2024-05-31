import random

def load_words(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

def generate_real_word(words):
    word = random.choice(words)
    return word

def generate_words(num_words, words):
    if num_words > len(words):
        raise ValueError("num_words is greater than the number of unique words available")

    generated_words = set()
    while len(generated_words) < num_words:
        word = generate_real_word(words)
        generated_words.add(word)
    return list(generated_words)
