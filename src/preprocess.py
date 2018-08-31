
from nltk import word_tokenize

def sentence2token(sentence):
    lines = sentence.split("\n")

    words = []
    for line in lines:
        # print(line)
        if not line: continue
        if line[0] != "<":
            # print(line)
            splitted = specialWordsplit(line.split())
            words += ["<BOS>"]+ splitted + ["<EOS>"]

    return words

def specialWordsplit(seq:list):
    splitted = []
    for word in seq:
        if word[-1] == ",":
            splitted += [word[:-1]] + [","]
        elif word[-1] == ".":
            splitted += [word[:-1]]
        elif word[-1] == "?":
            splitted += [word[:-1]] + ["?"]
        elif word[-1] == "!":
            splitted += [word[:-1]] + ["!"]
        elif word[-1] == '"':
            splitted += [word[:-1]] + ['"']
        elif word[0] == '"':
            splitted += ['"'] + [word[1:]]
        else:
            splitted += [word]
    return splitted

def token2dic(tokens, freqs=100):
    from collections import Counter
    dictionary = {}
    limit_of_counts = 0
    for word, count in Counter(tokens).most_common():
        # limit_of_counts += count
        if count > freqs:
            dictionary[word] = len(dictionary)
            print(word, count)
    print(dictionary)



if __name__ == "__main__":
    sentence = open("data/train.tags.de-en.en").read()
    words = sentence2token(sentence=sentence)
    token2dic(words, freqs=100)

