from collections import Counter


def sentence2token(sentence: str):
    lines = sentence.split("\n")

    words = []
    for line in lines:
        # print(line)
        if not line:
            continue
        if line[0] != "<":
            # print(line)
            splitted = specialWordsplit(line.split())
            words += ["<BOS>"] + splitted + ["<EOS>"]

    return words


def specialWordsplit(seq: list):
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
    dictionary = {}
    limit_of_counts = 0
    for word, count in Counter(tokens).most_common():
        if count > freqs:
            dictionary[word] = len(dictionary)
