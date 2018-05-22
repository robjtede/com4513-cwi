"""
Scrabble score functions.

Scores taken from official scrabble website:
https://scrabble.hasbro.com/en-us/faq
"""

scores = {
    'a': 1,
    'b': 3,
    'c': 3,
    'd': 2,
    'e': 1,
    'f': 4,
    'g': 2,
    'h': 4,
    'i': 1,
    'j': 8,
    'k': 5,
    'l': 1,
    'm': 3,
    'n': 1,
    'o': 1,
    'p': 3,
    'q': 10,
    'r': 1,
    's': 1,
    't': 1,
    'u': 1,
    'v': 4,
    'w': 4,
    'x': 8,
    'y': 4,
    'z': 10
}


def score_letter(letter):
    letter = letter.lower()
    return scores[letter] if letter in scores else 0


def scrabble_score(word):
    return sum([score_letter(x) for x in word])


def avg_scrabble_score(word):
    return scrabble_score(word) / len(word)


if __name__ == '__main__':
    print(scrabble_score('flexed'), avg_scrabble_score('flexed'))
    print(scrabble_score('their'), avg_scrabble_score('their'))
    print(scrabble_score('muscles'), avg_scrabble_score('muscles'))
    print(scrabble_score('and'), avg_scrabble_score('and'))
    print(scrabble_score('I'), avg_scrabble_score('I'))
