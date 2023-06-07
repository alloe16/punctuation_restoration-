
import sys
import random
import numpy as np


if len(sys.argv) > 1:
    if isinstance(sys.argv[1], str):
        with open(sys.argv[1], "r") as file:
            lines = file.read()
        processed_text = lines.split("\n")
else:
    print("There is no specified text to process")


processed_words = [sentence.split() for sentence in processed_text]


flat_words = [item for sublist in processed_words for item in sublist]
unique_words = list(set(flat_words))

print(f"Unique words in text: {len(unique_words)}")


punctList = [
    ".PERIOD",
    "?QUESTIONMARK",
    "!EXCLAMATIONMARK",
    ",COMMA",
    ";SEMICOLON",
    ":COLON",
    "-DASH",
]

for elem in punctList:
    if elem in unique_words:
        delIdx = unique_words.index(elem)
        del unique_words[delIdx]


punctCountsTmp = [flat_words.count(punct) for punct in punctList]
punctCounts = list(zip(punctList, punctCountsTmp))

print(f"The occurrences of punctuation marks in the text: {punctCounts}")


def apply_wer(
    nperc,
    wordList=processed_words[: int(0.2 * len(processed_words))],
    randomWords=unique_words,
    punctuations=punctList,
):
    scoreList = []

    for i in range(len(wordList)):
        scoreList.append(list(np.random.uniform(0, 1, len(wordList[i]))))

    dels = 0.25 * nperc
    ins = dels + 0.25 * nperc
    subs = ins + 0.5 * nperc
    for i in range(len(scoreList)):
        for j in range(len(scoreList[i])):

            if wordList[i][j] not in punctuations:
                if scoreList[i][j] < dels:
                    wordList[i][j] = "DELETION"
                elif scoreList[i][j] < ins:
                    wordList[i].insert(j + 1, random.choice(randomWords))
                elif scoreList[i][j] < subs:
                    wordList[i][j] = random.choice(randomWords)


    wordListFinish = [
        list(filter(lambda x: x not in ["DELETION"], sublist)) for sublist in wordList
    ]
    return wordListFinish
