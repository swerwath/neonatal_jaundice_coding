import snowballstemmer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from random import shuffle
from gensim.models import Phrases
import pickle
from get_data_from_csvs import get_relevant_hamds, get_relevant_notesets, get_all_neonate_hamds

stemmer = snowballstemmer.stemmer('english');
punct = '.,_/#-><()&%[]:=;+?!\'"1234567890'

stop_words = set(stopwords.words('english'))
extra_removal = set(["cm", "mm", "x", "please", "is", "are", "be", "been"])
to_remove = stop_words.union(extra_removal)

replacement_file_path = "./data/clever_replacements"
replacements = pickle.load(open(replacement_file_path, 'rb'))

# Preprocesses a single noteset for a hospital stay
def preprocess_noteset(noteset):
    noteset = noteset.lower()

    # Get rid of anonymized tokens
    noteset = re.sub(r'\[\*\*.*\*\*\]', ' ', noteset)

    # Punctuation replacement
    noteset = noteset.replace('\n', ' ').replace('Dr.', 'Dr')
    for p in punct:
        noteset = noteset.replace(p, '')
    noteset = noteset.replace('  ', ' ')

    noteset = ' ' + noteset + ' '
    for r in replacements:
        noteset = noteset.replace(r[0], r[1])
    noteset = noteset.strip()

    # Word tokenize
    words = word_tokenize(noteset)

    # Remove stopwords
    words = [w for w in words if (not w in to_remove) and (not w.isdigit())]

    # Apply word stemming
    # words = stemmer.stemWords(words)

    # Apply negation
    nrange = 3
    for i in [i for i, w in enumerate(words) if w.strip() == 'NEGEX']:
        to_negate = [i + j for j in range(nrange) if i + j < len(words)]
        for j in to_negate:
            words[j] = ''


    words = [w for w in words if w.strip() != '']
    noteset = ' '.join(words)

    return noteset


import argparse
import pickle
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('csv_dir_path')
    parser.add_argument('out_path')
    args = parser.parse_args()

    neonate_hamds = get_all_neonate_hamds(dir_path=args.csv_dir_path)
    pos_hamds = get_relevant_hamds(dir_path=args.csv_dir_path)
    neg_hamds = neonate_hamds.difference(pos_hamds)

    print(len(neonate_hamds))
    print(len(pos_hamds))
    print(len(neg_hamds))

    pos_notesets = get_relevant_notesets(pos_hamds).values()
    neg_notesets = get_relevant_notesets(neg_hamds).values()

    preproced_pos = list(map(preprocess_noteset, pos_notesets))
    preproced_neg = list(map(preprocess_noteset, neg_notesets))

    all = preproced_pos + preproced_neg
    stream = [word_tokenize(a) for a in all]
    bigram = Phrases(stream, min_count=5, threshold=15)

    preproced_pos = map(lambda x: " ".join(bigram[word_tokenize(x)]), preproced_pos)
    preproced_neg = map(lambda x: " ".join(bigram[word_tokenize(x)]), preproced_neg)


    pos = [(p, 1) for p in preproced_pos]
    neg = [(n, 0) for n in preproced_neg]

    all = list(filter(lambda x : len(x[0]) >= 1000, pos + neg))
    shuffle(all)

    with open(args.out_path, 'wb') as out_file:
        pickle.dump(all, out_file)
