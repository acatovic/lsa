import argparse
import io
import json
import numpy as np
import os
import re
import sys

from collections import Counter

# constants for storing indexed output
out_directory = './out'
docs_path = os.path.join(out_directory, 'docs.txt')
sk_inv_path = os.path.join(out_directory, 'sk_inv.bin')
uk_path = os.path.join(out_directory, 'uk.bin')
vk_t_path = os.path.join(out_directory, 'vk_t.bin')
word_to_ix_path = os.path.join(out_directory, 'word_to_ix.json')

def build_query_vector(s, word_to_ix, Sk_inv, Uk):
    q = np.zeros(len(word_to_ix))
    words = parse(s)
    for word in words:
        if word in word_to_ix:
            q[word_to_ix[word]] += 1
    
    q_hat = np.dot(np.dot(q, Uk), Sk_inv)

    return q_hat

def build_td_matrix(docs, word_to_ix, use_tfidf=False):
    num_docs = len(docs)
    num_words = len(word_to_ix)
    td_matrix = np.zeros((num_words, num_docs))

    for doc_ix, doc in enumerate(docs):
        words = parse(doc)
        for word in words:
            if word in word_to_ix:
                td_matrix[word_to_ix[word]][doc_ix] += 1
    
    if use_tfidf:
        td_matrix = tfidf(td_matrix)
    
    return td_matrix

def build_word_to_ix(docs, stopwords=None):
    if stopwords is None:
        stopwords = set()
    
    c = Counter()
    for doc in docs:
        c.update(set([word for word in parse(doc) if word not in stopwords]))
    
    word_to_ix = {}
    for word in c:
        if c[word] > 1:
            word_to_ix[word] = len(word_to_ix)
    
    return word_to_ix

def cosine_distance(u, v):
    uv = np.dot(u, v)
    if uv == 0:
        return 0
    return uv/(np.linalg.norm(u) * np.linalg.norm(v))

def find_matches(q_hat, Vk_t, min_score=0.9):
    matches = []
    for ix, d in enumerate(Vk_t):
        r = cosine_distance(q_hat, d)
        if r >= min_score:
            matches.append((ix, r))
    
    matches = sort_matches(matches)

    return matches

def index(path, rank=2, use_stopwords=False, use_tfidf=False):
    docs = load_docs(path)

    if len(docs) < 2:
        sys.exit('Error: number of documents must be greater than 1')
    
    if  rank < 2 or rank > len(docs):
        sys.exit('''Error: rank must be smaller or equal to number of 
            documents, and greater than 1''')
    
    stopwords = None
    if use_stopwords:
        stopwords = load_stopwords('./stopwords.txt')
    
    word_to_ix = build_word_to_ix(docs, stopwords=stopwords)

    Uk, Sk_inv, Vk_t = svd(docs, word_to_ix, rank, use_tfidf)

    mkdir(out_directory)
    save_docs(docs, docs_path)
    save_word_to_ix(word_to_ix, word_to_ix_path)
    save_matrix(Sk_inv, sk_inv_path)
    save_matrix(Uk, uk_path)
    save_matrix(Vk_t, vk_t_path)

def load_docs(path):
    docs = []

    if os.path.isfile(path):
        # all docs in one file, each doc on new line
        with open(path) as f:
            for line in f:
                docs.append(line.strip())
    else:
        # path is directory, each doc in separate file
        for name in os.listdir(path):
            filename = os.path.join(path, name)
            with open(filename) as f:
                docs.append(f.read().replace('\n', ' '))
    
    return docs

def load_matrix(path):
    with open(path, 'rb') as f:
        dtype, rows, cols = str(f.readline()).split()
        return np.fromfile(f, dtype=dtype).reshape((int(rows), int(cols)))

def load_stopwords(path):
    """Loads stopwords from `path`, assume each word in new line."""
    stopwords = set()
    with open(path) as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords

def load_word_to_ix(path):
    with open(path) as f:
        return json.load(f)

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == 17:
            pass
        else:
            sys.exit(e)

def nonzeros(u):
    """Return number of non-zero items in list `u`."""
    return len([val for val in u if val != 0])

def parse(s):
    s = preprocess(s)
    return tokenize(s)

def preprocess(s):
    s = s.replace('*', '')
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace('- ', ' ')
    s = s.replace(', ', ' ')
    s = s.replace('. ', ' ')
    s = s.replace(': ', ' ')
    s = s.replace('? ', ' ')
    s = s.replace('! ', ' ')
    s = s.lower()
    return s

def query(s, min_score=0.9):
    try:
        word_to_ix = load_word_to_ix(word_to_ix_path)
        Sk_inv = load_matrix(sk_inv_path)
        Uk = load_matrix(uk_path)
        Vk_t = load_matrix(vk_t_path)
    except IOError:
        return None
    
    q_hat = build_query_vector(s, word_to_ix, Sk_inv, Uk)
    matches = find_matches(q_hat, Vk_t, min_score=min_score)

    if not matches:
        return None
    
    try:
        docs = load_docs(docs_path)
    except IOError:
        return None
    
    return [(docs[ix], score) for ix, score in matches]

def save_docs(docs, path):
    with open(docs_path, 'w') as f:
        for doc in docs:
            f.write('%s\n' %doc)

def save_matrix(m, path):
    with open(path, 'wb+') as f:
        f.write(b'{0:s} {1:d} {2:d}\n'.format(m.dtype, *m.shape))
        m.tofile(f)

def save_word_to_ix(word_to_ix, path):
    with open(path, 'w') as f:
        json.dump(word_to_ix, f)

def sort_matches(matches):
    indexes = list(reversed(np.argsort([score for ix, score in matches])))
    return [matches[ix] for ix in indexes]

def svd(docs, word_to_ix, rank, use_tfidf=True):
    X = build_td_matrix(docs, word_to_ix, use_tfidf=use_tfidf)

    U, s, V = np.linalg.svd(X, full_matrices=True)

    Sk = np.diag(s[:rank])
    Sk_inv = np.linalg.inv(Sk)
    Uk = U[:, :rank]
    Vk = V[:rank, :]
    Vk_t = np.transpose(Vk)

    return Uk, Sk_inv, Vk_t

def tfidf(td_matrix):
    num_terms = len(td_matrix)
    num_docs = len(td_matrix[0])

    for i in range(num_terms):
        idf = np.log10(float(num_docs) / (1 + nonzeros(td_matrix[i])))
        for j in range(num_docs):
            tf = np.log10(1 + td_matrix[i][j])
            td_matrix[i][j] = tf * idf
    
    return td_matrix

def tokenize(s):
    return s.split()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', dest='docs_path')
    parser.add_argument('--min-score', dest='min_score', type=float, default=.9)
    parser.add_argument('--query', dest='query')
    parser.add_argument('--rank', dest='rank', type=int, default=2)
    parser.add_argument('--stopwords', dest='use_stopwords', 
        action='store_true', default=False)
    parser.add_argument('--tfidf', dest='use_tfidf', action='store_true', 
        default=False)

    args = parser.parse_args()

    if not (args.docs_path or args.query):
        parser.print_help()
        sys.exit(1)

    if args.docs_path:
        # indexing
        index(args.docs_path, rank=args.rank, use_stopwords=args.use_stopwords, 
            use_tfidf=args.use_tfidf)
    else:
        # query
        matches = query(args.query, min_score=args.min_score)
        if matches is not None:
            print matches
        else:
            print "No matches found"