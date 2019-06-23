# Latent Semantic Analysis (LSA)

Simple implementation of latent semantic analysis based on (Dumais et al, 1994). 
LSA is a completely unsupervised method for indexing and topic modelling of text 
corpora, using only a linear algebra technique called singular value 
decomposition (SVD). It turns out that LSA deals well with synonymy and 
polysemy, whereas traditional lexicon-based indexing (e.g. BoW and TF-IDF) fail. Synonymy is where a single object can be described in many different ways. Polysemy is where the same word can describe multiple and semantically different objects.

LSA works by first constructing a term-by-document matrix `A`, optionally 
normalizing the matrix (e.g. using TF-IDF). SVD is then applied, splitting the 
term-document matrix into left and right singular vectors, `U` and `Vt` and singular values, `S`. The top `k` highest singular values are chosen and then 
the three components are multiplied together again to create a lower-rank 
approximation of the original term-document matrix, `Ak = Uk . Sk . Vk_t`. `Uk` 
then becomes latent space for all the terms, and `Vk_t` becomes latent space for 
all the documents. Not only does this operation compress the original matrix, 
thereby saving memory and compute resouces, it also collapses documents into topics/concepts. This gives us the power to find document similarity from a 
query, even when the query doesn't share any terms with the relevant documents. 
E.g. in (Landauer et al, 1988), the query "human computer interaction" also 
retrieves documents "The EPS user interface management system" and "Relation of 
user-perceived response time to error measurement".

After we index our text corpus, we can query/search for similar documents by 
treating our query as a pseudo-document and encoding the query in this latent 
document space. Then we simply perform a cosine distance comparison between the 
query vector and all the document vectors.

A note on reproducing original (Dumais et al, 1994) paper: the term-document 
matrix in table 3 doesn't match the text corpus in table 2; even though they 
don't use any stemming, they implicitly treat "application" and "applications" 
as one and the same, as well as a number of other terms, e.g. "problem" and 
"problems", etc. Therefore I've modified `sample/en.txt` to reflect this, so 
you obtain the same results as in the paper.

Also for stop words, I've adopted the words used in SMART Information Retrieval System, Gerard Salton, Cornell University.

## Pre-requisites

- Python 2.7
- [NumPy](http://www.numpy.org)

## Usage

There are two parts - indexing, and query. You first perform indexing on a 
corpus of text documents, thereby obtaining left and right singular vectors, as 
well as singular values, `Uk`, `Vk_t`, and `Sk` respectively. Then you run a 
query and obtain a set of matching documents and corresponding cosine distance 
scores (the higher the better).

The documents can be either in a single directory with each document in a 
separate file, or they can all be in a single text file, where each document is 
on a separate line (see `sample/en.txt`). The document can be a title, an 
abstract, or a full report/paper/novel/etc.

There are a number of options when indexing, such as whether to use stop words 
(see `--stopwords`) and whether to normalize the term-document matrix by TF-IDF (good idea for any reasonably sized corpus, see `--tfidf`). Also, one of the 
most important parameters is the rank of the approximation matrix (see 
`--rank`). For a dozen documents, a rank of 2-3 is ok, for a large collection of
 documents (thousands) this number should be between 100 and 300. Having a high 
 rank will turn LSA performance into that of a lexicon-based search/indexing, while having a very low rank value will collapse documents into a too-small set 
 of topics/concepts. So the real challenge with LSA is to pick the right rank 
 value (this is often done using cross-validation on production systems).

```bash
python lsa.py --stopwords --docs ./sample/en.txt
```

To query, you specify the query string and optionally the minimum cosine 
distance score (0.9 by default).

```bash
python lsa.py --query "application and theory"
```

## References

Using Linear Algebra for Intelligent Information Retrieval, S. T. Dumais, M. W. 
Berry, G. W. O'Brien, 1994

Indexing by Latent Semantic Analysis, T. K. Landauer S. Deerwester, S. T. Durmais, G. W. Furnas, R. Harshman, 1988