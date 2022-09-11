from collections import Counter, defaultdict
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

def get_vocab(X, split=False, vocab_size=10000, visualize=True, num_stop_words=0):
    stop_words = set(stopwords.words('english'))
    raw_words = []
    for x in X:
        if split:
            x = x.split(' ')
        raw_words.extend(x)
    
    word_count = Counter(raw_words)
    vocab, word2int, int2word = [], defaultdict(lambda : 0), defaultdict(lambda : 'UNK')
    counts = []
    j = 1
    for i, (word, count) in enumerate(word_count.most_common(vocab_size)):
        word = word.lower()
        if word not in stop_words and word[0] != '/' and word[0] != '<':
            vocab.append(word)
            counts.append(count)
            word2int[word] = j
            int2word[j] = word
            j += 1

    if visualize:
        plt.figure(figsize=(15, 5))
        plt.bar(vocab[:50], counts[:50])
        plt.xticks(rotation=90)
        plt.show()
        
    return vocab, word2int, int2word