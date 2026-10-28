#phase 1: setup and preprocessing
#import
import re
import random
from collections import Counter

class PreProcessor:
    def __init__(self, lowercase=True):
        self.lowercase = lowercase


    #cleaning data: lowercase and extra white spaces
    def clean(self, text):
        if self.lowercase:
            text = text.lower()


        text = re.sub(r'\s+', ' ', text).strip()
        return text


    #splits text into sentences
    def split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        return sentences


    #converts sentence string into list of tokens including punctuation
    def tokenize(self, sentence):
        tokens = re.findall(r"[\w']+|[.,!?;]", sentence)
        return tokens


    #adds padding
    def pad(self, tokens, n):
        start_pads = ['<s>'] * (n - 1)
        return start_pads + tokens + ['</s>']

#implementing UNK handling
def handle_unk(sentences, unk_threshold=2):
    word_counts=Counter()

    #count Unigram
    for sent in sentences:
        for token in sent:
            if token not in ['<s>', '</s>']:
                word_counts[token] += 1

    # identify rare words
    rare_words = {w for w, c in word_counts.items() if c < unk_threshold}

    # replace rare words with <UNK>
    new_sentences = []
    for sent in sentences:
        new_sent = [
            '<UNK>' if token in rare_words else token
            for token in sent
        ]
        new_sentences.append(new_sent)

    return new_sentences
   
#EXECUTION
#1. load data
with open('eng_news_2024_100K-sentences.txt', 'r', encoding='utf-8') as file:
    #read first 10000 lines
    data = [line.strip() for _, line in zip(range(100000), file)]




#2. process
prep=PreProcessor()
processed_sentences=[]


for line in data:
    cleaned=prep.clean(line)
    sentences=prep.split_sentences(cleaned)
    for sentence in sentences:
        tokens=prep.tokenize(sentence)
        padded=prep.pad(tokens,3)
        processed_sentences.append(padded)




#3. train/text split
random.seed(42)
random.shuffle(processed_sentences)
split_index=int(0.8*len(processed_sentences))
train_data=processed_sentences[:split_index]
test_data=processed_sentences[split_index:]


train_data = handle_unk(train_data, unk_threshold=2)
test_data  = handle_unk(test_data, unk_threshold=2)


print(f"Processed {len(processed_sentences)} sentences.")
print(f"Sample: {train_data[0]}")


#phase 2: ngram model
import time


class NGramModel:
    def __init__(self):
        #make count for 1, 2 and 3 grams
        self.counts = {1: Counter(), 2: Counter(), 3: Counter()}
        self.vocab = set()
        self.vocab = set(['<UNK>'])
   
    def count_ngrams(self, sentences):
        start_time=time.time()
        for tokens in sentences:
            self.vocab.update([t for t in tokens if t not in ['<s>', '</s>']])#update vocab excluding markers
            length = len(tokens)
            for n in range(1, 4):
                for i in range(length - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    self.counts[n][ngram] += 1
                    self.vocab.update(ngram)
        elapsed_time=time.time()-start_time
        return elapsed_time
   
#execution
counter=NGramModel() #intialize counter
time_taken=counter.count_ngrams(train_data)
print(f"N-gram counting completed in {time_taken:.2f} seconds.")
print(f"Vocabulary size: {len(counter.vocab)}")
print(f"Sample unigram counts: {counter.counts[1].most_common(5)}")
print(f"Sample bigram counts: {counter.counts[2].most_common(5)}")
print(f"Sample trigram counts: {counter.counts[3].most_common(5)}")


#test case for my protfolio section 4.1(mirochecks)
test_sent = [['<s>', '<s>', 'the', 'cat', 'sat', '</s>']]
test_counter = NGramModel()
test_counter.count_ngrams(test_sent)
#verification
assert test_counter.counts[2][('the', 'cat')] == 1
print("Micro-check: Bigram count for ('the', 'cat') is correct.")


#engineering hygeine snipper fro p[art 5 of my portfolio
import sys
#calculate size of trigram dictionary
trigram_size = sys.getsizeof(counter.counts[3]) / (1024 * 1024)
print(f"Trigram Memory: {trigram_size:.2f} MB")



#phase 3: model and inference logic
import math
class NGramLanguageModel:
    def __init__(self, ngram_counts, flavor="vanilla", unk_threshold=2):
        self.counter=ngram_counts
        self.flavor=flavor
        self.vocab=ngram_counts.vocab
        self.ngram_counts = ngram_counts.counts
        self.vocab_size = len(self.vocab)

    def map_unk(self, tokens):
        return [
            token if token in self.vocab else '<UNK>'
            for token in tokens
        ]


    def get_ngram_prob(self, ngram, flavor=None):
        flavor= flavor or self.flavor
        n=len(ngram)
        self.ngram_counts[n][ngram] 

        #get counts for the ngram and its prefix
        ngram_count=self.ngram_counts[n][ngram]
        context=len(ngram)

        if n == 1:
            count_context = sum(self.counter.counts[1].values())
        else:
            context=ngram[:-1]
            count_context = self.counter.counts[n-1][context]


        #vanilla flavor
        if flavor=="vanilla":
            if count_context==0: 
                return 0
            return ngram_count / count_context
       
        #laplace flavor
        elif flavor=="laplace":
            return (ngram_count + 1) / (count_context + self.vocab_size)
        
    #calculate probability of a sentence
    def sentence_probability(self, tokens,n=3, flavor=None):
        tokens = self.map_unk(tokens)
        log_prob=0

        for i in range(len(tokens)-n+1):
            ngram=tuple(tokens[i:i+n])
            prob=self.get_ngram_prob(ngram,flavor)
            if prob==0:
                return 0
            log_prob+=math.log(prob)
        return math.exp(log_prob)

    def linear_interpolation(self, tokens):
        tokens = self.map_unk(tokens)
        log_prob = 0
        for i in range(2, len(tokens)):
            tri = tuple(tokens[i-2:i+1])
            bi = tuple(tokens[i-1:i+1])
            uni = tuple(tokens[i:i+1])
        
            p_tri = self.get_ngram_prob(tri, flavor="laplace")
            p_bi = self.get_ngram_prob(bi, flavor="laplace")
            p_uni = self.get_ngram_prob(uni, flavor="laplace")
        
            # interpolation weights depending on n
            if n == 1:
                mixed = p_uni
            elif n == 2:
                mixed = 0.3*p_bi + 0.7*p_uni
            else:
                mixed = 0.6*p_tri + 0.3*p_bi + 0.1*p_uni

            log_prob += math.log(mixed)
        return math.exp(log_prob)


#calculate perplexity
def calculate_perplexity(model, data, n):
    log_prob_sum = 0
    token_count = 0

    for sent in data:
        prob = model.sentence_probability(sent, n)
        if prob > 0:
            log_prob_sum += math.log(prob)
            token_count += len(sent) - (n - 1)

    return math.exp(-log_prob_sum / token_count)

def generate_sentence(model, max_len=15):
    sentence = ['<s>'] * 2  # trigram padding
    for _ in range(max_len):
        context = tuple(sentence[-2:])
        # get all candidate words
        candidates = [(ng[2], c) for ng, c in model.ngram_counts[3].items() if ng[:2] == context]
        if not candidates:
            break
        # sample next word based on probability
        total = sum(c for _, c in candidates)
        import random
        r = random.uniform(0, total)
        upto = 0
        for word, count in candidates:
            upto += count
            if upto >= r:
                sentence.append(word)
                break
        if sentence[-1] == '</s>':
            break
    return ' '.join(sentence[2:-1])  # remove <s> padding and </s>

#model evaluation
vanilla = NGramLanguageModel(counter, flavor="vanilla")
laplace = NGramLanguageModel(counter, flavor="laplace")

print("\n" + "="*55)
print(f"{'Flavor':<15} | {'Unigram':<10} | {'Bigram':<10} | {'Trigram':<10}")
print("-" * 55)
for lab, mod in [("Vanilla", vanilla), ("Laplace", laplace),]:
    u = calculate_perplexity(mod, test_data, 1)
    b = calculate_perplexity(mod, test_data, 2)
    t = calculate_perplexity(mod, test_data, 3)
    print(f"{lab:<15} | {u:10.2f} | {b:10.2f} | {t:10.2f}")
print("="*55)

print("\nLinear Interpolation Perplexity:")
for n in [1, 2, 3]:
    log_prob_sum = 0.0
    token_count = 0
    for sent in test_data:
        prob = vanilla.linear_interpolation(sent)  # use interpolation
        if prob > 0:
            log_prob_sum += math.log(prob)
            token_count += len(sent) - 2  # because trigram
    ppl = math.exp(-log_prob_sum / token_count)
    print(f"{n}-gram interpolation perplexity: {ppl:.2f}")

#generate example sentences
print("\nGenerated Sentences (Vanilla Trigram LM):")  # add generation output
for _ in range(5):
    print(generate_sentence(vanilla))
    