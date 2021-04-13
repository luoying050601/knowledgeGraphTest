
import spacy
from datasets import load_dataset
from spacy.matcher import Matcher
from common_util import split_into_sentences,preprocess

nlp = spacy.load('en_core_web_sm')

def get_relation(sent):
  doc = nlp(sent)
 # Matcher class object
  matcher = Matcher(nlp.vocab)
 #define the pattern
  pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}]
  matcher.add("matching_1", [pattern])
  # pattern = [{"LOWER": "hello"}, {"LOWER": "world"}]
  # matcher.add("HelloWorld", [pattern])
  matches = matcher(doc)
  k = len(matches) - 1
  span = doc[matches[k][1]:matches[k][2]]
  return span.text

corpus = load_dataset('wikicorpus', 'raw_en')
passage = corpus['train']['text'][0]
count = 1
for sentence in split_into_sentences(preprocess(passage)):
        print(count,':',sentence)
        # doc = nlp(sentence)
        print(get_relation(sentence))
        # for tok in doc:
        #     print(tok.text, "...", tok.dep_)
        count = 1 + count
        if count == 10:
            break
