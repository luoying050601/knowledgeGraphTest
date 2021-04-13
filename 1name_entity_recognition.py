from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from datasets import load_dataset
from common_util import split_into_sentences,preprocess

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
corpus = load_dataset('wikicorpus', 'raw_en')
passage = corpus['train']['text'][0]
count = 1
for sentence in split_into_sentences(preprocess(passage)):
        print(count,':',sentence)
        ner_results = nlp(sentence)
        print(ner_results)
        count = 1 + count
        if count == 10:
            break
