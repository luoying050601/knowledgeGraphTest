# 句子分割Sentence Segmentation

from datasets import load_dataset
from common_util import split_into_sentences,preprocess


def get_entities(sent):
    entity_pair = []
    ## chunk 1
    # 我在这个块中定义了一些空变量。
    # prv tok dep和prv tok text将分别保留句子中前一个单词和前一个单词本身的依赖标签。
    # 前缀和修饰符将保存与主题或对象相关的文本。
    ent1 = ""
    ent2 = ""
    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence
    prefix = ""
    modifier = ""
    #############################################################
    for tok in nlp(sent):
        ## chunk 2
        # 接下来，我们将遍历句子中的记号。我们将首先检查标记是否为标点符号。
        # 如果是，那么我们将忽略它并转移到下一个令牌。
        # 如果标记是复合单词的一部分(dependency tag = compound)，我们将把它保存在prefix变量中。
        # 复合词是由多个单词组成一个具有新含义的单词(例如“Football Stadium”, “animal lover”)。
        # 当我们在句子中遇到主语或宾语时，我们会加上这个前缀。我们将对修饰语做同样的事情，例如“nice shirt”, “big house”
        # if token is a punctuation mark then move on to the next token
        dep_ = tok.dep_
        if dep_ != "punct":
            # check: token is a compound word or not
            if dep_ == "compound":
                prefix = tok.text
            # if the previous word was also a 'compound' then add the current word to it
            if prv_tok_dep == "compound":
                prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if dep_.endswith("mod"):
                modifier = tok.text
            # if the previous word was also a 'compound' then add the current word to it
            if prv_tok_dep == "compound":
                modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            # 在这里，如果令牌是主语，那么它将作为ent1变量中的第一个实体被捕获。变量如前缀，修饰符，prv tok dep，和prv tok文本将被重置。
            if tok.dep_.find("subj") == 1:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            # 在这里，如果令牌是宾语，那么它将被捕获为ent2变量中的第二个实体。变量，如前缀，修饰符，prv tok dep，和prv tok文本将再次被重置。
            if dep_.find("obj") == 1:
                ent2 = modifier + " " + prefix + " " + tok.text
            ## chunk 5
            # 一旦我们捕获了句子中的主语和宾语，我们将更新前面的标记和它的依赖标记。
            # update variables
                prv_tok_dep = dep_
                prv_tok_text = tok.text
            if ent1 != "" and ent2 != "":
                    entity_pair.append([ent1.strip(), ent2.strip()])
                    ent1 = ""
                    ent2 = ""


    #############################################################
    return entity_pair


import spacy
nlp = spacy.load('en_core_web_sm')

corpus = load_dataset('wikicorpus', 'raw_en')
passage = corpus['train']['text'][0]
    # print(passage)
count = 1
for sentence in split_into_sentences(preprocess(passage)):
        print(count,':',sentence)
        # doc = nlp(sentence)
        print(get_entities(sentence))
        # for tok in doc:
        #     print(tok.text, "...", tok.dep_)
        count = 1 + count
        if count == 4:
            break


