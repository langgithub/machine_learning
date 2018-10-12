import gensim

model=gensim.models.Word2Vec.load("wiki.zh.text.model")

result=model.most_similar("足球")

print(result)