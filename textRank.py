import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence


text = codecs.open('Pre_Processed_reviews.txt', 'r', 'utf-8').read()
tr4w = TextRank4Keyword()
 
# py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
tr4w.analyze(text=text, lower=True, window=2)  

print()
print( 'key frase:' )
for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num= 2):
    print(phrase)
 
tr4s = TextRank4Sentence()
tr4s.analyze(text=text, lower=True, source = 'all_filters')
 
print()
print( 'substract' )
for item in tr4s.get_key_sentences(num=3):
    print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重