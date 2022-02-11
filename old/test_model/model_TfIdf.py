from underthesea import word_tokenize

docA = "có bao nhiêu phương thức xét tuyển"
docB = "điểm chuẩn là bao nhiêu là"
docA = word_tokenize(docA, format="text")
docB = word_tokenize(docB, format="text")

wordsA = docA.split()
wordsB = docB.split()




wordSet = set(wordsA).union(set(wordsB))
print(wordSet)


wordDictA = dict.fromkeys(wordSet, 0) 
wordDictB = dict.fromkeys(wordSet, 0)

#print(wordDictA)

for word in wordsA:
    wordDictA[word]+=1
    
for word in wordsB:
    wordDictB[word]+=1

print(wordDictA)
print(wordDictB)


# def computeTF(wordDict, words):
#     tfDict = {}
#     wordsCount = len(words)
#     for word, count in wordDict.items():
#         tfDict[word] = count/float(wordsCount)
#     return tfDict


# tfdocA = computeTF(wordDictA, wordsA)
# tfdocB = computeTF(wordDictB, wordsB)


# print(tfdocA)
# print(tfdocB)

# def computeIDF(docList):
#     import math
#     idfDict = {}
#     N = len(docList)
    
#     idfDict = dict.fromkeys(docList[0].keys(), 0)
#     for doc in docList:
#         for word, val in doc.items():
#             if val > 0:
#                 idfDict[word] += 1
    
#     for word, val in idfDict.items():
#         idfDict[word] = math.log10(N / float(val))
        
#     return idfDict


# idfs = computeIDF([wordDictA, wordDictB])

# print(idfs)

# def computeTFIDF(tfDocs, idfs):
#     tfidf = {}
#     for word, val in tfDocs.items():
#         tfidf[word] = val*idfs[word]
#     return tfidf

# tfidfDocA = computeTFIDF(tfdocA, idfs)
# tfidfDocB = computeTFIDF(tfdocB, idfs)

# print(tfidfDocA)
# print(tfidfDocB)