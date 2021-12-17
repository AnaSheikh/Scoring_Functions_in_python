import snowballstemmer as snowballstemmer
from collections import OrderedDict
import math

from Tools.scripts.treesync import raw_input
from bs4 import BeautifulSoup
import re
import glob
import operator


# class for storing number of terms of each document
class document(object):
    def __init__(self, num):
        self.no_of_terms = num


# class for storing number of terms and text of each query
class query(object):
    def __init__(self):
        self.text = []
        self.no_of_terms = 0


# document vector for scoring
class document_vector(object):
    def __init__(self):
        self.values = []


# query vector for scoring
class query_vector(object):
    def __init__(self):
        self.values = []


# class used for storing info of term
class Termmm(object):
    def __init__(self, id, offset, no_of_words, no_of_documents):
        self.id = id
        self.offset = offset
        self.no_of_words = no_of_words
        self.no_of_documents = no_of_documents


# class to store score of a document
class score(object):
    def __init__(self, scoree, doc):
        self.scoree = scoree
        self._doc = doc


# class to keep all query scores with each document
class query_scores(object):
    def __init__(self, id):
        self.id = id
        self.scores = {}


class tf_d_(object):
    def __init__(self, val):
        self.tf = val


class terms_tf(object):
    def __init__(self, t):
        self.name = t
        self.tfs = {}


docs = {}
queries = {}
termid = {}
docid = {}
terms = {}
tf_d_results = {}
query_results = {}
term_index = open("D:\LastSemester\IR\\ass2\\term_index.txt", 'r', encoding='utf-8')
# storing term info in terms hashmap
term_info = open("D:\LastSemester\IR\\ass2\\term_info.txt", 'r', encoding='utf-8')
data = term_info.read()
data1 = data.split("\n")
for line in data1:
    data2 = line.split("\t")
    if len(data2) == 4:
        terms[data2[0]] = Termmm(data2[0], data2[1], data2[2], data2[3])
# reading forward index to get number of terms of each document and average length of docuemnt
forward_index = open("D:\LastSemester\IR\\ass2\docindex.txt", 'r', encoding='utf-8')
forward_index_data = forward_index.read()
lines = forward_index_data.split('\n')
average_doc_length = 0
for line in lines:
    if line != '':
        doc_data = line.split('\t')
        if doc_data[0] not in docs:
            docs[doc_data[0]] = document(len(doc_data) - 2)
        else:
            docs[doc_data[0]].no_of_terms += (len(doc_data) - 2)
        average_doc_length += (len(doc_data) - 2)
total_doc_length = average_doc_length
average_doc_length /= len(docs)
stemmer = snowballstemmer.stemmer('english')
# loading stop words in stop_words_data
stop_words = open("D:\LastSemester\IR\stoplist.txt", 'r', encoding='utf-8')
stop_words_data = stop_words.read()
# opening file that has queries
handler = open("topics.xml").read()
soup = BeautifulSoup(handler, "xml")
average_query_length = 0
# parsing the file and storing its text and number of terms in queries hashmap
for topic in soup.find_all('topic'):
    queryy = topic.find("query")
    tokenized_query = re.split('\\W+(\\.?\\W+)*', queryy.text)
    queryid = topic["number"]
    for token in tokenized_query:
        if token is None:
            continue
        if stop_words_data.find(token) == -1:
            token = token.lower()
            word = stemmer.stemWord(token)
            if queryid not in queries:
                queries[queryid] = query()
            queries[queryid].text.append(word)
            queries[queryid].no_of_terms += 1
            average_query_length += 1
average_query_length /= 10
# loading termids in termid hashmap
termids = open("D:\LastSemester\IR\\ass2\\termids.txt", 'r', encoding='utf-8')
termids1 = termids.read()
termids2 = termids1.split('\n')
for line in termids2:
    if line != '':
        termids3 = line.split('\t')
        termid[termids3[1]] = termids3[0]
# loading docids in docid hashmap
docids = open("D:\LastSemester\IR\\ass2\docids.txt", 'r', encoding='utf-8')
docids1 = docids.read()
docids2 = docids1.split('\n')
for line in docids2:
    if line != '':
        docids3 = line.split('\t')
        docid[int(docids3[0])] = docids3[1]
temp = OrderedDict(sorted(docid.items()))
docid = temp
# getting input of method
method = raw_input("Enter Method to score:")
method = method.rstrip("\n")
queries_done = 0
for q in queries:
    for term in queries[q].text:
        if term not in termid:
            tf_d_results[term] = terms_tf(term)
            for d in docs:
                tf_d_results[term].tfs[d] = tf_d_(0)
        else:
            current_doc = 1
            tf_d_results[term] = terms_tf(term)
            index = 1
            temp_doc = 0
            term_index.seek(int(terms[termid[term]].offset))
            data = term_index.readline()
            data1 = data.split('\t')
            for d in docid:
                frequency = 0
                ind = -1
                ind1 = -1
                for i in range(index, len(data1), 1):
                    if data1[i] == "\n":
                        tf_d_results[term].tfs[d] = tf_d_(frequency)
                        break
                    for j in range(0, len(data1[i]), 1):
                        if data1[i][j] == ':':
                            ind = j
                            break
                    if temp_doc > int(d):
                        tf_d_results[term].tfs[d] = tf_d_(frequency)
                        index = i
                        break
                    temp_doc = temp_doc + int(data1[i][0:ind])
                    if temp_doc == int(d):
                        frequency += 1
        print("Term")

    print("Query")

if method == "tf_idf":
    # writing results in the file
    out = open("D:\LastSemester\IR\\ass2\\tf_idf.txt", 'w+')
    for q in queries:
        for d in docs:
            queri = query_vector()
            doc = document_vector()
            for term in queries[q].text:
                t = termid[term]
                # processing for query vector calculating tf_idf
                tf_idf_q = 0
                for query_word in queries[q].text:
                    if term == query_word:
                        tf_idf_q += 1
                term_frequency_q = queries[q].no_of_terms / average_query_length
                occurrences_query = 0
                for q1 in queries:
                    for query_word1 in queries[q1].text:
                        if term == query_word1:
                            occurrences_query += 1
                            break
                term_frequency_q = term_frequency_q * math.log((10 / occurrences_query), 2)
                queri.values.append(term_frequency_q)
                # processing for document vector calculating tf_idf
                if term not in termid:
                    tf_idf_d = 0
                else:
                    tf_idf_d = tf_d_results[term].tfs[int(d)].tf
                term_frequency_d = docs[d].no_of_terms / average_doc_length
                term_frequency_d = term_frequency_d * math.log((3463 / int(terms[t].no_of_documents)), 2)
                doc.values.append(term_frequency_d)
            # cosing similarity of two vectors
            result = sum(i[0] * i[1] for i in zip(doc.values, queri.values))
            magnitude_d = math.sqrt(sum(i ** 2 for i in doc.values))
            magnitude_q = math.sqrt(sum(i ** 2 for i in queri.values))
            # to avoid math error if magnitude of document vector is 0 then result is 0
            if magnitude_d == 0:
                result = 0
            else:
                result /= (magnitude_d * magnitude_q)
            # adding query scores in query_results.scores
            if q not in query_results:
                query_results[q] = query_scores(q)
            query_results[q].scores[d] = score(result, docid[int(d)])
            queries_done += 1
            print (queries_done)
        # sorting scores in descending order
        temp = OrderedDict(sorted(query_results[q].scores.items(), key=lambda x: x[1].scoree, reverse=True))
        query_results[q].scores = temp
        rank = 1
        for s in query_results[q].scores:
            out.write(q)
            out.write(" 0 ")
            out.write(query_results[q].scores[s]._doc)
            out.write(" ")
            out.write(str(rank))
            rank += 1
            out.write(" ")
            out.write(str(query_results[q].scores[s].scoree))
            out.write(" run1\n")
        out.write("\n")

if method == "bm25":
    # writing results in the file
    out = open("BM25.txt", 'w')
    for q in queries:
        for d in docs:
            doc = document_vector()
            for term in queries[q].text:
                t = termid[term]
                # processing for query vector calculating tf
                tf_q = 0
                for query_word in queries[q].text:
                    if term == query_word:
                        tf_q += 1
                # processing for document vector calculating tf
                if term not in termid:
                    tf_d = 0
                else:
                    tf_d = tf_d_results[term].tfs[int(d)].tf
                k = 1.2 * ((1 - 0.75) + 0.75 * (docs[d].no_of_terms / average_doc_length))
                k2 = 100
                doc.values.append((math.log((3463 + 0.5) / (int(terms[t].no_of_documents) + 0.5))) * (
                            ((1 + 1.2) * tf_d) / (k + tf_d)) * (((1 + k2) * tf_q) / (k2 + tf_q)))
            # calculating score
            result = sum(val for val in doc.values)
            # adding query scores in query_results.scores
            if q not in query_results:
                query_results[q] = query_scores(q)
            query_results[q].scores[d] = score(result, docid[int(d)])
        # sorting scores in descending order
        temp = OrderedDict(sorted(query_results[q].scores.items(), key=lambda x: x[1].scoree, reverse=True))
        query_results[q].scores = temp
        rank = 1
        for s in query_results[q].scores:
            out.write(q)
            out.write(" 0 ")
            out.write(query_results[q].scores[s]._doc)
            out.write(" ")
            out.write(str(rank))
            rank += 1
            out.write(" ")
            out.write(str(query_results[q].scores[s].scoree))
            out.write(" run1\n")
        out.write("\n")
if method == "jm":
    for q in queries:
        for d in docs:
            doc = document_vector()
            for term in queries[q].text:
                t = termid[term]
                # processing for document vector calculating tf
                tf_d = tf_d_results[term].tfs[int(d)].tf
                tf_c = int(terms[t].no_of_words)
                doc.values.append(
                    (0.6 * (float(tf_d) / docs[d].no_of_terms)) + ((0.4) * (float(tf_c) / total_doc_length)))
            # calculating score
            result = 0
            for val in doc.values:
                if val != 0:
                    result += math.log(val, 2)
            # adding query scores in query_results.scores
            if q not in query_results:
                query_results[q] = query_scores(q)
            query_results[q].scores[d] = score(result, docid[int(d)])
    # writing results in the file
    out = open("jm.txt", 'w')
    for q in query_results:
        # sorting scores in descending order
        temp = OrderedDict(sorted(query_results[q].scores.items(), key=lambda x: x[1].scoree, reverse=True))
        query_results[q].scores = temp
        rank = 1
        for s in query_results[q].scores:
            out.write(q)
            out.write(" 0 ")
            out.write(query_results[q].scores[s]._doc)
            out.write(" ")
            out.write(str(rank))
            rank += 1
            out.write(" ")
            out.write(str(query_results[q].scores[s].scoree))
            out.write(" run1\n")
        out.write("\n")
