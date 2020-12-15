from os import listdir
from os.path import join, isfile
import textract
import re
from nltk.corpus import stopwords
from pymongo import MongoClient
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk.stem import *
import numpy as np
import random
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as nrm
from google.cloud import bigquery
import json

client = MongoClient('mongodb://localhost:27017/')
db = client.uir
stemmer = SnowballStemmer('english')

basis = {}
sw = set(stopwords.words('english'))


def extract_pattern(arr):
    if arr:
        return re.sub(r'\\.*', '', arr[0])
    return ''


texts = []
pk = 1
for i in range(6):
    filenames = [join(f'core/scripts/{i}', f) for f in listdir(f'core/scripts/{i}') if isfile(join(f'core/scripts/{i}', f))]
    for filename in filenames:
        try:
            text = textract.process(filename, encoding='unicode_escape').decode('utf-8', 'ignore')
        except Exception as e:
            print(e)
            print(f'{filename} failed!')
            continue
        original = text
        title = extract_pattern(re.findall(r'([\w\s\-,\.]+)\\', text))
        author_email = extract_pattern(re.findall(r'\S+\@\S+', text))
        university = extract_pattern(re.findall(r'([^\\\d,\.]*university[^\\\d,\.]*)\\', text, re.IGNORECASE))
        text = re.sub(r'-\n', '', text.lower())
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'[,.]', '', text)
        text = re.sub(r'\s+', ' ', text)
        arr = word_tokenize(text)
        arr = [stemmer.stem(word) for word in arr]
        arr = list(filter(lambda x: x not in sw, arr))
        arr = list(filter(lambda x: not re.match(r'\d+$', x), arr))
        arr = list(filter(lambda x: re.match(r'\w+$', x), arr))
        arr = list(filter(lambda x: len(x) > 3, arr))
        arr = list(filter(lambda x: len(x) < 15, arr))
        text_vector = {}
        for w in arr:
            if w not in basis:
                basis[w] = 1
            else:
                basis[w] += 1
            if w not in text_vector.keys():
                text_vector[w] = 1
            else:
                text_vector[w] += 1
        texts.append({
            'class': i,
            'words': text_vector,
            'pk': pk,
        })
        db.origins.insert_one({
            'title': title,
            'text': original,
            'author': author_email,
            'university': university,
            'pk': pk,
        })
        print(f'{filename} processed!')
        pk += 1

frequences = {}
for val in basis.values():
    freq = 0
    for x in basis.values():
        if x == val:
            freq += 1
    if val not in frequences.keys():
        frequences[val] = freq
    else:
        frequences[val] += freq

new_basis = {x: basis[x] for x in list(filter(lambda k: basis[k] >= 100, basis.keys()))}
basis_words = sorted(list(new_basis.keys()))

tf_idfs = []
counted = []
for text in texts:
    to_save = []
    for w in basis_words:
        dt = 0
        size = sum(text['words'].values())
        for d in texts:
            if w in d['words'].keys():
                dt += 1
        if w in text['words'].keys():
            to_save.append(text['words'][w]/size * np.log10(1 + len(texts)/dt))
        else:
            to_save.append(0)
    tf_idfs.append(to_save)
    counted.append({
        'class': text['class'],
        'vector': to_save,
    })

tfidf_word = [[basis_words[i], sum(x[i] for x in tf_idfs)] for i in range(len(basis_words))]
tfidf_word.sort(key=lambda x: x[1])

for item in tfidf_word:
    if item[1] < 0.3:
        basis_words.remove(item[0])
        tfidf_word.remove(item)

pk = 1
for text in texts:
    to_save = []
    for w in basis_words:
        dt = 0
        size = sum(text['words'].values())
        for d in texts:
            if w in d['words'].keys():
                dt += 1
        if w in text['words'].keys():
            to_save.append(text['words'][w]/size * np.log10(1+ len(texts)/dt))
        else:
            to_save.append(0)
    db.vectors.insert_one({
        'class': text['class'],
        'vector': to_save,
        'pk': pk
    })
    pk += 1


def distance(vec1, vec2):
    return sum([(vec1[i] - vec2[i]) ** 2 for i in range(len(vec1))]) ** 0.5


def diff(l1, l2):
    if len(l1) != len(l2):
        raise IndexError
    return sum([distance(l1[i], l2[i]) for i in range(len(l1))]) / len(l1)


# k_means
n = len(basis_words)
k = 6
src_data = db.vectors.find({})
data = [{'vector': x['vector'], 'class': None, 'etalon': x['class']} for x in src_data]
X = [x['vector'] for x in data]
pca = PCA(n_components=500)
X = nrm(X)
reducer = umap.UMAP(n_neighbors=5, metric='cosine', min_dist=0.05, n_components=2)
pca_graph = PCA(n_components=2)

steps = 0
d = 10e6
centers_dumped = [[0 for i in range(n)] for j in range(k)]
centers = [random.choice(data)['vector'] for i in range(k)]
centers_emb = pca_graph.fit_transform(centers)
while d > 1e-10:
    for vec in data:
        a = vec['vector']
        dist = [distance(a, centers[j]) for j in range(k)]
        vec['class'] = dist.index(min(dist))

    X_embedded = pca_graph.fit_transform(X)
    xs = [x[0] for x in X_embedded]
    ys = [x[1] for x in X_embedded]
    plt.scatter(xs, ys, c=[x['class'] for x in data])
    plt.show()

    centers_dumped = centers.copy()
    for i in range(k):
        i_class = list(filter(lambda x: x['class'] == i, data))
        if len(i_class) == 0:
            steps += 1
            continue
        buf = [0 for i in range(n)]
        for vec in i_class:
            buf = [buf[i] + vec['vector'][i] for i in range(n)]
        buf = [x / len(i_class) for x in buf]
        centers[i] = buf
    steps += 1

    d = diff(centers_dumped, centers)


# vectors to google big query
client = bigquery.Client.from_service_account_json('data/uir2019-3591fde89074.json')
project_id = 'uir2019'

src_data = db.vectors.find({})

i = 1

for item in src_data:
    cls = item['class']
    vector = json.dumps(item['vector'])
    pk = item['pk']

    sql = f'''
        INSERT INTO
      `uir2019.data.articles` (class, vector, pk)
     VALUES ({cls}, '{vector}', {pk})
    '''

    query_job = client.query(sql)
    print(i)
    i += 1
results = query_job.result()


# origins to google big query
client = bigquery.Client.from_service_account_json('data/uir2019-3591fde89074.json')
project_id = 'uir2019'

src_data = db.origins.find({})
src_data = src_data[:699]

i = 1

for item in src_data:
    title = item['title']
    text = item['text']
    author = item['author']
    university = item['university']
    pk = item['pk']

    sql = f'''
        INSERT INTO
      `uir2019.data.origins` (title, text, author, university)
     VALUES ('{title}', '{text}', '{author}', '{university}', {pk})
    '''

    query_job = client.query(sql)
    print(i)
    i += 1
results = query_job.result()

to_dump = []


# Dumping cluster names
for i in range(6):
    cs = db.vectors.find({'class': i}, {'vector': 1})
    data = [x['vector'] for x in cs]
    valuebles = [sum(x[i] for x in data) for i in range(len(basis_words))]
    m1, m2, m3 = 0, 0, 0
    i1, i2, i3 = 0, 0, 0
    for j in range(len(basis_words)):
        if valuebles[j] > m1:
            m1 = valuebles[j]
            i1 = j
        elif valuebles[j] > m2:
            m2 = valuebles[j]
            i2 = j
        elif valuebles[j] > m3:
            m3 = valuebles[j]
            i3 = j
    to_dump.append(' '.join([basis_words[i1], basis_words[i2], basis_words[i3]]))

with open('data/clusters.json', 'w') as f:
    f.write(json.dumps(to_dump))