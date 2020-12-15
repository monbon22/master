from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import textract
import datetime
from pymongo import MongoClient
import json
from core.classifier import knnClassify


# ToDo Logging, DB


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# Think about different classifiers


@app.route('/classify')
def classify():
    # ToDo Parse pdf raw data and classify it
    return render_template('classify.html')


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    fname = f"{(datetime.datetime.now() - datetime.datetime(1970, 1, 1)).total_seconds()}.pdf"
    f.save(secure_filename(fname))
    return redirect(url_for('success', fname=fname))


@app.route('/success')
def success():
    fname = request.args['fname']
    text = textract.process(fname, encoding='unicode_escape').decode('utf-8', 'ignore')
    client = MongoClient('mongodb://localhost:27017/')
    db = client.uir
    texts = list(db.vectors.find({}))
    cls = knnClassify(text, texts, 20)
    with open('data/clusters.json', 'r') as f:
        s = f.readline()
        titles = json.loads(s)
    return render_template('success.html', cls=titles[cls])


@app.route('/statistics')
def statistics():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.uir
    data = []
    titles = None
    with open('data/clusters.json', 'r') as f:
        s = f.readline()
        titles = json.loads(s)
    for i in range(6):
        pks = []
        cs = db.vectors.find({'class': i}, {'vector': 0})
        for item in cs:
            pks.append(item['pk'])
        cluster_data = []
        cs = db.origins.find({'pk': {"$in": pks}}, {'text': 0})
        for item in cs:
            cluster_data.append({
                'title': item['title'],
                'author': item['author'],
                'university': item['university'],
            })
        print(titles[i])
        data.append({
            'size': len(cluster_data),
            'items': cluster_data,
            'title': titles[i],
        })
    # What category how much articles has
    return render_template('statistics.html', file_content=data)


if __name__ == '__main__':
    app.run(debug=True)
