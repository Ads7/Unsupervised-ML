# todo index 20 news group and DUC 2001 summarization
# Index each dataset separetely in Elastic Search (one index for each dataset). First set up the indexes/types/fields
# in Kibana, then use an API to send all docs to the index. At the minimum you will need two fields:
# "doc_id", and "doc_text"; you can add other fields. For DUC dataset add a field "gold_summary".
import os

from datetime import datetime

from elasticsearch.helpers import parallel_bulk, streaming_bulk
import elasticsearch
from pyexpat import ExpatError
from sklearn.datasets import fetch_20newsgroups

from HW2 import DATA_DIR

es = elasticsearch.Elasticsearch(http_auth=('elastic', 'elastic'))


def index(path):
    newsgroups = fetch_20newsgroups(data_home=DATA_DIR, remove=('headers', 'footers', 'quotes'),subset='all')

    def get_formated_data(i):
        structured_json_body = {
            '_op_type': 'index',
            '_index': 'news-group',
            '_type': 'document',
            '_id': newsgroups.filenames[i].split('/')[-1],
            '_source': {'doc_id': newsgroups.filenames[i].split('/')[-1], 'doc_text': newsgroups.data[i],
                    'label': newsgroups.target[i]}
        }
        return structured_json_body

    for (res, i) in streaming_bulk(client=es, raise_on_exception=True, raise_on_error=True,
                                  actions=[get_formated_data(i) for i in
                                           range(len(newsgroups.target))]):
        print(res, i)

    # es.index(index='news-group',doc_type='document', body={'doc_id': newsgroups.filenames[0].split('/')[-1], 'doc_text': newsgroups.data[0],
    #                                    'label': newsgroups.target_names[0]},
    #          id=newsgroups.filenames[0].split('/')[-1])
    # for (dir_path, dir_names, file_names) in os.walk(path):
    #     structured_json_body = ({
    #         '_op_type': 'index',
    #         '_index': 'news-group',  # index name Twitter
    #         '_type': 'docs',  # type is tweet
    #         '_id': doc['tweet_id'],  # id of the tweet
    #         '_source': doc
    #     } for doc in self.__generate_field(body))
    # if not dir_names:
    #     labels = dir_path[len(path) + 1:].split('.')
    #     elasticsearch.helpers.parallel_bulk()
    #     for doc in file_names:
    #         print(doc)
    #         es.index(index="news-group", doc_type="news", id=42,
    #                  body={"any": "data", "timestamp": datetime.now()})


if __name__ == '__main__':
    index(path='/Users/aman/workspace/DataMining/20news_home/20news-bydate-test')
