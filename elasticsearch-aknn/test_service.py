#
# Code to test aknn plugin
#


#
# Add hash table
#


import numpy as np
import requests
import random



LSHMODEL = 'lsh_default' # LSH model name
NHASH = 4               # number of hash tables of the model
BITS = 2                # number of bits for LSH model
AKNN_URI = "aknn_models/aknn_model/{}".format(LSHMODEL) # ES url to default LSH model
ES_URL = 'http://localhost:9200'
EDIM = 8

def init_LSH(name, nhash, bits, dim):
    '''
    Function to generate hash tables and load on Elasticsearch
    :param name: name of LSH model
    :param nhash: number of hash tables
    :param bits: number of bits for each table
    :param dim: dimensionality of vectors to be hashed
    '''
    # Create vectors necessary for LSH model and loads into elasticsearch
    np.random.seed(nhash * bits * dim) # set random seed so that for given size of model the hash tables are the same
    hashvec = np.random.rand(2 * nhash * bits, dim) # generate the hash tables
    model_description = 'LSH model for semantic search.\nembedding dim = {}\n# hash tables = {}\n# bits{}'.format(dim, nhash, bits)
    # create body to add tables to Elasticsearch
    body_model = {
        "_index":   "aknn_models",
        "_type":    "aknn_model",
        "_id":      name,
        "_source": {
            "_aknn_description": model_description,
            "_aknn_nb_tables": nhash,
            "_aknn_nb_bits_per_table": bits,
            "_aknn_nb_dimensions": dim
        },
        "_aknn_vector_sample": hashvec.tolist()
    }
    # Create document for model on elastic
    url = '{}/_aknn_create'.format(ES_URL)
    client = requests.Session()
    client.headers = 'application/json'
    r = client.post(url, json = body_model)
    client.close()

init_LSH(name = LSHMODEL, nhash = NHASH, bits = BITS, dim = EDIM)

#
# Add documents
#


def body_add_test(indexname, doctype, aknnuri, docid, vec, name, surname):
    '''
    Returns body to be POSTED to elasticsearch when adding single item to index. The format is
    body = {"_index": indexname,
                "_type": doctype,
                "_aknn_uri": aknnuri,
                "_aknn_docs": [{"_id": str(docid),
                                "_source": {"_aknn_vector": vec,
                                            "name": name,
                                            "surname": surname,
                                            [args in kwargs]}
                                }]
                }
    '''
    # build body
    source = {}
    source["_name"] = name
    source["_surname"] = surname
    source["_aknn_vector"] = vec
    body = {"_index": indexname,
                "_type": doctype,
                "_aknn_uri": aknnuri,
                "_aknn_docs": [{"_id": str(docid),
                                "_source": source
                                }]
                }
    return body

def add_item_to_index(indexname, name, surname, itemid):
    '''
    Compute vector and add item to es index
    '''
    nvec = np.random.rand(EDIM)
    # add file to es index
    body = body_add_test(indexname = indexname,
                         doctype = "doc",
                         aknnuri = AKNN_URI,
                         docid = itemid,
                         vec = nvec.tolist(),
                         name = name,
                         surname = surname)
    url = '{}/_aknn_index'.format(ES_URL)
    requests.post(url, json = body)

names = ['Andrea', 'Carlo', 'Pietro', 'Giulia']
surnames = ['Bianchi', 'Rossi', 'Verdi', 'Dutto']

for id in range(100):
    add_item_to_index('testindex', random.choice(names), random.choice(surnames), str(id))

