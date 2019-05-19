import numpy as np
import os
import glob
import re
from numpy.linalg import norm
from pyvi import ViTokenizer # for tokenization 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from time import time
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pickle 

# CONSTANT
DIR_PATH="./Train_Full"
SPECIAL_CHARACTER_REGEX = "[^a-zắằẳẵặăấầẩẫậâáàãảạđếềểễệêéèẻẽẹíìỉĩịốồổỗộôớờởỡợơóòõỏọứừửữựưúùủũụýỳỷỹỵ_ ]"
STOPWORD_PATH='stopword.txt'

class DataLoader:
    def __init__(self, path):
        self.path = path
    def load_data(self):
        X = []
        y = []
        for dir_path in glob.glob(self.path+"/*"):
            count = 0
            for file_path in glob.glob(dir_path+"/*"):
                with open(file_path, encoding="utf-16", errors="ignore") as file:
                    content = file.read()
                    X.append(content)
                    count+=1
                    if count == 500:
                        break

        return X
    def loadStopword(self):
        with open(self.path) as file:
            content = file.read().strip()
            stopword = content.split('\n')
        
        return stopword

class NLP:
    '''
    -Tiền xử lý đoạn văn bản 
    '''
    def __init__(self):
        pass
    # Loai ky tu dac biet 
    def _remove_special_character(self, document):
        document = document.lower()
        document= re.sub(SPECIAL_CHARACTER_REGEX, " ", document)
        return document
    # Tach tu
    def _tokenization(self, document):
        document = ViTokenizer.tokenize(document)
        return document
    def _remove_stopword(self, document):
        stopword = DataLoader(STOPWORD_PATH).loadStopword()
        document = ' '.join([word for word in document.split(' ') if word not in stopword])
        return document
    
    def get_processed_document(self, pre_docment):
        processed_document = self._remove_special_character(pre_docment)
        processed_document = self._tokenization(processed_document)
        processed_document = self._remove_stopword(processed_document)
        return processed_document
    
class FeatureExtraction:
    def __init__(self):
        self.features=None
        self.pre_documents=None
        
    def _build_feature(self):
        
        # Tiền xử lý văn bản 
        corpus = [] # Tập văn bản đã xử lý 
        nlp = NLP()
        for doc in self.pre_documents:
            processed_doc = nlp.get_processed_document(doc)
            corpus.append(processed_doc)
        print('Ví dụ 1 tin tức sau khi tiền xử lý:')
        print(processed_doc[0])
        # Chuyển đổi văn bản sang tf-idf vector 
        self.tf = TfidfVectorizer(min_df=5, max_df=0.7)  # BỎ từ xuất hiện trong ít hơn 5 văn bản, và nhiều hơn 70% văn bản 
        self.features = self.tf.fit_transform(corpus)
            
    def get_feature(self, pre_documents):
        t0 = time()
        self.pre_documents = pre_documents
        self._build_feature()
        print(f'Trích xuất đặc trưng thành công. Time={time()-t0}s')
        return self.features, self.tf
    
class Kmeans:
    '''Implementing Kmeans algorithm.'''

    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initializ_centroids(self, X):
        '''
        - Chọn ngẫu nhiên n_clusters vector trong X làm centroids
        - Output: centroids shape (n_cluster, n_dictionary)
        '''
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        '''
        - Tính lại centroids cho mỗi cụm 
        - Input: X, labels (index của các cụm: từ 0->n_cluster-1)
        - Output: centroids mới 
        '''
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids
    
    def assign_labels(self, X, centroids):
        '''
        - Gán label cho các vector có độ tương đồng với centroid  nhất (hay khoảng cách nhỏ nhất)
        '''
        # Tính khoảng cách giữa X và centroids
        D = cdist(X, centroids)
        # Trả về label centroid gần nhất 
        return np.argmin(D, axis = 1)
    
    # Error = tổng bình phương khoảng cách từ các điểm đến centroid của nó 
    def compute_sse(self, X, labels, centroids):
        '''
        - Hàm lỗi SSE
        '''
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def fit(self, X):
        '''
        - Hàm chính phân cụm Kmeans
        + Cập nhật center trong max_iter vòng lặp hoặc đến khi 2 tập centroid không thay đổi 
        '''
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            self.labels = self.assign_labels(X, self.centroids)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)
    
    def predict(self, X):
        return self.assign_labels(X, self.centroids)
    
class Cluster:
    def __init__(self, features=None, estimator=None):
        self.features = features
        self.estimator = estimator
    def clustering(self, features):
        self.estimator.fit(features)
        return self.estimator
    
# Load data 
news = DataLoader(DIR_PATH).load_data()
print(f"Tổng số tin tức: {len(news)}")
print(f'Tổng số stopword: {len(DataLoader(STOPWORD_PATH).loadStopword())}')

# Trích xuất đặc trưng, kết quả là file features.pickle 
# features, tfidf = FeatureExtraction().get_feature(news)
# with open('features.pickle', 'wb') as obj:
#     pickle.dump(features, obj)
#with open('tfidf.pickle', 'wb') as obj:
#    pickle.dump(tfidf, obj)
    
with open('features.pickle', 'rb') as obj:
    features = pickle.load(obj)
X = features.toarray()  # Array các vector tf-idf của các văn bản 

# Phân cụm 
n_clusters=14
t0=time()
estimator = Kmeans(n_clusters=n_clusters)
models = Cluster(estimator=estimator).clustering(X)
print(f'Phân cụm thành công trong {time()-t0}s')

# Kết quả 
labels = models.predict(X)

# Top word
print("Các từ đại diện trong mỗi cụm :")
# Sắp xếp 
order_centroids = estimator.centroids.argsort()[:, ::-1]
with open('tfidf.pickle', 'rb') as obj:
    tfidf = pickle.load(obj)
terms = tfidf.get_feature_names()
for i in range(n_clusters):
    print("Cụm %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print('%s--' % terms[ind], end='')
    print()

