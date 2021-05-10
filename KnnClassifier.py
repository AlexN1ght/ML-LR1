import math

class KNNClassifier(object):
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
    def fit(self, data, lables):
        self.data = data.to_numpy()
        self.lables = lables
    def predict_one(self, features):
        dist_vec = [0 for _ in range(len(self.data))]
        for k in range(len(self.data)):
            for i in range(len(self.data[k])):
                dist_vec[k] += pow(abs(self.data[k][i] - features[i]), 2)
            dist_vec[k] = math.sqrt(dist_vec[k])
        dist_vec, labl = zip(*sorted(zip(dist_vec, self.lables)))
        
        ans = {}
        for i in range(self.n_neighbors):
            if labl[i] not in ans:
                ans[labl[i]] = 1 / dist_vec[i]
            else:
                ans[labl[i]] += 1 / dist_vec[i]
        max_v = 0
        r_l = -1
        for k, v in ans.items():
            if v > max_v:
                max_v = v
                r_l = k
        return r_l
    def predict(self, features_set):
        ans = []
        for features in features_set:
            ans.append(self.predict_one(features))
        return ans
    def score(self,features_set, lables):
        prediction = self.predict(features_set.to_numpy())
        cnt = 0
        for i in range(len(lables)):
            if lables[i] == prediction[i]:
                cnt += 1
        return cnt / len(lables)
        
