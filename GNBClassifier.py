from math import pow
import math

def P_xi_y(x, mu, sig):
    return 1 / math.sqrt(2 * math.pi * pow(sig, 2)) * math.exp(-pow(x - mu, 2) / (2 * pow(sig, 2)))


class GNBClassifier(object):
    def __init__ (self):
        return
    def fit(self, data, lables):
        self.data = data.to_numpy()
        self.lables = lables
        self.ProbForC = {}
        for lable in self.lables:
            if lable not in self.ProbForC:
                self.ProbForC[lable] = {'classP':1, 'features':[{'mu': 0, 'sig':1} for _ in range(len(self.data[0]))]} 
            else:
                self.ProbForC[lable]['classP'] += 1
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                self.ProbForC[lables[i]]['features'][j]['mu'] += self.data[i][j]
        for lable in self.ProbForC:
            for j in range(len(self.data[0])):
                self.ProbForC[lable]['features'][j]['mu'] /= self.ProbForC[lable]['classP']
        
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                self.ProbForC[lables[i]]['features'][j]['sig'] += pow((self.data[i][j] - self.ProbForC[lables[i]]['features'][j]['mu']), 2)
        for lable in self.ProbForC:
            for j in range(len(self.data[0])):
                self.ProbForC[lable]['features'][j]['sig'] /= self.ProbForC[lable]['classP']
            self.ProbForC[lable]['classP'] /= len(self.lables)

    def predict_one(self, features):
        dist_vec = [0 for _ in range(len(self.data))]
        for k in range(len(self.data)):
            for i in range(len(self.data[k])):
                dist_vec[k] += pow(abs(self.data[k][i] - features[i]), 2)
            dist_vec[k] = math.sqrt(dist_vec[k])
        dist_vec, labl = zip(*sorted(zip(dist_vec, self.lables)))
        
        ans = {}
        for lable in self.ProbForC:
            ans[lable] = self.ProbForC[lable]['classP']
            for j in range(len(self.data[0])):
                ans[lable] *= P_xi_y(features[j], self.ProbForC[lable]['features'][j]['mu'], self.ProbForC[lable]['features'][j]['sig'])

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
        
