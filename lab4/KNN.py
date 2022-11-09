import numpy as np
from distances import *

class KNearestNeighbors:
    def __init__(self, k, distance_metric="euclidean", weight="uniform"):
        distance_functions = {
            "euclidean": euclidean_distance,
            "cosine": cosine_distance,
            "manhattan": manhattan_distance
        }

        self.k = k
        self.distance_metric = distance_metric
        self.distance_function = distance_functions[distance_metric]
        self.weight = weight

    def fit(self, X, y):
        """
        Store the 'prior knowledge' of you model that will be used
        to predict new labels.
        :param X : input data points, ndarray, shape = (R,C).
        :param y : input labels, ndarray, shape = (R,).
        """
        self.X = X
        self.y = y

    def predict(self, df_to_predict):
        """Run the KNN classification on X.
        :param X: input data points, ndarray, shape = (N,C).
        :return: labels : ndarray, shape = (N,).
        """
        predict_result = np.empty(shape=(df_to_predict.shape[0], ), dtype="object")

        # Per ogni elemento da predire
        for i_to_pred, el_to_pred in enumerate(df_to_predict):
            distances = self._getDistances(el_to_pred)
            res = self._getResult(distances)

            predict_result[i_to_pred] = res

            #if not isinstance(res, type(self.y[0])):
                #print("Res:" , res)
                #print(type(self.y[0]))
                #raise Exception("Result differs from train set in type")


        return predict_result

    def _getDistances(self, predict_element):
        distances = []
        for i, e in enumerate(self.X):
            dist_per_el = [float("inf"), "None"]
            d = self.distance_function(e, predict_element)
            dist_per_el[0] = d
            dist_per_el[1] = self.y[i]
            distances.append(dist_per_el)

        return distances

    def _getResult(self, distances):
        sortedDistances = sorted(distances, key=lambda x: x[0])
        sortedTop = np.array(sortedDistances[:self.k])
        sortedTopResults = np.array([x[1] for x in sortedTop])

        if(self.weight == "uniform"):
            unique, frequency = np.unique(sortedTopResults, return_counts=True)
            item_votes = list(zip(unique, frequency))

        elif(self.weight == "distance"):
            item_unique = np.unique(sortedTopResults)
            item_votes_dict = {}
            for item in item_unique:
                item_votes_dict[item] = 0

            for n in sortedTop:

                item_votes_dict[n[1]] += 1/float(n[0])

            item_votes = list(item_votes_dict.items())


        max_voted = max(item_votes, key=lambda x: x[1])
        return max_voted[0]