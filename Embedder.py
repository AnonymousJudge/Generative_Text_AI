# word2vec principle 


# 3 Layers (input hidden output)
# input layer n nodes wher n is the length of the encoding
# hidden layer m nodes where n is the vector length
# output layer n nodes where n is the lengt of the encoding (gives the porbablitiy of the word to follow up next)  

from NeuralNetwork import NeuralNetwork
import numpy

class Embedder: 

    neural_network: NeuralNetwork
    input_token: int

    def __init__(self, input_token: int, nn_weigths_path:str = "./data/embedder/weigths.json"):
        self.input_token = input_token
        self.neural_network = NeuralNetwork(path = nn_weigths_path, layers = [input_token, 300, input_token])
        print(self.neural_network)


    def get_successor_dict(self, encoded_token:list[int], min_combinations:int, max_combinations: int = 3, s_dict = {}) -> dict[str, dict[str, dict[str, int]]]:
        
        successor_dict: dict[str, dict[str, dict[str, int]]] = s_dict
                
        token_and_successor:list[tuple[list[int], int]] = list()
        for comb_len in range(min_combinations, max_combinations + 1) : 
            if len(encoded_token) > comb_len:
                for i in range(len(encoded_token) - comb_len):
                    token_and_successor.append((encoded_token[i : i + comb_len], encoded_token[i + comb_len]))

        # update
        for token, successor in token_and_successor:
            token_key = ", ".join(list(map(str, token)))
            if str(len(token)) not in successor_dict.keys():
                successor_dict[str(len(token))] = {}
            if token_key not in successor_dict[str(len(token))].keys():
                successor_dict[str(len(token))][token_key] = {}
            if str(successor) not in successor_dict[str(len(token))][token_key].keys():
                successor_dict[str(len(token))][token_key][str(successor)] = 1
            else:
                successor_dict[str(len(token))][token_key][str(successor)] = successor_dict[str(len(token))][token_key][str(successor)] + 1

        return successor_dict


    def train_neural_network(self, epochs, encoded_texts:list[list[int]], displayUpdate, min_keys:int = 1, max_keys:int = 1):

        X = list()
        Y = list()

        # create succesor dict
        successor_dict: dict[str, dict[str, dict[str, int]]] = {}
        for et in encoded_texts:
            successor_dict = self.get_successor_dict(et, min_keys, max_keys, successor_dict)

        # turn succesor dict to data
        for length in successor_dict.keys():
            for key in successor_dict[length].keys():
                try:
                    # gen input
                    keys = list(map(int, key.split(", ")))
                    x = [0] * self.input_token 
                    for k in keys:
                        x[k] = x[k] + 1

                    # gen lables
                    total_sucsessors = sum(successor_dict[length][key].values())
                    y:list[float] = [0.0] * self.input_token 
                    for k in successor_dict[length][key].keys():
                        y[int(k)] = successor_dict[length][key][k] / total_sucsessors
                    
                    X.append(x)
                    Y.append(y)
                except:
                    pass
                    
        self.neural_network.fit(numpy.array(X), numpy.array(Y), epochs, displayUpdate)
        self.neural_network.export_weights()


    def get_prediction_probabilities(self, x: list[int]) -> list[float]:
        
        X = [0] * self.input_token
        for index in x:
            X[index] = 1
        pred = self.neural_network.predict(X)
        return pred # type: ignore


    def get_verctor(self):
        #TODO 
        pass
        