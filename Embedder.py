# word2vec principle 


# 3 Layers (input hidden output)
# input layer n nodes wher n is the length of the encoding
# hidden layer m nodes where n is the vector length
# output layer n nodes where n is the lengt of the encoding (gives the porbablitiy of the word to follow up next)  

from NeuralNetworkJAX import NeuralNetworkJAX
import numpy

class Embedder: 

    neural_network: NeuralNetworkJAX
    input_token: int

    def __init__(self, input_token: int, alpha:float = 0.1, nn_weigths_path:str = "./data/embedder/jaxNN/weights.json"):
        self.input_token = input_token
        self.neural_network = NeuralNetworkJAX(path = nn_weigths_path, layers = [input_token, 300, input_token], alpha = alpha)


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
                    x = [0.0] * self.input_token 
                    for index in range(len(keys)):
                        x[keys[index]] = x[keys[index]] + ((index + 1) / len(keys))

                    # gen lables

                    # square occurances to incerase difference
                    for k in successor_dict[length][key].keys():
                        successor_dict[length][key][k] = pow(successor_dict[length][key][k], 2)

                    # sum up occurances
                    total_sucsessors = sum(successor_dict[length][key].values())
                    # turn occurances in to percentages
                    y:list[float] = [0.0] * self.input_token 
                    for k in successor_dict[length][key].keys():
                        y[int(k)] = successor_dict[length][key][k] / total_sucsessors

                    # add generated data
                    X.append(x)
                    Y.append(y)
                except:
                    pass
                    
        self.neural_network.train(numpy.array(X), numpy.array(Y), epochs, displayUpdate)
        self.neural_network.save()


    def get_prediction_probabilities(self, x: list[int]):
        
        X = [0.0] * self.input_token
        for index in range(len(x)):
            X[x[index]] = X[x[index]] + ((index + 1) / len(x))
        pred = self.neural_network.predict(numpy.array(X))
        return pred


    def get_verctor(self, x: list[int], v_layer = 0):
        """
        Converts a list of indices to a one-hot encoded vector and then passes it 
        through the neural network layers.
        
        :param x: List of indices representing active features.
        :param v_layer: Number of layers to pass trouh (default is 0).
        :return: A list of float values representing the output vector after neural network processing.
        """
        X = [0.0] * self.input_token
        for index in range(len(x)):
            X[x[index]] = X[x[index]] + ((index + 1) / len(x))
        vector = self.neural_network.layer_vector(x=numpy.array(X), layer_index=v_layer)
        return vector
        