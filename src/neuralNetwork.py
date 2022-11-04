# env imports
import sys
import json
from pathlib import Path
from tensorflow import keras

# setup work directory
src_path = Path(__file__).parents[0].__str__()
sys.path.append(src_path) if src_path not in sys.path else None

# local imports
from data import Data 



class NeuralNetwork():
    
    def __init__(self, input):
        '''
        testing
        '''

        self.samples = []
        
        self.features = None
        self.features_scaling = []

        self.labels = None
        self.label_scaling = []

        self.model = None
        self.structure(input)


    def structure(self, input):

        self.model = keras.Sequential()

        self.model.add(
            keras.layers.Dense(
                input["spread"]*90, 
                activation='relu', 
                use_bias = False,
                input_shape=[90]))

        for i in range(1, input["depth"]):
            self.model.add(
                keras.layers.Dense(
                        input["spread"]*90, 
                        use_bias = False,
                        activation='relu'
                        )
                    )

        self.model.add(keras.layers.Dense(1))

        self.model.compile( loss='mse',
                            optimizer='RMSProp', #Adadelta
                            metrics=['mae', 'mse'])
        
        self.model.summary()


    def fit(self, training):
        '''
        Train neural network using the provided samples `training` to train the network
        the samples are normlized before feeding to network.
                    `training` => normalization => fiting 
        
        Argument
        ----------
        training    :   ndarray
                        Training samples the will tune model parameters  
        '''
        
        self.samples = training.copy()

        self.features = self.samples.iloc[:,:-1]
        self.labels = self.samples.iloc[:,-1]

        self.set_scaling()

        self.features = self.normalize(self.features)
        self.labels = self.normalize(self.labels)


        history = self.model.fit(
            self.features, 
            self.labels,
            epochs = 1000,
            verbose=0)
            
        print(history)

    
    def test(self, testing):
        '''
        Take testing data to see the model perfromance and tace the model quality measures
        
        Argument
        ----------
        testing     :   ndarray
                        Testing samples to evaluate the model performance
        '''

        samples = testing.copy()

        features = samples.iloc[:,:-1]
        labels = samples.iloc[:,-1]

        features = self.normalize(features, scale="features")
        labels = self.normalize(labels, scale="labels")

        results = self.model.evaluate(features, labels)

        # print(tabulate([results], headers=['test loss', 'test Accuracy'], tablefmt='rst'), '\n')




    def set_scaling(self):
        self.features_scaling = [self.features.max(), self.features.min()]
        self.label_scaling = [self.labels.max(), self.labels.min()]


    def predict(self, features):
        
        
        features = self.normalize(features, scale="features")

        label = self.model.predict(features)
        
        labels = self.denormaloze(self.labels, scale="labels")

        return labels


    
    def normalize(self, samples, scale="None"):
        '''
        normalize the data on the scale defined by training data
        
        Argument
        ----------
        sample  :   dataframe
                    Represent the data, could be features or labels
        scale   :   string
                    This is used when data need to mapped to training data scale 
                    either `features` or `labels` scale.    
        '''

        if scale == "None":
            max, min =  samples.max(), samples.min()
        elif scale == "features":
            max, min =  self.features_scaling
        elif scale == "labels":
            max, min =  self.label_scaling
        else:
            raise NotImplementedError("There is typo in scale string, Please check method documentation")

        normalize = lambda data : (data - min) / (max - min)

        return normalize(samples)


    def denormaloze(self, samples, scale):
        '''
        denormalize the data on the scale defined by training data
                
        Argument
        ----------
        sample  :   dataframe
                    Represent the data, could be features or labels
        scale   :   string
                    This is used when data need to mapped to training data scale 
                    either `features` or `labels` scale.    
        '''
        if scale == "features":
            max, min =  self.features_scaling
        elif scale == "labels":
            max, min =  self.label_scaling
        else:
            raise NotImplementedError("There is typo in scale string, Please check method documentation")

        denormalize = lambda data : (data)*(max - min) + min 

        return denormalize(samples)


    def summary(self):
        pass

if __name__ == "__main__":


    settings = json.load(open(Path(src_path + '/resources/settings.json')))
    
    data = Data(settings["data"])
    data.clean()
    traning, testing = data.split_samples(settings["data"]["training ratio"])
    data.summary()

    NN = NeuralNetwork(settings["neural network"])

    NN.fit(traning)
    NN.test(testing)

    # labels = NN.predict(features)



    NN.summary()
