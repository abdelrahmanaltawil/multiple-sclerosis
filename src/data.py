# env imports
import sys
import json
import itertools
import pandas as pd
from pathlib import Path
import sklearn.model_selection as skl
from tabulate import tabulate

# setup work directory
src_path = Path(__file__).parents[0].__str__()
sys.path.append(src_path) if src_path not in sys.path else None

# local imports


class Data:
    '''
    A class used to represent an Dataset

    ...

    Attributes
    ----------
    samples     :   Dataframe
                    list of all Node objects the construct the optimal path.
    training    :   Dataframe
                    Portion of data set used for model training
    test        :   Dataframe
                    Portion of data set used for model testing
    num_legs    :   int
                    the number of legs the animal has (default 4)

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    '''

    def __init__(self, input, path=None, feature_indeces=None, label_index=None, decimal=','):
        '''
        return the optimal path between start and goal nodes
        
        Arguments
        ----------
        path            :   path
                            The path of the data file with spcifing the formate extension
        feature_indeces :   list
                            list of indecis or indecis ranges of the feature columns in the data set
        label_index     :   list
                            index of the label column in the data set
        decimal         :   string, optional
                            Define the decimal format scheme either EU `,` or US `.` Default is `,`
        '''  
        
        self.features = []
        self.labels = []

        # for summary
        self.report = []

        self.samples = self.read(
            path=Path(input["data path"]),
            decimal=input["decimal separator"],
            input=input
            )


    def parse_input(self):
        pass


    def read(self, path: Path, decimal: str, input: dict) -> pd.DataFrame:
        '''
        Reading the data file and reduce it to feature labels data only
        '''
        
        thousands = ',' if decimal=='.' else '.'

        if path.suffix == '.csv':
            df = pd.read_csv(path, sep=';', decimal=decimal, thousands=thousands)

        elif path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path, decimal=decimal, thousands=thousands)

        else:
            raise NotImplementedError("The following file extension is not supported,"
                                    + "consider the following extensions" 
                                    + ".csv, .xlsx, .xls")


        for x0, xn in input["features"]:
            self.features.extend(list(df.columns[x0:xn]))
        
        self.labels = list(df.columns[input["label"]])

        samples = df[self.features + self.labels].copy()

        self.report.append(["Raw Data", f"{len(samples)}"])

        return samples


    def clean(self):
        '''
            This method by no means meant to be compahinsive, however the user have to descide what cleaning schemes to use

            Arguments
            ----------
            No arguments
            
            Return
            -------
            No return "works on `self.samples`"
        ''' 

        self.samples = self.samples.dropna()

        self.report.append(["After Cleaning", len(self.samples)])




    def split_samples(self, train_size=0.7):
        training, testing = skl.train_test_split(self.samples, train_size=train_size)

        self.report.append(["Training", f"{len(training)} ({int(train_size*100)}%)"])
        self.report.append(["Testing", f"{len(testing)} ({int((1-train_size)*100)}%)"])

        return training, testing


    def summary(self):
        '''
        Display samples states information in tabular fromat
        '''

        # Features and labels
        data = [[x, y] for x, y in itertools.zip_longest(self.features, self.labels)]
        self.table( data=data,
                    headers=['Features', 'Labels'],
                    Notes=f'Total features: {len(self.features)} \n'
                            +f'Total labels: {len(self.labels)}'
                        )
        
        # Sample Stats
        self.table( data=self.report,
                    headers=['Samples', 'Count'],
                        )

    
    def table(self, data, headers, Notes=None):
        table = tabulate(data, headers=headers, tablefmt='simple')
        
        table_width = len(table[table.find('\n'): table.rfind("-\n")])
        devider = lambda symbol: symbol*table_width

        print(devider("="))
        print(table)
        if Notes != None:
            print(devider("-"))
            print(Notes)
        print(devider("="),'\n')

        return table

        



if __name__ == "__main__":

    settings = json.load(open(Path(src_path + '/resources/settings.json')))


    data = Data(settings["data"])

    data.clean()
    traning, testing = data.split_samples(train_size=0.7)

    data.summary()