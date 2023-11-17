import pandas as pd
import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark-palette')

'''Intent is to us a spreadsheet data source for statistical comparisons'''


class measurements:
    def __init__(self, Job_number):
        '''Initializes a class object for comparison of measurements.'''
        ## Job_number is a string and is only used for tracking.
        self.Job_number = Job_number
        self.bp = {'b':'Bushing', 'p':'Pin', 'full':'Bushing and Pin together', 'half':'Bushing and Pin separate'}


    def data_in(self, filepath, column_range, sheet_name=0):
        '''Takes a specified filepath and reads entries, eliminating unnecessary columns and rows. Filepath and column_range must both be strings.'''
        ## Using colums specified with format 'C:I' for example
        self.dataframe = pd.read_excel(filepath, usecols=column_range, sheet_name=sheet_name)
        ## Removing the pesky 'NaN' values using the first column as a guide
        self.dataframe.dropna(subset=['Unnamed: 5'], inplace=True)
        # Resetting the indices so everything is sequential
        self.dataframe.reset_index(drop=True,inplace=True)
        ## Renaming columns and recording the target values
        self.targets = {}
        self.tolerances = {}
        for i in x.dataframe:
            name = x.dataframe.iloc[0][i]
            self.targets[name] = (x.dataframe.iloc[1][i])
            self.tolerances[name] = (str(x.dataframe.iloc[2][i]))
            self.dataframe.rename(columns={i:name},inplace=True)
        ## Dropping the standard header and resetting indices once more, renaming first column
        self.dataframe.drop(axis=0, index=[0,1,2], inplace=True)
        self.dataframe.rename(columns={'Dimension':'Item'},inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)
        del self.targets['Dimension']
        del self.tolerances['Dimension']
        return self.dataframe



    def B_P_filter(self, half):
        '''Will cut data in half for analyzing bushing or pin half alone'''
        x = self.dataframe[self.dataframe['Item'].str.contains(half.lower())]
        y = self.dataframe[self.dataframe['Item'].str.contains(half.upper())]
        if x.size>y.size:
            return x
        else:
            return y

        #return self.dataframe[self.dataframe['Item'].str.contains(half.lower() or half.upper())]



    def quick_stats(self, dimension):
        '''Returns pandas standard quick statistics about a specific dimension or dimensions. Dimension must be provided as a string.'''
        full_half = str(input("Specify 'full' for full set, or 'b' / 'p' for bushing or pin respectively: "))
        if full_half.lower() == 'full':
            S = self.dataframe[dimension].astype(float).describe()
        else:
            S = self.B_P_filter(full_half)[dimension].astype(float).describe()
        return S



    def distribution(self, dimension, full_half = 'full'):
        '''Will return functions for plotting a distribution graph of the specified dimension'''
        if full_half != 'full':
            values = self.B_P_filter(full_half)[dimension].astype(float)
        else:
            values = self.dataframe[dimension].astype(float)
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof = 1)
        tstat_0 = t.interval(0.95, n-1)
        CI = (tstat_0[1]*std)/np.sqrt(n)
        x = np.linspace([mean-6*CI], [mean+6*CI])     ##With 95% accuracy for dataset 1
        y = norm.pdf(x, mean, std)
        return [(x,y), full_half, mean]



    def plotting(self, dimension, full_half = None):
        '''Intended to plot a single dimension specified, either as one pdf or two (one for each half).'''
        ## Prompting user
        if full_half is None:
            full_half = str(input("Specify 'full' for full set, 'half' for two peaks, or 'b' / 'p' for bushing or pin respectively: "))
        if full_half.lower() == 'half':
            S = [self.distribution(dimension, 'b'), self.distribution(dimension, 'p')]
        else:
            S = [self.distribution(dimension, full_half)]
        ## The actual plotting part
        plt.figure(figsize = (10,6))
        for i in S:
            largest = 0
            x,y = (i[0])
            plt.plot(x,y, label = self.bp[i[1]])
            #plt.hist(x, bins=100)
            if max(y) > largest:
                largest = max(y)
            plt.annotate(f'{i[2].round(4)}', xy=(i[2], largest))
        plt.title(f"{self.Job_number} Density distributions of {dimension} dimension for {self.bp[full_half]}")
        plt.xlabel("Measurement")
        plt.ylabel("Probability density")
        plt.vlines(self.tols(dimension), 0, largest, color='orange', label='Tolerance')
        plt.vlines(self.targets[dimension], 0, largest, color='red', label='Target')
        plt.legend(loc='lower right')
        plt.show()


        
    def plot_all(self):
        for i in self.dataframe:
            if i != 'Item':
                self.plotting(i, 'half')



    def tols(self, dimension):
        '''A tool to return the acceptable error bounds that will be plotted on the graphs.'''
        ## real brick head way of doing this
        tolerance = self.tolerances[dimension]
        if '±' in tolerance:
            low = self.targets[dimension] - float(tolerance[1:])
            high = self.targets[dimension] + float(tolerance[1:])
        elif '-' in tolerance:
            low = self.targets[dimension] - float(tolerance[1:])
            high = self.targets[dimension]
        elif '+' in tolerance:
            low = self.targets[dimension]
            high = self.targets[dimension] + float(tolerance[1:])

        return (low,high)



if __name__ == "__main__":
    x = measurements('Z149A3')
    x.data_in(r'C:\Users\inspect\Desktop\Z149A3 top plate inspection report.xlsx', 'F:O', sheet_name=0)
    x.plot_all()
