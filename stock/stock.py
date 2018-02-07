from pandas_datareader import data, wb
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

class Stock(object):

    def __init__(self, name, begin, end):
        self.name = name
        self.begin = begin
        self.end = end
        self.data = None


    def get_data_set(self):
        self.data = data.get_data_google(self.name, self.begin, self.end)

        lastrowvalue = None
        self.data.insert(5, "today_dir", False)
        self.data.insert(6, "percent", 0.0)
        self.data.insert(7, "last_one", False)
        self.data.insert(8, "last_two", False)
        
        for index, row in self.data.iterrows():
            if lastrowvalue:
                self.data.loc[(index,"today_dir")] = row["Close"] > lastrowvalue
                self.data.loc[(index,"percent")] = 1.0 * (row["Close"] - lastrowvalue)/ lastrowvalue
                
            lastrowvalue = row["Close"]


        i = 0
        for index, row in self.data.iterrows():
            

            if (i-1 > 0):
                self.data.loc[(index,"last_one")] = self.data.iloc[i-1]["percent"] > 0.005
            if (i-2 > 0):
                self.data.loc[(index,"last_two")] = self.data.iloc[i-2]["percent"] > 0.005
            i += 1


    def get_decision_tree(self):
        dataset = self.data
        clf = DecisionTreeClassifier(random_state=14)
        
        y_true = dataset["today_dir"].values
        X_previouswins = dataset[["last_one", "last_two"]].values
        scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
        print "Accuracy: {0:.1f}%".format(np.mean(scores) * 100)


    def count_up_time(self):

        for index, row in self.data.iterrows():
            if row["last_one"] and row["last_two"] and row["today_dir"]:
                print "two day growth > 0.5%", index, row
            if row["last_one"] and row["last_two"] and not row["today_dir"]:
                #print "false====", index
                pass
    
    def draw_plot(self):

            plt.plot([index for index, row in self.data.iterrows()], [row["percent"] for index, row in self.data.iterrows()])

            plt.savefig('stock.png')


if __name__ == "__main__":

    s= Stock('BAC','1/1/2017','2/28/2017')

    s.get_data_set()

    #print s.data
    s.get_decision_tree()
    #s.count_up_time()
    s.draw_plot()
