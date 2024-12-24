import pandas as pd
import numpy as np

class process():
    #Input_Data - Pandas DataFrame of Data
    #Response - Feature we are trying to predict
    def __init__(self, input_data, response, back_days=15):
        self.x = []
        back = list(range(0,back_days))
        for col in input_data:
            self.x.append(
                np.expand_dims(
                np.array(
                pd.concat(list(map(
                lambda n: col.shift(n), back # Add 15 lagged features 
                )),axis=1)
                ).reset_index(drop=True).loc[:,::-1]) # Reverse the order of the columns
                ,axis=2)
        self.x = np.concatenate(tuple(self.x), axis=2)
        