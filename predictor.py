import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# sort nums
def sort_nums(df):
    df = df.to_numpy()
    number_set = []
    for row in df:
        sorted_num = sorted(row.tolist())
        number_set.append(sorted_num)
    return number_set


# read excel
draws = pd.read_excel("data.xlsx")
draws = pd.DataFrame(sort_nums(draws))

# use latest 10 draws
draws = draws.iloc[-10:]

# set the y and x
y = draws.to_numpy()
x = np.array([*range(1, len(y) + 1)])

# setup model
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

x_test = np.array([len(x) + 1]).reshape(-1, 1)

# predict
prediction = model.predict(x_test)
result = np.round(prediction[0])

print(result)
