import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv(r"C:\Users\sumit\Desktop\Code\python\Machine Learning\homeprices.csv")
print(df)
plt.scatter(df.area,df.price)
plt.show()

reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
area1 = 3300 
arr = np.array([[area1]])
predict = reg.predict(arr)
print(predict)

#print(f"Predicted value for {area1}: {predicted_value[0]}")