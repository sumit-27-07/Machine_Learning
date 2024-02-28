import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
def gradient_descent(x,y):
    m_curr=b_curr=0
    iter=100
    n=len(x)
    learning_rate=0.0002
    for i in range(iter):
        yp=m_curr*x+b_curr
        cost=(1/n)*sum([val**2 for val in (y-yp)])
        md=-(2/n)*sum(x*(y-yp))
        bd=-(2/n)*sum(y-yp)
        m_curr=m_curr-learning_rate*md
        b_curr = b_curr - learning_rate * bd
        print("M={},B={},cost={},iteration {}\n".format(m_curr,b_curr,cost,i))

df=pd.read_csv(r"C:\Users\sumit\Desktop\Code\python\Machine Learning\Grediant Desent\test_scores.csv")

math=np.array(df.math)

cs=np.array(df.cs)
gradient_descent(math,cs)
