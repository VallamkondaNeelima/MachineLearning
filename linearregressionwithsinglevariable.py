import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr

def coef(x,y):
    n=np.size(x)
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    b1= (np.sum(y*x)-n*x_mean*y_mean)/(np.sum(x*x)-n*x_mean*x_mean)#assosciativity of ** is greater than *
    b0=y_mean-b1*x_mean
    return b0,b1

x=np.array([0,1,2,3,4,5,6,7,8,9])
y=np.array([1,3,2,5,7,8,8,9,10,12])
    
b0,b1 = coef(x,y)
print(b0,b1)

def plotting(x,y):
    plt.scatter(x,y,color='k',label="original")
    plt.legend()
    y_l=b0+b1*x
    plt.plot(x,y_l,label="linear")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.show()
plotting(x,y)
reg=lr().fit(x.reshape(-1,1),y.reshape(-1,1))
print(reg.intercept_,reg.coef_)

    
    
    
