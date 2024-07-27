import math
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

#Delta is the change of option value for a unit change in underlying S

#CallDelta = N(d1)
def calculateCallDelta(S,K,t,r,vol):
    d1 = (math.log(S/K) + (r + vol**2/2)*t) / (vol * math.sqrt(t))
    return norm.cdf(d1)

#PutDelta = -N(-d1)
def calculatePutDelta(S,K,t,r,vol):
    d1 = (math.log(S/K) + (r + vol**2/2)*t) / (vol * math.sqrt(t))
    return -norm.cdf(-d1)

# print("Calldelta = N(d1) = {:.2f}".format(calculateCallDelta(S,K,t,r,vol)))

#Delta sensitivity to underlying price S
def deltaSensitivityToS(S,K,t,r,vol,start,end,step):
    S_list = []
    
    i = start
    
    while i < end+step:
        S_list.append(i)
        i += step
    
    CallDelta_list = list()
    PutDelta_list = list()

    for i in S_list:
        CallDelta_list.append(round(calculateCallDelta(i,K,t,r,vol),2))
        PutDelta_list.append(round(calculatePutDelta(i,K,t,r,vol),2))
        
    data = {"S":S_list, "CallDelta":CallDelta_list, "PutDelta":PutDelta_list}
    df = pd.DataFrame(data=data)
    print(df)

    x = df["S"]
    y1 = df["CallDelta"]
    y2 = df["PutDelta"]
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.legend(df.columns[1:])
    plt.xlabel("Underlying S")
    plt.ylabel("Delta")
    plt.axvline(x = K, linestyle="dotted")
    plt.title("Underlying S0 = {} to S1 = {}".format(start, end))
    plt.show()


S = 45 # underlying
K = 60 # strike
t = 0.8 # in years
r = 0.1 # in %
vol = 0.2 # in %

start = K-30
end = K+30
step = 1

deltaSensitivityToS(S,K,t,r,vol,start,end,step)

#Delta Sensitivity to Volitility