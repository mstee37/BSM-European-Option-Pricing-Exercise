from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Delta is the change of option value for a unit change in underlying S

#CallDelta = N(d1)
def calculateCallDelta(S,K,t,r,vol):
    d1 = (np.log(S/K) + (r + vol**2/2)*t) / (vol * np.sqrt(t))
    return norm.cdf(d1)

#PutDelta = -N(-d1)
def calculatePutDelta(S,K,t,r,vol):
    d1 = (np.log(S/K) + (r + vol**2/2)*t) / (vol * np.sqrt(t))
    return -norm.cdf(-d1)

print("CallDelta = N(d1) = {:.2f}".format(calculateCallDelta(45,45,1,0.02,0.1)))
print("PutDelta = -N(-d1) = {:.2f}".format(calculatePutDelta(45,45,1,0.02,0.1)))

#Delta sensitivity to underlying price S
def deltaVsStrike(S,K,t,r,vol,start,end,num):
    
    K = np.linspace(start, end, num) #(start, end, number of even space)

    CallDelta = calculateCallDelta(S,K,t,r,vol)
    PutDelta = calculatePutDelta(S,K,t,r,vol)

    plt.figure(figsize=(10, 6))
    plt.plot(K,CallDelta, label="Call Delta")
    plt.plot(K,PutDelta, linestyle="--", label="Put Delta")
    plt.xlabel("Strike Price K")
    plt.ylabel("Delta")
    plt.axvline(x = S, linestyle="dotted", label="ATM Strike")
    plt.title("Delta VS Strike (Underlying S = {})".format(S))
    plt.legend()
    plt.grid(True)
    plt.show()

#Gamma is the rate of change in Delta per unit change in Underlying S; second order derivative of Delta
def calculateGamma(S,K,t,r,vol):
    d1 = (np.log(S/K) + (r + vol**2/2)*t) / (vol * np.sqrt(t))
    dN = np.exp(-d1**2/2) / np.sqrt(2*np.pi)
    return dN / (S*vol*np.sqrt(t))

# print(calculateGamma(S,K,t,r,vol))

def gammaVsStrike(S,K,t,r,vol,start,end,num):
    
    K = np.linspace(start, end, num) #(start, end, number of even space)

    CallGamma = calculateGamma(S,K,t,r,vol)
    PutGamma = calculateGamma(S,K,t,r,vol)
    # print(len(CallGamma))

    plt.figure(figsize=(10, 6))
    plt.plot(K,CallGamma, label="Call Gamma")
    plt.plot(K,PutGamma, linestyle="--", label="Put Gamma")
    plt.xlabel("Strike Price K")
    plt.ylabel("Gamma")
    plt.axvline(x = S, linestyle="dotted", label="ATM Strike")
    plt.title("Gamma VS Strike (Underlying S = {})".format(S))
    plt.legend()
    plt.grid(True)
    plt.show()

S = 50 # underlying spot price
K = 50 # strike
t = 2 # in years
r = 0.02 # in %
vol = 0.1 # in %

start = S-30
end = S+30
num = 100

deltaVsStrike(S,K,t,r,vol,start,end,num)
gammaVsStrike(S,K,t,r,vol,start,end,num)

#Vega is the change in Option Value with respect to Volitility of underlying asset

#Theta is the change  in (usually loss in) Option Value for change in time remaining; AKA option value's time decay, usually expressed in negative value

