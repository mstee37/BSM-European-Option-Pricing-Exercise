import math
from scipy.stats import norm

# C = S*N(d1) - K*e^(-rt) * N(d2) (max(S-K,0))
# P = K*e^(-rt) N(-d2) - S*N(-d1) (max(K-S,0))
# d1 = (ln(S/K) + (r + vol^2/2)t) / (vol * sqrt of t)
# d2 = d1 - vol * sqrt of t

# C = call option price (max(S-K,0))
# P = put option price (max(K-S,0))
# S = underlying
# K = strike
# r = risk free
# t = time to maturity in years
# N = a normal dist
# vol = implied vol
# N(d1) = delta -> probability of how far ITM the underlying price will be
# N(d2) = probability that the option expires ITM

S = 45 # underlying
K = 60 # strike
t = 0.8 # in years
r = 0.1 # in %
vol = 0.2 # in %

d1 = (math.log(S/K) + (r + vol**2/2)*t) / (vol * math.sqrt(t))
d2 = d1 - vol * math.sqrt(t)

C = S * norm.cdf(d1) - K * math.exp(-r*t) * norm.cdf(d2)
P = K * math.exp(-r*t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
# N(d1) = delta -> probability of how far ITM the underlying price will be
# N(d2) = probability that the option expires ITM
print("d1 = {:.5f}".format(d1))
print("Call N(d1) = {:.5f}".format(norm.cdf(d1)))
print("Call N(d2) = {:.5f}".format(norm.cdf(d2)))
print()
print("-d1 = {:.5f}".format(-d1))
print("Put N(-d1) = {:.5f}".format(norm.cdf(-d1)))
print("Put N(-d2) = {:.5f}".format(norm.cdf(-d2)))
print()
print("call option price = {:.2f}".format(C))
print("put option price = {:.2f}".format(P))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def callPrice(S, K, t, r, vol):
    d1 = (math.log(S/K) + (r + vol**2/2)*t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    return S * norm.cdf(d1) - K * math.exp(-r*t) * norm.cdf(d2)

def putPrice(S, K, t, r, vol):
    d1 = (math.log(S/K) + (r + vol**2/2)*t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    return K * math.exp(-r*t) * norm.cdf(-d2) - S * norm.cdf(-d1)


def changesInK(S, K, t, r, vol, strikeRangeFromS, ax):
    
    K_list = list()
    C_list = list()
    P_list = list()

    for strike in range(S-strikeRangeFromS, S+strikeRangeFromS+1):
        K_list.append(strike)
        C_list.append(round(callPrice(S,strike,t,r,vol),2))
        P_list.append(round(putPrice(S,strike,t,r,vol),2))
        
    data = {"K":K_list, "Call Prem":C_list, "Put Prem":P_list}
    df = pd.DataFrame(data=data)

    x = df["K"]
    y1 = df["Call Prem"]
    y2 = df["Put Prem"]
    ax.plot(x,y1)
    ax.plot(x,y2, "-.")
    ax.legend(df.columns[1:])
    ax.set_xlabel("strike")
    ax.set_ylabel("Option Prem")
    ax.axvline(x = S, linestyle="dotted")
    ax.set_title("Underlying price S = {}".format(S))
    ax.grid(True)

def changesInS(S, K, t, r, vol, SRangeFromK, ax):
    
    S_list = list()
    C_list = list()
    P_list = list()

    for i in range(K-SRangeFromK, K+SRangeFromK+1):
        S_list.append(i)
        C_list.append(round(callPrice(i,K,t,r,vol),2))
        P_list.append(round(putPrice(i,K,t,r,vol),2))
        
    data = {"S":S_list, "Call Prem":C_list, "Put Prem":P_list}
    df = pd.DataFrame(data=data)

    x = df["S"]
    y1 = df["Call Prem"]
    y2 = df["Put Prem"]
    ax.plot(x,y1)
    ax.plot(x,y2, "-.")
    ax.legend(df.columns[1:])
    ax.set_xlabel("Underlying S")
    ax.set_ylabel("Option Prem")
    ax.axvline(x = K, linestyle="dotted")
    ax.set_title("Strike K = {}".format(K))
    ax.grid(True)

def changesInT(S, K, T, r, vol, start, end, step, ax):
    
    t_list = list()
    
    i = start
    
    while i < end+step:
        t_list.append(i)
        i += step
    # print(t_list[-1])
    
    C_list = list()
    P_list = list()

    for i in t_list:
        C_list.append(round(callPrice(S,K,i,r,vol),2))
        P_list.append(round(putPrice(S,K,i,r,vol),2))
        
    data = {"t":t_list, "Call Prem":C_list, "Put Prem":P_list}
    df = pd.DataFrame(data=data)
    # print(df)

    x = df["t"]
    y1 = df["Call Prem"]
    y2 = df["Put Prem"]
    ax.plot(x,y1)
    ax.plot(x,y2, "-.")
    ax.legend(df.columns[1:])
    ax.set_xlabel("Time to maturity")
    ax.set_ylabel("Option Prem")
    # plt.axvline(x = K, linestyle="dotted")
    ax.set_title("t0 = {} to t1 = {}".format(start, end))
    ax.grid(True)

def changesInR(S, K, T, r, vol, start, end, step, ax):
    
    r_list = list()
    
    i = start
    
    while i < end+step:
        r_list.append(i)
        i += step
    # print(r_list[-1])

    
    C_list = list()
    P_list = list()

    for i in r_list:
        C_list.append(round(callPrice(S,K,T,i,vol),2))
        P_list.append(round(putPrice(S,K,T,i,vol),2))
        
    data = {"r":r_list, "Call Prem":C_list, "Put Prem":P_list}
    df = pd.DataFrame(data=data)
    # print(df)

    x = df["r"]
    y1 = df["Call Prem"]
    y2 = df["Put Prem"]
    ax.plot(x,y1)
    ax.plot(x,y2, "-.")
    ax.legend(df.columns[1:])
    ax.set_xlabel("Risk free rate (in %)")
    ax.set_ylabel("Option Prem")
    # plt.axvline(x = K, linestyle="dotted")
    ax.set_title("r0 = {} to r1 = {}".format(start, end))
    ax.grid(True)
    
def changesInVol(S, K, T, r, vol, start, end, step, ax):
    
    vol_list = list()
    
    i = start
    
    while i < end+step:
        vol_list.append(i)
        i += step
    # print(vol_list[-1])
    
    C_list = list()
    P_list = list()

    for i in vol_list:
        C_list.append(round(callPrice(S,K,T,r,i),2))
        P_list.append(round(putPrice(S,K,T,r,i),2))
        
    data = {"vol":vol_list, "Call Prem":C_list, "Put Prem":P_list}
    df = pd.DataFrame(data=data)
    # print(df.index.values)

    x = df["vol"]
    y1 = df["Call Prem"]
    y2 = df["Put Prem"]
    ax.plot(x,y1)
    ax.plot(x,y2, "-.")
    ax.legend(df.columns[1:])
    ax.set_xlabel("Volitility (in %)")
    ax.set_ylabel("Option Prem")
    # plt.axvline(x = K, linestyle="dotted")
    ax.set_title("vol0 = {} to vol1 = {}".format(start, end))
    ax.grid(True)

fig, axs = plt.subplots(2,3, figsize=(18, 9))

S = 45 # underlying
K = 45 # strike
t = 1.5 # in years
r = 0.02 # in %
vol = 0.2 # in %

#option price sensitivity to strike price = K
changesInK(S,K,t,r,vol,10,axs[0,0])

#option price sensitivity to underlying = S
changesInS(S,K,t,r,vol,10,axs[1,0])

#option price sensitivity to time = t
start = 1
end = 5
step = 1
changesInT(S,K,t,r,vol,start,end,step,axs[0,1])

#option price sensitivity to risk-free rate = r
start = 0.01
end = 0.08
step = 0.005
changesInR(S,K,t,r,vol,start,end,step,axs[1,1])

#option price sensitivity to volitility = vol
start = 0.01
end = 0.3
step = 0.01
changesInVol(S,K,t,r,vol,start,end,step,axs[0,2])

plt.tight_layout()
plt.show()

#Put-call Parity

def putCallParityTest(S,K,t,r,vol):
    #Call and present value of Strike = K
    call = round(callPrice(S,K,t,r,vol) + K * np.exp(-r*t),5)
    print("Call price + present value of strike price K = {:.2f}".format(call))

    #Put and current price of underlying = S
    put = round(putPrice(S,K,t,r,vol) + S,5)
    print("Put price + current price of underlying S = {:.2f}".format(put))
    
    print("Put-call parity test is {}".format(call == put))

putCallParityTest(S,K,t,r,vol)
