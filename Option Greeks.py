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

# print("CallDelta = N(d1) = {:.2f}".format(calculateCallDelta(45,45,1,0.02,0.1)))
# print("PutDelta = -N(-d1) = {:.2f}".format(calculatePutDelta(45,45,1,0.02,0.1)))

#Delta sensitivity to underlying price S
def deltaVsStrike(S,K,t,r,vol,start,end,num,ax):
    
    K = np.linspace(start, end, num) #(start, end, number of even space)

    CallDelta = calculateCallDelta(S,K,t,r,vol)
    PutDelta = calculatePutDelta(S,K,t,r,vol)

    # plt.figure(figsize=(10, 6))
    # plt.plot(K,CallDelta, label="Call Delta")
    # plt.plot(K,PutDelta, linestyle="--", label="Put Delta")
    # plt.xlabel("Strike Price K")
    # plt.ylabel("Delta")
    # plt.axvline(x = S, linestyle="dotted", label="ATM Strike")
    # plt.title("Delta VS Strike (Underlying S = {})".format(S))
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    ax.plot(K, CallDelta, label="Call Delta")
    ax.plot(K, PutDelta, linestyle="--", label="Put Delta")
    ax.set_xlabel("Strike Price K")
    ax.set_ylabel("Delta")
    ax.axvline(x=S, linestyle="dotted", label="ATM Strike")
    ax.set_title("Delta VS Strike (Spot Price = {})".format(S))
    ax.legend()
    ax.grid(True)
    
    df = pd.DataFrame({
        'Strike Price': K,
        'Call Delta': CallDelta,
        'Put Delta': PutDelta
    })
    df.to_csv("deltaVsStrike.csv", index=False)

#Gamma is the rate of change in Delta per unit change in Underlying S; second order derivative of Delta
def calculateGamma(S,K,t,r,vol):
    d1 = (np.log(S/K) + (r + vol**2/2)*t) / (vol * np.sqrt(t))
    dN = np.exp(-d1**2/2) / np.sqrt(2*np.pi)
    return dN / (S*vol*np.sqrt(t))

# print(calculateGamma(S,K,t,r,vol))

def gammaVsStrike(S,K,t,r,vol,start,end,num,ax):
    
    K = np.linspace(start, end, num) #(start, end, number of even space)

    CallGamma = calculateGamma(S,K,t,r,vol)
    PutGamma = calculateGamma(S,K,t,r,vol)
    # print(len(CallGamma))

    # plt.figure(figsize=(10, 6))
    # plt.plot(K,CallGamma, label="Call Gamma")
    # plt.plot(K,PutGamma, linestyle="--", label="Put Gamma")
    # plt.xlabel("Strike Price K")
    # plt.ylabel("Gamma")
    # plt.axvline(x = S, linestyle="dotted", label="ATM Strike")
    # plt.title("Gamma VS Strike (Underlying S = {})".format(S))
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    ax.plot(K, CallGamma, label="Call Gamma")
    ax.plot(K, PutGamma, label="Put Gamma", linestyle="--")
    ax.set_xlabel("Strike Price K")
    ax.set_ylabel("Gamma")
    ax.axvline(x=S, linestyle="dotted", label="ATM Strike")
    ax.set_title("Gamma VS Strike (Spot Price = {})".format(S))
    ax.legend()
    ax.grid(True)

    #export to csv
    
    df = pd.DataFrame({
        'Strike Price': K,
        'Call Gamma': CallGamma,
        'Put Gamma': PutGamma
    })
    df.to_csv("gammaVsStrike.csv", index=False)

#Vega is the change in Option Value to changes in Volitility of underlying asset

def calculateVega(S,K,t,r,vol):
    d1 = (np.log(S/K) + (r + vol**2/2)*t) / (vol * np.sqrt(t))
    dN = np.exp(-d1**2/2) / np.sqrt(2*np.pi)
    return S*np.sqrt(t)*dN

def vegaVsStrike(S,K,t,r,vol,start,end,num,ax):
    
    K = np.linspace(start, end, num) #(start, end, number of even space)

    CallVega = calculateVega(S,K,t,r,vol)
    PutVega = calculateVega(S,K,t,r,vol)
    # print(len(CallGamma))

    # plt.figure(figsize=(10, 6))
    # plt.plot(K,CallVega, label="Call Vega")
    # plt.plot(K,PutVega, linestyle="--", label="Put Vega")
    # plt.xlabel("Strike Price K")
    # plt.ylabel("Vega")
    # plt.axvline(x = S, linestyle="dotted", label="ATM Strike")
    # plt.title("Vega VS Strike (Underlying S = {})".format(S))
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    ax.plot(K, CallVega, label="Call Vega")
    ax.plot(K, PutVega, label="Put Vega", linestyle="--")
    ax.set_xlabel("Strike Price K")
    ax.set_ylabel("Vega")
    ax.axvline(x=S, linestyle="dotted", label="ATM Strike")
    ax.set_title("Vega VS Strike (Spot Price = {})".format(S))
    ax.legend()
    ax.grid(True)

    #export to csv
    
    df = pd.DataFrame({
        'Strike Price': K,
        'Call Vega': CallVega,
        'Put Vega': PutVega
    })
    df.to_csv("VegaVsStrike.csv", index=False)

#Theta (Time Decay) is the change  in (usually loss in) Option Value to the passage time (time remaning); AKA option value's time decay, usually expressed in negative value

def calculateTheta(S,K,t,r,vol, option_type):
    d1 = (np.log(S/K) + (r + vol**2/2)*t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    
    pdf_d1 = np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)  # Standard normal PDF
    term1 = - (S * vol * pdf_d1) / (2 * np.sqrt(t))

    if option_type == 'call':
        theta = term1 - (r * K * np.exp(-r * t) * norm.cdf(d2))
    elif option_type == 'put':
        theta = term1 - (-(r * K * np.exp(-r * t) * norm.cdf(-d2)))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return theta / 365  # Convert to per day

def thetaVsStrike(S,K,t,r,vol,start,end,num,ax):
    
    K = np.linspace(start, end, num) #(start, end, number of even space)

    CallTheta = calculateTheta(S,K,t,r,vol, "call")
    PutTheta = calculateTheta(S,K,t,r,vol, "put")
    # print(len(CallGamma))

    # plt.figure(figsize=(10, 6))
    # plt.plot(K,CallTheta, label="Call Theta")
    # plt.plot(K,PutTheta, linestyle="--", label="Put Theta")
    # plt.xlabel("Strike Price K")
    # plt.ylabel("Theta")
    # plt.axvline(x = S, linestyle="dotted", label="ATM Strike")
    # plt.title("Theta VS Strike (Underlying S = {})".format(S))
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    ax.plot(K, CallTheta, label="Call Theta")
    ax.plot(K, PutTheta, linestyle="--", label="Put Theta")
    ax.set_xlabel("Strike Price K")
    ax.set_ylabel("Theta")
    ax.axvline(x=S, linestyle="dotted", label="ATM Strike")
    ax.set_title("Theta VS Strike (Spot Price = {})".format(S))
    ax.legend()
    ax.grid(True)

    #export to csv
    
    df = pd.DataFrame({
        'Strike Price': K,
        'Call Theta': CallTheta,
        'Put Theta': PutTheta
    })
    df.to_csv("ThetaVsStrike.csv", index=False)

S = 50 # underlying spot price
K = 50 # strike
t = 2 # in years
r = 0.015 # in %
vol = 0.1 # in %

start = S-30
end = S+30
num = 100

fig, axs = plt.subplots(2,2, figsize=(14, 9))

deltaVsStrike(S,K,t,r,vol,start,end,num,axs[0,0])
gammaVsStrike(S,K,t,r,vol,start,end,num,axs[0,1])
vegaVsStrike(S,K,t,r,vol,start,end,num,axs[1,1])
thetaVsStrike(S,K,t,r,vol,start,end,num,axs[1,0])

plt.tight_layout()
plt.show()
