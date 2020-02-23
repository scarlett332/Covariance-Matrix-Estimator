import numpy as np
import pandas as pd


class Estimator:
    def __init__(self):
        """
        Create Stock estimator
        """
    def estimate(self,est_type=0,stock_info = None, market_info = None):
        if est_type==0:
            return self.__cal_sample_cov_est(stock_info)
        elif est_type==1:
            return self.__cal_market_est(market_info,stock_info)
        elif est_type==2:
            return self.__cal_shrinkage_est(market_info,stock_info)

    def __cal_sample_cov_est(self,x):
        """
        args:
        x: T x N, stocks returns
    
        returns:
        S : N x N, shrinkage estimators
    
        """
        S = np.cov(x,rowvar = False)
        return S
    
    def __cal_market_est(self,x,y):
        """
        args:
        x: T x 1 market returns
        y: T x N stocks returns
    
        returns:
        F : N x N market estimators
    
        """
        T, N = y.shape
        #Linear regression 
        x_bar = x.mean()
        y_bar = y.mean(axis=0)
        beta_1 = np.sum((y-y_bar)*(x),axis=0)/np.sum((x-x_bar)*x,axis=0)
        beta_0 = y_bar - beta_1*x_bar
        residual = y - (beta_1*x+beta_0)
        delta = np.sum(residual**2,axis=0)/(N-2)
        D = np.diag(delta)
        s_2 = x.var()
        F =  s_2*beta_1[:,np.newaxis]@beta_1[:,np.newaxis].T + D
        return F
    
    def __cal_shrinkage_est(self,x,y):
        """
        args:
        x:  T x 1 market returns
        y:  N x N stocks returns
    
        returns:
        E :  shrinkage estimators
    
        """
        T, N = y.shape
    
        x_bar = x.mean()
        y_bar = y.mean(axis=0)
        F = self.__cal_market_est(y,x)
        S = self.__cal_sample_cov_est(y)
    
        #Calculate the covariance between y,x
        yx = np.concatenate((y,x),axis=1)
        yx_cov = np.cov(yx,rowvar=False)
        s_x0 = np.expand_dims(yx_cov[-1,:-1],axis=1)
        #Calculate gamma
        gamma = ((F-S)**2).sum().sum()
    
        #Calculate pi
        pi_mat = ((((y-y_bar)**2).T@((y-y_bar)**2)- 2*(y-y_bar).T@(y-y_bar)*S + T*S*S)/T)
        pi = pi_mat.sum().sum()
    
        #Calculate rou
        rou_mat = np.zeros((N,N))
        s_00 = yx_cov[-1,-1]
        for j in range(T):
            y_e = np.expand_dims(y[j,:]-y_bar,axis=1)
            rou_mat+=((x[j]-x_bar)*(s_00*s_x0@y_e.T+s_00*y_e@s_x0.T-s_x0@s_x0.T*(x[j]-x_bar))*(y_e@y_e.T)/(s_00**2) - F*S)
        np.fill_diagonal(rou_mat,np.diag(pi_mat))
        rou =rou_mat.sum().sum()
        #Calculate the coefficient
        k = (pi-rou)/gamma
        #Get the estimator
        E = k/T * F +(T-k)/T* S
        return E
        
