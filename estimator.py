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
        F = self.__cal_market_est(x,y)
        S = self.__cal_sample_cov_est(y)
    
        #Calculate the covariance between y,x
        yx = np.concatenate((y,x),axis=1)
        yx_cov = np.cov(yx,rowvar=False)
        s_x0 = np.expand_dims(yx_cov[-1,:-1],axis=1)

        #Calculate gamma
        gamma = ((F-S)**2).sum().sum()
    
        #Calculate pi
        pi_mat = np.zeros((N,N))
        pi_mat = ((((y-y_bar)**2).T@((y-y_bar)**2)- 2*(y-y_bar).T@(y-y_bar)*S + T*S*S)/T)
        pi = pi_mat.sum().sum()
    
        #Calculate rou with vectorization
        rou_mat = np.zeros((N,N))
        s_00 = yx_cov[-1,-1]
        s_x0_vec = np.repeat(s_x0,T,axis=1)
        y_norm = y-y_bar
        x_norm = x-x_bar

        s1 =  (s_x0_vec*y_norm.T)@(x_norm*(y_norm**2))/s_00
        s2 =  s1.T
        s3 =  (x_norm.T*s_x0_vec*y_norm.T)@(x_norm*(s_x0_vec.T*y_norm))/(s_00**2)

        rou_mat = (s1+s2-s3 - T*F*S)/T
        np.fill_diagonal(rou_mat,np.diag(pi_mat))
        rou =rou_mat.sum().sum()

        #Calculate the coefficient
        k = (pi-rou)/gamma

        #Get the estimator
        E = k/T * F +(T-k)/T* S
        return E
        
