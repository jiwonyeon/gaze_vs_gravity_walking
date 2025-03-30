import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def main_sequence(x, a, c):
        return a * (1 - 1 / np.exp(x/c))

class saccade_filter():
    def __init__(self):        
        self.amplitude = None
        self.duration = None
        self.peak_velocity = None        
        self.filter_res = None
        self.amp_vel_res = {}
        self.amp_dur_res = {}
    
    def _check_name(self, res_name):
        a = "amplitude"
        if res_name == "amp_vel_res":
            b = "peak_velocity"
        elif res_name == "amp_dur_res":
            b = "duration"        
            
        return a, b
    
    def _save_fitting_res(self, popt, res_name, mask):
        """ 
            mask is a boolean. To consider all data points, create a boolean with "True" value with the number of the data points.
        """
        a, b = self._check_name(res_name)
        y_pred = main_sequence(getattr(self, a)[mask], *popt)
        ss_total = np.sum((getattr(self, b)[mask]-np.mean(getattr(self, b)[mask]))**2)
        ss_residual = np.sum((getattr(self, b)[mask]-y_pred)**2)
        r_squared = 1 - (ss_residual/ss_total)
        residual_std = np.std(getattr(self, b)[mask]-y_pred)
        
        getattr(self, res_name)['popt'] = popt
        getattr(self, res_name)['y_pred'] = y_pred
        getattr(self, res_name)['r_squared'] = r_squared
        getattr(self, res_name)['residual_std'] = residual_std
    
    def _filtering(self, res_name, sigma=2):
        a, b = self._check_name(res_name)
        data = getattr(self, b)
        popt = getattr(self, res_name)['popt']
        y_pred = main_sequence(getattr(self,a), *popt)
        residual_std = getattr(self, res_name)['residual_std']
        
        lower_boundary = y_pred - residual_std * sigma
        upper_boundray = y_pred + residual_std * sigma
        valid_indices = (data >= lower_boundary) & (data <= upper_boundray)
        
        # save sigma
        getattr(self, res_name)['sigma'] = sigma
        
        return valid_indices
    
    def fitting(self, amplitude, duration, peak_velocity, p0 = None):
        """ 
            All three variables are in numpy array 
            amplitude in deg
            peak_velocity in deg/s
            duration in ms
            p0: (2,1) entry. The first value is for amp vs. peak_vel, and the second value is for amp vs. duration
        """
        self.amplitude = np.array(amplitude)
        self.duration = np.array(duration)
        self.peak_velocity = np.array(peak_velocity)
        
        if p0 is None:
            p0 = [400, 120]
        
        # first filter with static numbers 
        mask = np.concatenate([(self.peak_velocity <= 1000) & (self.amplitude <= 60) & (self.duration <= 150) 
                               & ~np.isnan(self.peak_velocity) & ~np.isnan(self.amplitude) & ~np.isnan(self.duration)])
        
        # get amplitude vs. peak_velocity curve 
        popt, _= curve_fit(main_sequence, self.amplitude[mask], self.peak_velocity[mask], 
                    p0 = [p0[0], 5], 
                    maxfev=10000)
        self._save_fitting_res(popt, 'amp_vel_res', mask)
        
        # filter with the amplitude vs. peak_velocity result
        mask_vel = self._filtering('amp_vel_res')

        # get amplitude vs. duration curve 
        popt, _= curve_fit(main_sequence, self.amplitude[mask], self.duration[mask], 
                    p0 = [p0[1], 5], 
                    maxfev=10000)
        self._save_fitting_res(popt, 'amp_dur_res', mask)
    
        # filter with the amplitude vs. duration result
        mask_dur = self._filtering('amp_dur_res')
        
        filter_res = [mask & mask_vel & mask_dur]        
        filter_res = filter_res[0]
        self.filter_res = filter_res
        
        return filter_res
    
    def draw_figure(self, res_name, ax = None, x_max = 60, y_max = None):
        a, b = self._check_name(res_name)
        
        # generate a figure if ax is not given
        if ax is None:
            fig, ax = plt.subplots()
        
        # plot saccade data point
        x = getattr(self, a)
        y = getattr(self, b)
        mask = self.filter_res
        ax.scatter(x[~mask], y[~mask])
        ax.scatter(x[mask], y[mask], color='y')
        
        # plot fitting curve
        popt = getattr(self, res_name)['popt']
        residual_std = getattr(self, res_name)['residual_std']
        sigma = getattr(self, res_name)['sigma']
        r_squared = getattr(self, res_name)['r_squared']
        
        amp_x = np.linspace(0,60)
        y_pred = main_sequence(amp_x, *popt)
        ax.plot(amp_x, y_pred, 'r', label='Main sequence')
        ax.plot(amp_x, y_pred - residual_std * sigma, 'g', label = 'Lower bound')
        ax.plot(amp_x, y_pred + residual_std * sigma, 'g', label = 'Upper bound')
        
        # figure limits and labeling
        if b == "peak_velocity":
            y_name = "Peak Velocity [deg/s]"
            if y_max is None:
                y_max = 800
            ax.set_ylim([0, y_max])
        elif b == "duration": 
            y_name = "Duration [ms]"
            if y_max is None: 
                y_max = 150
            ax.set_ylim([0, y_max])
        ax.set_xlim([0, x_max])
        ax.set_xlabel('Amplitude [deg]')
        ax.set_ylabel(y_name)
        ax.set_title(f"R-squared: {np.round(r_squared,3)}")
        
    