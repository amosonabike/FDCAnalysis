#!/usr/bin/env python
# coding: utf-8

# Importing modules

# In[1]:


import numpy as np

import pandas as pd

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
from matplotlib import cm


import scipy
from scipy import stats
from scipy import signal
from scipy.optimize import curve_fit

import sklearn
from sklearn.linear_model import LinearRegression


# Setting plot style
# 

# In[2]:


figure_width = 4 * 90 / 25.4 #conversion to mm is 25.4
figure_height = 4 * 55.62 / 25.4 #conversion to mm is 25.4
figure_size = (figure_width, figure_height)

resolution = 600 #dpi

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

mpl.rcParams['axes.linewidth']=1.2



tick_size = 18
fontlabel_size = 18


params = {
    'lines.markersize' : 2,
    'axes.labelsize': fontlabel_size,
    'legend.fontsize': fontlabel_size,
    'xtick.labelsize': tick_size,
    'ytick.labelsize': tick_size,
    'figure.figsize': figure_size,
    'xtick.direction':     'in',     # direction: {in, out, inout}
    'ytick.direction':     'in',     # direction: {in, out, inout}
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.pad':  8,
    'ytick.major.pad':  8
}
plt.rcParams.update(params)


# In[3]:


def gaussian_rolling_average(array_like, rolling_window = 3):
    'takes array like and odd number for rolling window. Returns rolling average + standard deviation window uses padd to expand data at edges'

    clip = int((rolling_window -1)/2)
    y_padded = pd.Series(np.pad(array_like, (rolling_window//2, rolling_window-1-rolling_window//2), mode='edge'))
    y_smooth = y_padded.rolling(rolling_window, center=True,win_type= 'gaussian').mean(std = rolling_window)[clip:-clip]
    y_error = y_padded.rolling(rolling_window, center=True,win_type= 'gaussian').std(std = rolling_window)[clip:-clip]
    return y_smooth, y_error

def normalise_for_plotting(myrange):
    
    myrange = np.array(myrange)
    my_scaled_range = (myrange - myrange.min()) / (myrange.max() - myrange.min())
    
    return my_scaled_range


# In[4]:


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def get_file_locations(dir_path):
    
    resultsfiles = 0
    multiresults = 0
    
    results_path = None
    multi_path = None
    
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            
            if file.endswith("Results.txt"):
                print('Found results file: ',file) 
                print('\t at: ', os.path.join(root, file),'\n')
                results_path = os.path.join(root, file)
                resultsfiles += 1
                

            if file.endswith('separation.txt'):
                print('Found multiple exposure file: ',file)
                print('\t at: ', os.path.join(root, file),'\n')
                multi_path = os.path.join(root, file)
                multiresults+= 1
            assert resultsfiles < 2, 'Too many results files found!\n\tCheck directory path is correct and remove any extra results folders'
            assert multiresults < 2, 'Too many multiple exposure results files found!\n\tCheck directory path is correct and remove any extra results folders'
    return results_path, multi_path


# # Handling raw experimental data

# ## Extract conditions from file

# In[5]:


def get_conditions(df_experimental):
    'extract and describe temperature and RH data for the experiments, so that the detailed data can be disregarded'

    #average two measurements for each point
    temperature_data = pd.Series((df_experimental['Temp 1 / C'] + df_experimental['Temp 2 / C']) / 2 )

    temperature_average = temperature_data.mean()
    temperature_standard_deviation = temperature_data.std()
    temperature_max = temperature_data.max()
    temperature_min = temperature_data.min()
    temperature_gradient = (temperature_max - temperature_min) / df_experimental['Time since dispensed / s'].max()

    print('\nTemperature average, std. dev., min, max, gradient \n', temperature_average, temperature_standard_deviation, temperature_min, temperature_max, temperature_gradient)

    RH_data = df_experimental['Relative Humidity / %'].loc[df_experimental['Relative Humidity / %'] > -1]

    RH_average = RH_data.mean()
    RH_standard_deviation = RH_data.std()
    RH_max = RH_data.max()
    RH_min = RH_data.min()
    RH_gradient = (RH_max - RH_min) / df_experimental['Time since dispensed / s'].max()

    print('\nRH average, std. dev., min, max, gradient \n',RH_average, RH_standard_deviation, RH_min, RH_max, RH_gradient)
    
    
    fig, ax1 = plt.subplots()

    ax1.plot(RH_data, color = '#D81B60', label = '% RH')
    ax1.plot(temperature_data, color = '#1E88E5', label = 'Temperature / °C')



    ax1.set_xlabel('Data point / --')
    ax1.set_ylabel(' % RH or Temperature / °C')
 
    plt.legend()
    plt.show()
    
    return


# ## Calculate trajectory from FDC position parameters
# Normalising to starting position of x = 0.1, y = 1

# In[6]:


def get_trajectory(df_experimental):
    'sum x position data and translation stage data and then normalise both the x and y data to (0.1, 1) for plotting on log graph.'
    df_experimental['Droplet x position / mm'] = df_experimental['x Coordinate Centre of Mass Postion / mm'] + df_experimental['Step distance / mm'] + 0.1 - df_experimental['x Coordinate Centre of Mass Postion / mm'].iloc[0] - df_experimental['Step distance / mm'].iloc[0]

    df_experimental['Droplet position / mm'] = df_experimental['Droplet position / mm'] - df_experimental['Droplet position / mm'].iloc[0] + 1
    
    return df_experimental
    


# ## Get volume equivalent diamter

# In[7]:


def get_d_e(df_experimental):
    df_experimental['Volume equivalent diameter / um'] = ((6 * df_experimental['Volume / um^3'])/( np.pi)) ** (1./3.)
    #df_experimental.plot('Time since dispensed / s', ['Image diameter / um', 'Volume equivalent diameter / um'])
    return df_experimental


# ## Filter columns of interest to new df
# Selecting only columns of interest. The fundamental measurements are image size, volume, and position (x,y)

# ### Give columns some shorter names

# In[8]:


def get_fundamental_measurements(df_experimental):
    'extract only the column names of interest and rename them to something more sensible'

    df_selected = df_experimental.filter(['Time since dispensed / s', 'Droplet x position / mm', 'Droplet position / mm', 'Image diameter / um', 'Volume equivalent diameter / um'])


    # new names
    df_selected = df_selected.rename(columns = {'Time since dispensed / s' :  'Time_s',
                           'Droplet x position / mm' : 'x_position_mm' ,
                           'Droplet position / mm' : 'y_position_mm',
                           'Image diameter / um' : 'd_i_um',
                           'Volume equivalent diameter / um': 'd_e_um'})
    return df_selected



# # Cleaning Data

# ## Filter columns of interest to new df
# Selecting only columns of interest. The fundamental measurements are image size, volume, and position (x,y)

# ### Give columns some shorter names

# In[9]:


def get_fundamental_measurements(df_experimental):
    'extract only the column names of interest and rename them to something more sensible'

    df_selected = df_experimental.filter(['Time since dispensed / s', 'Droplet x position / mm', 'Droplet position / mm', 'Image diameter / um', 'Volume equivalent diameter / um'])


    # new names
    df_selected = df_selected.rename(columns = {'Time since dispensed / s' :  'Time_s',
                           'Droplet x position / mm' : 'x_position_mm' ,
                           'Droplet position / mm' : 'y_position_mm',
                           'Image diameter / um' : 'd_i_um',
                           'Volume equivalent diameter / um': 'd_e_um'})
    return df_selected



# ### Get data within user specified z score threshold
# Default value 1.
# Pass a dataframe to this function and get a 'clean' version back in return. Useful for groupby --> apply

# In[10]:


def get_fundamental_measurements(df_experimental):
    'extract only the column names of interest and rename them to something more sensible'

    df_selected = df_experimental.filter(['Time since dispensed / s', 'Droplet x position / mm', 'Droplet position / mm', 'Image diameter / um', 'Volume equivalent diameter / um'])


    # new names
    df_selected = df_selected.rename(columns = {'Time since dispensed / s' :  'Time_s',
                           'Droplet x position / mm' : 'x_position_mm' ,
                           'Droplet position / mm' : 'y_position_mm',
                           'Image diameter / um' : 'd_i_um',
                           'Volume equivalent diameter / um': 'd_e_um'})
    return df_selected



# ### Get data within user specified z score threshold
# Default value 1.
# Pass a dataframe to this function and get a 'clean' version back in return. Useful for groupby --> apply

# In[11]:


def get_data_within_z_score(df, z_score_threshold = 1):
    'Takes grouped object and returns df without outliers ( values outside z threshold)'

    
    df_z_score = df.drop('Time_s', axis = 1).apply(stats.zscore).abs()
    df_z_score.insert(0,'Time_s',0)
    df_clean = df[df_z_score < z_score_threshold].dropna()
    return df_clean
    


# ## Get cleaned data

# ### Group data by time values and get data within z score
# 

# In[12]:


def get_cleaned_data (df_dirty, z_score_threshold = 2):
    

    #group by time column
    grp_dirty = df_dirty.groupby('Time_s')

    #apply cleaning function
    df_cleaned = grp_dirty.apply(get_data_within_z_score, z_score_threshold)

    # reset column names correctly
    df_cleaned.reset_index(inplace=True, drop = True)

    return df_cleaned, z_score_threshold


# ## Calculate mean and std of data

# In[13]:


def get_mean_and_std(df_clean):
    
    'takes cleaned data and groups by time, then gets mean and std and returns these in new column names. Sets std of Nan to 0.'

    #calculate mean and std of data
        
    df_average = df_clean.groupby('Time_s').agg(['mean', 'std'])

    #removes extra header row created
    df_average.reset_index(level = 0, inplace=True)

    #flatten hierarchical indexing of columns to include value type in column name
    df_average.columns = ['_'.join(col).strip() for col in df_average.columns.values]
    df_average.rename(columns={"Time_s_": "Time_s"}, inplace = True)

    df_average.fillna(0, inplace = True)
    
    return df_average


# # Calculations on experimental data

# ## Calculating diameter squared values

# In[14]:


def get_diameter_squared (df):

    #image diameter
    df['d_i_2_um_2'] = df.d_i_um_mean ** 2
    df['d_i_2_err_um_2'] = 2 * df.d_i_um_mean * df.d_i_um_std

    #volumne equivalent diamter
    df['d_e_2_um_2'] = df.d_e_um_mean ** 2
    df['d_e_2_err_um_2'] = 2 * df.d_e_um_mean * df.d_e_um_std

    return df


# ## Calculating evaporation rate from de^2

# In[15]:


def get_differential (x_series, y_series):
    'used for calculating gradient of diameter data to identify linear periods'
    gradient_series = pd.Series(np.gradient(y_series, x_series))
    return gradient_series


# In[16]:


def get_evaporation_rate (df):
    df['evaporation_rate_um_2_per_s'] = -get_differential(df.Time_s, df.d_e_2_um_2 / 4)
    return df


# ## Using position data to calculate d_a and relaxation time

# ## Calculate velocity
# 

# In[17]:


def get_velocity (df):
    'differentiates position data for x and y'

    #calculates velocity as backwards difference between consectuive points. First point is NaN.
    df['y_velocity_mm_per_s'] = df.y_position_mm_mean.diff() / df.Time_s.diff()
    df['y_velocity_err_mm_per_s'] = df.y_position_mm_std.rolling(2).sum() / df.Time_s.diff()

    df['x_velocity_mm_per_s'] = df.x_position_mm_mean.diff() / df.Time_s.diff()
    df['x_velocity_err_mm_per_s'] = df.x_position_mm_std.rolling(2).sum() / df.Time_s.diff()
    return df


# ## Calculate realtive speed of particle to gas phase
# using vectors, for Reynolds number
# 
# v_rel = v_p - v_g
# 
# x relative velocity is simply x as there's no horizontal gas flow
# y relative velocity has already been calculated
# 
# Simply apply pythagoras theroem to get relative speed.
# Error calculated using partial differential method where dv_rel/dv_i =  v_i / v_rel

# In[18]:


def get_relative_speed (df):
    df['relative_speed_mm_per_s'] = np.sqrt((df.x_velocity_mm_per_s) ** 2 + (df.y_velocity_corrected_mm_per_s) ** 2)
    df['relative_speed_err_mm_per_s'] = df.x_velocity_err_mm_per_s * (df.x_velocity_mm_per_s / df.relative_speed_mm_per_s) + df.y_velocity_err_mm_per_s * (df.y_velocity_mm_per_s / df.relative_speed_mm_per_s)
    return df


# ## Calculate acceleration from relative velocity

# In[19]:


def get_acelleration (df):
    df['relative_acceleration_mm_per_s2']  = df.relative_speed_mm_per_s.diff() / df.Time_s.diff()
    return df


# ## Calculate particle Reynolds number
# 
# Re_p = (d_i * v_rel * rho_gas) / (dynamic_visc_gas)

# In[20]:


def get_reynolds_number (df):

    rho_air = 1.2041 #kgM^-3
    dynamic_visc_air = 1.81E-5 #kgm^-1s^-1

    df['Re_p'] = ((df.d_i_um_mean / 1000000) * (df.relative_speed_mm_per_s / 1000) * rho_air) / (dynamic_visc_air)
    df['Re_p_err'] = (rho_air / dynamic_visc_air) * ((df.d_i_um_std / 1000000) * (df.relative_speed_mm_per_s / 1000) + ((df.d_i_um_mean / 1000000) * (df.relative_speed_err_mm_per_s / 1000)))

    return df


# # Calculate aerodynamic diameter

# In[21]:


def get_aerodynamic_diameter (df):

    df['d_a_2_um_2'] =  1000000 ** 2 * ( 18 * (1.81E-5) * (df.y_velocity_corrected_mm_per_s / 1000) ) / (997 * 9.81)
    df['d_a_2_err_um_2'] = 1000000 ** 2 * ( 18 * (1.81E-5) * (df.y_velocity_err_mm_per_s / 1000) ) / (997 * 9.81)

    df['d_a_um'] = np.sqrt(df.d_a_2_um_2)
    df['d_a_err_um'] = 0.5 * (1 / np.sqrt(df.d_a_2_um_2)) * df.d_a_2_err_um_2

    if df.d_a_2_um_2.min() < 0:
        print( "Warning: some d_a^2 values < 0. Check gas flow values and corrected velocity")

    return df
    


# In[22]:


def get_density_ratio(df):
    df['density_ratio'] = df['d_a_2_um_2']/df['d_e_2_um_2']
    df['density_ratio_err'] = df.density_ratio * np.abs(df['d_e_2_err_um_2'] / df['d_e_2_um_2']) + np.abs(df['d_a_2_err_um_2'] / df['d_e_2_um_2'])
    return df


# # Thresholding data

# ## Get Stokes regime data
# This returns data within stokes regime and data not within

# In[23]:


def get_stokes_regime_data(df, stokes_threshold = 0.1):
    'selects data while stokes number is within specified limit. Returns data in and out of range specified and the threshold'

    df_in_range = df.loc[df['Re_p'] <= stokes_threshold]
    df_out_of_range = df.loc[df['Re_p'] >= stokes_threshold]
    return df_in_range, df_out_of_range, stokes_threshold


# ### Get period with acceleration greater than threshold for calculating relaxation time
# This returns data during the relaxation regime and the settling regime
# 

# In[24]:


def get_acceleration_period(df, reynolds_threshold = 3, acceleration_threshold = 100, stokes_threshold = 0.1):
    'Take a specified max Re_p limit and min deceleration value to select a deceleration period for calculating relaxation time. Takes the data with acceleration below specfied value as the settling regime. Returns the deceleration period data and data within the settling regime as well as the parameters that defined that range'

    print('\nRetrieving Relaxation Period')
    
    if reynolds_threshold == stokes_threshold:

        print('\tUsing Stokes treshold: ', stokes_threshold)
        condition = (np.abs(df.relative_acceleration_mm_per_s2) > acceleration_threshold) & (df.Re_p <= stokes_threshold) & ((df.Time_s) < 0.1)

    else:
        print('\tUsing custom Reynolds treshold: ', reynolds_threshold)
        condition = (np.abs(df.relative_acceleration_mm_per_s2) > acceleration_threshold) & (df.Re_p <= reynolds_threshold) & ((df.Time_s) < 0.1)

    #deceleration data selection
    df_slowing = df.loc[condition]

    #settling regime data selection
    settling_regime = ((np.abs(df.relative_acceleration_mm_per_s2) < acceleration_threshold) & (df.Re_p <= stokes_threshold)) | ((df.Time_s) > 0.1) 

    df_no_acceleration = df.loc[settling_regime]
    
    #removes very large values of evaporation rate from erroneously selected settling regime data
    df_no_acceleration = df_no_acceleration.loc[df_no_acceleration.evaporation_rate_um_2_per_s < df_no_acceleration.evaporation_rate_um_2_per_s.mean()*10]
    
    return df_slowing, df_no_acceleration, reynolds_threshold, acceleration_threshold
    


# In[ ]:





# ## Use relaxation period to calculate relaxation time

# ### Curve fitting to deceleration period

# In[25]:


#params = amplitude (approx initial value), relaxation time, vertical offset
def relaxation_func(x, a, Tau, c):
    return a * np.exp(-x / Tau) + c


# In[26]:


def get_relaxation_fits(df, relaxation_initial_guess = [100, 0.005, -10]):
    '''fits relative speedd (2D), x velocity and y velocity using initial guess. 
    Returns [[relative speed fit], [x fit], [yfit]]'''
    
    '''d_a^2 = 18 * viscosity * relaxation_time / (density * C_c)'''
    try:

        #relative speed to gas
        popt, pcov = curve_fit(relaxation_func, df.Time_s, df.relative_speed_mm_per_s, p0= relaxation_initial_guess)
        print('total fit: Initial amplitude = %5.3f, Tau= %5.4f, Vertical offset= %5.3f' % tuple(popt))
   
    except:
        popt, pcov = np.zeros(3), np.zeros(3)
    try:

        # x velocity
        xpopt, xpcov = curve_fit(relaxation_func, df.Time_s, df.x_velocity_mm_per_s, p0= relaxation_initial_guess)
        print('x fit: Initial amplitude = %5.3f, Tau= %5.4f, Vertical offset= %5.3f' % tuple(xpopt))

    except:
        xpopt, xpcov = np.zeros(3), np.zeros(3)
    try:

        # y velocity
        ypopt, ypcov = curve_fit(relaxation_func, df.Time_s, df.y_velocity_corrected_mm_per_s, p0= relaxation_initial_guess , maxfev = 5000)
        print('y fit: Initial amplitude = %5.3f, Tau= %5.4f, Vertical offset= %5.3f' % tuple(ypopt))
    
    except:
        ypopt, ypcov = np.zeros(3), np.zeros(3)
    
    return dict(relative = dict( parameters = popt, errors =  np.sqrt(np.diag(pcov))),
                x = dict( parameters = xpopt, errors =  np.sqrt(np.diag(xpcov))),
                y = dict( parameters = ypopt, errors =  np.sqrt(np.diag(ypcov))))


# In[ ]:





# # Correcting for gas flow in column

# In[27]:


def get_corrected_settling_velocity (df, gas_flow_rate = 0, gas_flow_correction_factor = 1):
    'Takes df with particle speed and removes gas flow speed. default gas flow = 0. corrects for gas flow to find true settling velocity'
    
    
    #column cross sectional area
    column_cross_sectional_area = (0.02 ** 2) #m^2

    #convert from sccm to m^3s^-1
    #divide by area
    gas_velocity = ((gas_flow_correction_factor * gas_flow_rate) / 60000000) / column_cross_sectional_area

    #subtract gas velocity from observed particle velocity
    df['y_velocity_corrected_mm_per_s'] = df.y_velocity_mm_per_s - gas_velocity * 1000

    #ignores early lifetime scattered data, looks for values below 0 as this translates into nagtive d_a^2
    if df.loc[df.Time_s > 0.1].y_velocity_corrected_mm_per_s.min() < 0:
        print('\nWarning: Min velocity is less than 0, check value of gas flow! \n\tMin value = ', df.y_velocity_corrected_mm_per_s.min())
    
    return df


# ## Correcting gas flow speed

# In[28]:


def get_gas_flow_correction_factor (df, gas_flow_experimental_value = 0, particle_density = 997, early_lifetime_threshold = 0.05):
    
    
    '''Receives settling regime data (or other if preferred) Takes early lifetime date from settling regime, but while evaporation may be assumed negligible.
    Iterates while the ratio (da^2 * rho_water) / (de^2 * rho_solution) is far from 1
    Returns gas flow corrrection factor and calculated actual experimental gas flow in column'''
    
    df_density_check = df.loc[df.Time_s <= early_lifetime_threshold].copy()
    
    if df_density_check.empty == True:

        
        print ('No settling regime data within first 0.05 s. Using first settling regime datapoint only\n',
              f'Datapoint time = {df.Time_s.iloc[0]}')
        df_density_check = df.head(1).copy()

    #initialise facotor as 1, meaning no error in gas flow
    gas_flow_corection_factor = 1

    #this ratio should be 1 for a system with no error in gas flow
    diameter_density_ratio = (df_density_check.d_e_2_um_2.mean() * 997) / (df_density_check.d_a_2_um_2.mean() * particle_density)
    print('Starting density ratio (da^2 * rho_water) / (de^2 * rho_solution): ', diameter_density_ratio)

    while np.abs(diameter_density_ratio - 1) > 0.001:

        diameter_density_ratio = (df_density_check.d_a_2_um_2.mean() * 997) / (df_density_check.d_e_2_um_2.mean() * particle_density)

        #make a small change to the correction factor based upon the current difference
        correction_change =  np.sign(diameter_density_ratio - 1) * 0.001
        gas_flow_corection_factor = gas_flow_corection_factor + correction_change

        #recalculate aerodynamic data based on correction factor
        df_density_check = get_corrected_settling_velocity(df_density_check, gas_flow_rate=gas_flow_experimental_value, gas_flow_correction_factor = gas_flow_corection_factor)
        df_density_check = get_aerodynamic_diameter(df_density_check)
        
        print('\r', 'Gas flow correction factor = ', gas_flow_corection_factor, end='')
        
        if gas_flow_corection_factor < 0.001:
            print("Could not find correction factor, setting to 0")
            gas_flow_corection_factor = 0
            break

    print('\nEstimated gas flow in column (sccm): ', gas_flow_corection_factor * gas_flow_experimental_value)
    print('Final density ratio (da^2 * rho_water) / (de^2 * rho_solution): ', diameter_density_ratio)

    
    return gas_flow_corection_factor, gas_flow_corection_factor * gas_flow_experimental_value


# # Getting Linear fit to d^2

# ### Fitting a line to linear section of data
# 

# In[29]:



def get_line_fit(x_series, y_series, r_sq_threshold = 0.99):
    'Takes series, specified from df. eg time series and diameter^2 data. Returns: fit_intercept, fit_evaporation_rate, r_sqared, fitted_window (t_min, t_max).'

    # define model to use
    line_fit = LinearRegression(fit_intercept=True)  
    
    #starting r^2 value of something small
    r_sq = 0
    
    while r_sq < r_sq_threshold:

        x_data = np.array(x_series).reshape(-1, 1)
        y_data = np.array(y_series).reshape(-1, 1)

        # fits to selected d^2 value 
        model = line_fit.fit(x_data, y_data)  


        d2_pred = line_fit.predict(x_data)
        r_sq = model.score(x_data, y_data)

        if r_sq < r_sq_threshold:


            n = 1
            x_series = x_series.drop(x_series.tail(n).index,inplace=False) # drop last n rows
            y_series = y_series.drop(y_series.tail(n).index,inplace=False) # drop last n rows


        else:
            print('good fit')
            plt.scatter(x_series,y_series)
            plt.plot(x_data, d2_pred, linestyle=':')
            plt.show()
            
            print(f'Fitted intial size / µm\t\t\t{round(np.sqrt(model.intercept_[0]),3)}\nCalculated eavporation rate / µm^2/s\t{round(-model.coef_[0][0],3)}\nR squared\t\t\t\t{round(r_sq,3)}')

            fit_params = dict(intercept = model.intercept_[0],
                              gradient = model.coef_[0][0],
                              r_sq = r_sq,
                              t_min = x_series.min(),
                              t_max = x_series.max())
            
    return fit_params


# # Running multiple functions

# In[30]:


def import_clean_data(df_import, chosen_z_score_threshold = 3):
    'receives imported data frame and performs all functions prior to calculation of aerodynamic dyameter. Therefore is not influenced by gas flow correction factor.'
    
    print('Experimental Conditions:\n')
    get_conditions(df_import)

    #calculate and normalise trajectory
    df_import = get_trajectory(df_import)

    #calculate d_e
    df_import = get_d_e(df_import)

    #select useful columns of data
    df_pre_clean = get_fundamental_measurements(df_import)

    #remove outliers
    df_clean, z_score_threshold = get_cleaned_data(df_pre_clean)    
    
    #calculate means and std devs
    df_average = get_mean_and_std(df_clean)

    #calculating diameter squared values
    df_average = get_diameter_squared(df_average)

    #calculate diameter squared
    df_average = get_evaporation_rate(df_average)

    #calculate velocity
    df_average = get_velocity(df_average)
    
    return df_pre_clean, df_clean, df_average, z_score_threshold
    
    


# # Gas flow dependent functions

# In[31]:


def gas_flow_dependent_calculations(df_average, gas_flow_correction = 1, gas_flow_experimental_value = 0):
    

    #calculate particle settling velocity
    df_average = get_corrected_settling_velocity(df_average, gas_flow_experimental_value, gas_flow_correction)

    #calculate relative speed
    df_average = get_relative_speed(df_average)

    #calculate acceleration
    df_average = get_acelleration(df_average)

    #calculate reynolds number
    df_average = get_reynolds_number(df_average)

    #calculate aerodynamic diameter
    df_average = get_aerodynamic_diameter(df_average)
    
    #calculate density ratio
    df_average = get_density_ratio(df_average)

    #find stokes regime
    df_stokes, df_not_stokes, stokes_threshold = get_stokes_regime_data(df_average)

    #find deceleration period and settling regime
    df_deceleration_period, df_settling_regime, reynolds_threshold, aceleration_threshold = get_acceleration_period(df_average)
    
    #perform fits for relaxation period
    fits_parameters = get_relaxation_fits(df_deceleration_period)
    
    return df_average, df_stokes, df_not_stokes, stokes_threshold, df_deceleration_period, df_settling_regime, reynolds_threshold, aceleration_threshold, gas_flow_experimental_value, fits_parameters


# In[32]:


def run_all_analysis(filepath, z_score_threshold = 3, gas_flow_rate = 0, sample_density =997):
    '''docstring here'''
    
    #get data from file
    df_import = pd.read_csv(filepath, delimiter='\t', header=0)
    
    #clean data and reduce to only useful columns
    df_pre_clean, df_clean, df_average, z_score_threshold = import_clean_data(df_import,1)
    
    df_average, df_stokes, df_not_stokes, stokes_threshold, df_deceleration_period, df_settling_regime, reynolds_threshold, aceleration_threshold, gas_flow_experimental_value, relaxation_fit_parameters = gas_flow_dependent_calculations(df_average,1 , gas_flow_rate)

    #plot_cleaned_data (df_average, df_pre_clean, df_clean, z_score_threshold)
        
    if gas_flow_rate > 0:
        #calculate gas flow correction factor
        gas_flow_correcction_factor, corrected_gas_flow = get_gas_flow_correction_factor(df_settling_regime, gas_flow_rate,particle_density= sample_density)
    
    else:
        gas_flow_correcction_factor = 1
        pass
    
    actual_gas_flow_rate = gas_flow_correcction_factor * gas_flow_rate
    
    df_average, df_stokes, df_not_stokes, stokes_threshold, df_deceleration_period, df_settling_regime, reynolds_threshold, aceleration_threshold, gas_flow_experimental_value, relaxation_fit_parameters = gas_flow_dependent_calculations(df_average, gas_flow_correcction_factor, gas_flow_experimental_value=gas_flow_rate)

    #plot_trajectory(df_average, df_stokes, df_not_stokes, df_deceleration_period, df_settling_regime, reynolds_threshold, stokes_threshold, 0,1,1,1,1)
    
    #plot_relaxation(df_deceleration_period, fits_parameters)
    
    #plot_da_de_comparison (df_average, df_stokes, df_not_stokes, df_deceleration_period, df_settling_regime, reynolds_threshold, stokes_threshold)

    #perform linear fit to settling regime data
    linear_fit_params = get_line_fit(df_settling_regime.Time_s, df_settling_regime.d_e_2_um_2)

    
    data_dict = dict(stokes = df_stokes,
                     not_stokes = df_not_stokes,
                     relaxation = df_deceleration_period,
                     settling = df_settling_regime,
                     all = df_average)
    
    parameters_dict = dict(SampleDensity = sample_density,
                           NominalGasFlow = gas_flow_rate,
                           CalculatedGasFlow = actual_gas_flow_rate,
                           FlowCorrectionFactor = gas_flow_correcction_factor,
                           StokesThreshold = stokes_threshold,
                           ReynoldsTreshold = reynolds_threshold,
                           LinearFit = linear_fit_params,
                           RelaxationFits = relaxation_fit_parameters)
    
    return  parameters_dict, data_dict
    


# In[33]:


def import_multiple_exposure_data(file_location):
    
    column_names = ['x_image', 'y_image', 'dx', 'dy', 'time_s', 'measurement_number', 'pixel_size_um']

    df = pd.read_csv(file_location, '\t',header=None, names = column_names)

    df['displacement_m'] = np.nan
    df['velocity_ms'] = np.nan
    df['velocity_ms_err'] = np.nan
    df['da_2'] = np.nan
    df['da_2_err'] = np.nan
    df['da'] = np.nan
    df['da_err'] = np.nan
        
    return df


# In[34]:


def sort_measurements(df):
    return df.sort_values('y_image')


# In[35]:


def do_calculations(df, delta_time, flow_speed):
        
    df.displacement_m = np.sqrt(df.x_image.diff().values ** 2 + df.y_image.diff().values ** 2) * df.pixel_size_um.values * 1e-6
    
    
    df.velocity_ms = df.displacement_m.copy()/delta_time - flow_speed/(60000000 * (0.02 ** 2))
    df.velocity_ms_err = np.sqrt(2 * (2e-6 *df.pixel_size_um.values) ** 2) / delta_time

    df.da_2 = ( 18 * (1.81E-5) * (df.velocity_ms.values) ) / (997 * 9.81)
    df.da_2_err = ( 18 * (1.81E-5) * (df.velocity_ms_err.values) ) / (997 * 9.81)

    df.da = np.sqrt(df.da_2.values)
    df.da_err = 0.5 * (1 / np.sqrt(df.da_2.values)) * df.da_2_err.values
    
    return df


# In[36]:


def function_over_time(df, time_diff, gas_flow):
    df_with_calcs = df.groupby('measurement_number').apply(do_calculations, time_diff, gas_flow)
    return df_with_calcs
    


# In[37]:


def time_sort(df):
    return df.groupby('measurement_number').apply(sort_measurements)


# In[38]:


def modified_z_score(df, thresh: float=3.5):
    '''https://www.statology.org/modified-z-score-excel/'''
    return 0.6745* (df - df.median())/df.mad()


# In[39]:


def median_comparison(df):
    '''compares magniture of value with median '''
    return df/df.median()
    


# In[40]:


def multi_remove_outliers(df, z_score_threshold = 3, iqr_threshold = 1, mod_threshold = 3.5, median_multiple = 2):
    '''
    Takes df and returns df without outliers (values outside z threshold or iqr or modified z score)
    Includes a median multiple cut off, for dropping values much larger than median value.
    '''

    df = df.drop(df[df.da_2.abs() > abs(df.da_2.median() * median_multiple)].index)
    
    df_z_score = df.apply(stats.zscore).abs()
    df_mod_z_score = df.apply(modified_z_score).abs()
    df_iqr_score = df.apply(stats.iqr).abs()
    
    df_clean = df[df_z_score.da_2 < z_score_threshold]
    df_clean = df[df_z_score.da_2 < iqr_threshold]
    df_clean = df[df_z_score.da_2 < mod_threshold]
    
    
    
    
    '''if (df_clean.da_2.mad()/1e-12) > 200:
        mod_threshold = 1
        print (r'handling wide distribution')
        print ('Using mod_threshold of: ', mod_threshold)
        df_clean = df[df_z_score.da_2 < mod_threshold]'''

    
    return df_clean
    


# In[41]:


def multi_exposure_averaging(df):
    return df.groupby('measurement_number').agg('mean')


# In[42]:


def get_multiple_exposure_data(path, flow, z_thresh = 3, iqr_thresh = 1.5, mod_thresh = 3.5, median_mult = 3):
    
    df_me = import_multiple_exposure_data(path)

    time_separation = float(path[path.find(' at ')+4:path.find(' us ')])/1e6

    #sorting   
    df_me = df_me.groupby('time_s').apply(time_sort).reset_index(level = 1, drop = 1).reset_index(level = 1, drop = 1)
    df_me = df_me.reset_index(level = 'time_s', drop = 1)

    #applying functions to grouped data
    df_processed = df_me.groupby('time_s').apply(function_over_time, time_separation, flow).reset_index(level = 1)

    df_processed.reset_index(level=0, drop = True, inplace = True)
    df_processed.drop('level_1', axis = 1, inplace= True)
    df_processed = df_processed[df_processed['velocity_ms'].notna()]#df_processed.dropna(inplace = True)

    #cleaning
    df_processed_clean = df_processed.groupby('time_s').apply(multi_remove_outliers, z_thresh, iqr_thresh, mod_thresh, median_mult)
    df_processed_clean.reset_index(drop = True, inplace=True)

    df_multi_data = df_processed_clean[df_processed_clean['velocity_ms'].notna()].groupby('time_s').apply(multi_exposure_averaging)

    time_points = df_multi_data.time_s.unique()
    violin_plot_data = [1e12 * df_multi_data.da_2.loc[df_multi_data.time_s == time].values for time in time_points]

    df_averaged = df_processed_clean.groupby('time_s').agg('mean').reset_index()

    return dict(timedata = time_points,
                violindata = violin_plot_data,
                multipleexposures = df_multi_data,
                multiexposureaveraged = df_processed_clean,
                average = df_averaged)


# In[43]:


def generate_plots(dict_of_dfs,exp_parameters, colour = 'dodgerblue', multi_exposure_dict = None, rolling_window = 3):

    
    #get colourmap and norm 
    bounds = np.power(10.0, np.arange(-4, 2))
    ncolors = len(bounds) -1
    cmap = cm.get_cmap('winter', ncolors) # Colour map (there are many others)
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=ncolors)
    alpha = 0.5
    mapper = cm.ScalarMappable(norm=norm, cmap= cmap)
    
    alt_colour = adjust_lightness(colour,0.2)
    mid_colour = adjust_lightness(colour, 0.6)
    markersize = 10
    ebarthickness = 1.5
    ecapsize = 5
    violinalpha = 0.6

    error_kwargs = {"zorder":0}
    
    fig_full_trajectory, ax = plt.subplots(figsize = figure_size)
    
    
    #Plot Trajectory
    
    df_key = 'all'
    
    ax.plot(gaussian_rolling_average(dict_of_dfs[df_key].x_position_mm_mean, rolling_window)[0],
            gaussian_rolling_average(dict_of_dfs[df_key].y_position_mm_mean, rolling_window)[0],
            color = 'k', linewidth = 0.5)
    
    #loop over each data point to plot
    for x, y, ex, ey, color in zip(dict_of_dfs[df_key].x_position_mm_mean,
                                   dict_of_dfs[df_key].y_position_mm_mean,
                                   dict_of_dfs[df_key].x_position_mm_std,
                                   dict_of_dfs[df_key].y_position_mm_std,
                                   np.array([(mapper.to_rgba(v)) for v in dict_of_dfs[df_key].Time_s])):

        ax.errorbar(x, y, ey, ex, alpha = alpha, lw=1, capsize=markersize, color=color, **error_kwargs)
        
        
    #Plot stokes regime
    df_key = 'stokes'
    
    ax.scatter(dict_of_dfs[df_key].x_position_mm_mean,
               dict_of_dfs[df_key].y_position_mm_mean, 
               c = dict_of_dfs[df_key].Time_s,
               alpha = alpha,
               s = markersize * dict_of_dfs[df_key].d_i_um_mean ** 3 / 1000, 
               cmap = cmap,
               norm= norm,
               label = 'Stokes regime')
    
    # plot settling
    df_key = 'settling'
    ax.scatter(dict_of_dfs[df_key].x_position_mm_mean,
               dict_of_dfs[df_key].y_position_mm_mean,
               c = 'k', marker = 'x', s = 2 * markersize, label = 'Settling regime')
    


    
    # Plot Relaxation Period
    df_key = 'relaxation'
    Re_thresh = exp_parameters['ReynoldsTreshold']
    ax.scatter(dict_of_dfs[df_key].x_position_mm_mean,
               dict_of_dfs[df_key].y_position_mm_mean,
               c = 'r', s = 5 * markersize, label = f'Deceleration period Re_p < {Re_thresh}')     

    
    #getting log colorbar
    cbar = plt.colorbar(cm.ScalarMappable(norm, cmap), label = 'Time / s',)
    cbar.ax.set_yticklabels([f'$10^{{{np.log10(b):.0f}}}$' for b in bounds])
    
    #plot scales
    
    if dict_of_dfs['all'].y_position_mm_mean.min() < 0.9:
        ymin = dict_of_dfs['all'].y_position_mm_mean.min() - 0.1
    else:
        ymin = 0.9
        
    ax.set_ylim(dict_of_dfs['all'].y_position_mm_mean.max()+10, ymin)
    ax.set_yscale('log')
    ax.set_xlim(0.09, dict_of_dfs['all'].x_position_mm_mean.max()+2)
    ax.set_xscale('log')

    ax.set_xlabel('Horizontal position / mm')
    ax.set_ylabel('Vertical position / mm')
    plt.show()
    
    
    fig_diameter, ax = plt.subplots(figsize = figure_size)
    
    # plot diameter
    df_key = 'settling'
    
    
    ax.plot(gaussian_rolling_average(dict_of_dfs[df_key].Time_s, rolling_window)[0],
            gaussian_rolling_average(dict_of_dfs[df_key].d_e_um_mean, rolling_window)[0],
            color = colour)

    ax.plot(gaussian_rolling_average(dict_of_dfs[df_key].Time_s)[0],
            gaussian_rolling_average(dict_of_dfs[df_key].d_a_um)[0],
            color = alt_colour)
    
    ax.errorbar(dict_of_dfs[df_key].Time_s,
                dict_of_dfs[df_key].d_e_um_mean,
                dict_of_dfs[df_key].d_e_um_std,
                color = colour,
                ms = markersize,
                fmt = 's',
                elinewidth=ebarthickness,
                capsize=ecapsize, 
                capthick=ebarthickness,
                label = 'd$_v$')
    
    ax.errorbar(dict_of_dfs[df_key].Time_s,
                dict_of_dfs[df_key].d_a_um,
                dict_of_dfs[df_key].d_a_err_um,
                color = alt_colour,
                ms = markersize,
                fmt = 'o',
                elinewidth=ebarthickness,
                capsize=ecapsize, 
                capthick=ebarthickness,
                label = 'd$_a$')
    
    ax.legend()
    ax.set_ylim(0)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Diameter / µm')
    plt.show()
    
    fig_diameter_sq, ax = plt.subplots(figsize = figure_size)
    
    # plot diameter squared
    df_key = 'settling'
    
    
    ax.plot(gaussian_rolling_average(dict_of_dfs[df_key].Time_s, rolling_window)[0],
            gaussian_rolling_average(dict_of_dfs[df_key].d_e_2_um_2, rolling_window)[0],
            color = colour)


    
    ax.errorbar(dict_of_dfs[df_key].Time_s,
                dict_of_dfs[df_key].d_e_2_um_2,
                dict_of_dfs[df_key].d_e_2_err_um_2,
                color = colour,
                ms = markersize,
                fmt = 's',
                elinewidth=ebarthickness,
                capsize=ecapsize, 
                capthick=ebarthickness,
                label = 'd$_v^2$')
    
    ax.plot(gaussian_rolling_average(dict_of_dfs[df_key].Time_s, rolling_window)[0],
            gaussian_rolling_average(dict_of_dfs[df_key].d_a_2_um_2, rolling_window)[0],
            color = alt_colour)

    ax.errorbar(dict_of_dfs[df_key].Time_s,
        dict_of_dfs[df_key].d_a_2_um_2,
        dict_of_dfs[df_key].d_a_2_err_um_2,
        color = alt_colour,
        ms = markersize,
        fmt = 'o',
        elinewidth=ebarthickness,
        capsize=ecapsize, 
        capthick=ebarthickness,
        label = 'd$_a^2$')
    
    ax.legend()
    
    ax.set_ylim(0)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Diameter Squared / µm$^2$')
    
    plt.show()
    

    
    if multi_exposure_dict == None:
        pass

    else:
        df_multi_period = pd.merge(pd.DataFrame(multi_exposure_dict['timedata'],columns = ['time']),
                                   dict_of_dfs['settling'],
                                   left_on='time', right_on='Time_s')
                       
        fig_multi_diameter_sq, ax = plt.subplots(figsize = figure_size) 
        ax.plot(gaussian_rolling_average(dict_of_dfs[df_key].Time_s, rolling_window)[0],
                gaussian_rolling_average(dict_of_dfs[df_key].d_e_2_um_2, rolling_window)[0],
                color = colour,
                zorder = 0)


    
        ax.errorbar(dict_of_dfs[df_key].Time_s,
                    dict_of_dfs[df_key].d_e_2_um_2,
                    dict_of_dfs[df_key].d_e_2_err_um_2,
                    color = colour,
                    ms = markersize,
                    fmt = 's',
                    elinewidth=ebarthickness,
                    capsize=ecapsize, 
                    capthick=ebarthickness,
                    label = 'd$_v^2$',
                    zorder = 0)
        
        ax.plot(gaussian_rolling_average(dict_of_dfs[df_key].loc[dict_of_dfs[df_key].Time_s < multi_exposure_dict['timedata'].min()].Time_s, rolling_window)[0],
                gaussian_rolling_average(dict_of_dfs[df_key].loc[dict_of_dfs[df_key].Time_s < multi_exposure_dict['timedata'].min()].d_a_2_um_2, rolling_window)[0],
                color = alt_colour)
        
        ax.errorbar(dict_of_dfs[df_key].loc[dict_of_dfs[df_key].Time_s < multi_exposure_dict['timedata'].min()].Time_s,
            dict_of_dfs[df_key].loc[dict_of_dfs[df_key].Time_s < multi_exposure_dict['timedata'].min()].d_a_2_um_2,
            dict_of_dfs[df_key].loc[dict_of_dfs[df_key].Time_s < multi_exposure_dict['timedata'].min()].d_a_2_err_um_2,
            color = alt_colour,
            ms = markersize,
            fmt = 'o',
            elinewidth=ebarthickness,
            capsize=ecapsize, 
            capthick=ebarthickness,
            label = 'd$_a^2$')
        
        
        #multiple exposure d_a
        '''ax.scatter(multi_exposure_dict['multipleexposures'].time_s,
                    multi_exposure_dict['multipleexposures'].da_2/1e-12,
                    s =  10,
                    marker = 'x',
                    c = alt_colour,
                    label = 'Multiple Exposure d$_a^2$')'''

        ax.plot(gaussian_rolling_average(multi_exposure_dict['average'].time_s, rolling_window)[0],
                gaussian_rolling_average(multi_exposure_dict['average'].da_2/1e-12, rolling_window)[0],
                color = alt_colour,
                lw = 0.5)

        violin_parts = ax.violinplot(multi_exposure_dict['violindata'],
                                     multi_exposure_dict['timedata'],
                                     widths=0.01,
                                     showmeans = 1,
                                     showmedians = 0,)
                                     #quantiles = len(multi_exposure_dict['violindata'])*[[0.25, 0.75]],)

        # Make all the violin statistics marks red:
        for partname in ('cbars','cmins','cmaxes', 'cmeans', 'cmedians', 'cquantiles'): #,'cmeans','cmedians'):
            if partname in violin_parts:
                vp = violin_parts[partname]
                vp.set_edgecolor(alt_colour)
                vp.set_linewidth(0.5)

        # Make the violin body blue with a red border:
        for vp in violin_parts['bodies']:
            vp.set_facecolor(alt_colour)
            vp.set_edgecolor(alt_colour)
            vp.set_linewidth(0.5)
            vp.set_alpha(violinalpha)
            
    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()
    # manually define a new patch 
    patch = mpatches.Patch(color=alt_colour, alpha = violinalpha, label='Multiple Exposure d$_a^2$')
    # handles is a list, so append manual patch
    handles.append(patch) 
        
    
    ax.legend(handles = handles)
    
    ax.set_ylim(0)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Diameter Squared / µm$^2$')
    plt.show()
    
    if multi_exposure_dict == None:
        pass

    else:
        fig_small_multi_diameter_sq, ax = plt.subplots(figsize = figure_size) 
        ax.plot(gaussian_rolling_average(dict_of_dfs[df_key].Time_s, rolling_window)[0],
                gaussian_rolling_average(dict_of_dfs[df_key].d_e_2_um_2, rolling_window)[0],
                color = colour,
                zorder = 0)


    
        ax.errorbar(dict_of_dfs[df_key].Time_s,
                    dict_of_dfs[df_key].d_e_2_um_2,
                    dict_of_dfs[df_key].d_e_2_err_um_2,
                    color = colour,
                    ms = markersize,
                    fmt = 's',
                    elinewidth=ebarthickness,
                    capsize=ecapsize, 
                    capthick=ebarthickness,
                    label = 'd$_v^2$',
                    zorder = 0)
    

        ax.plot(gaussian_rolling_average(multi_exposure_dict['average'].time_s, rolling_window)[0],
                gaussian_rolling_average(multi_exposure_dict['average'].da_2/1e-12, rolling_window)[0],
                color = alt_colour,
                lw = 0.5)

        violin_parts = ax.violinplot(multi_exposure_dict['violindata'],
                                     multi_exposure_dict['timedata'],
                                     widths=0.01,
                                     showmeans = 1,
                                     showmedians = 0,)
                                     #quantiles = len(multi_exposure_dict['violindata'])*[[0.25, 0.75]],)

        # Make all the violin statistics marks red:
        for partname in ('cbars','cmins','cmaxes', 'cmeans', 'cmedians', 'cquantiles'): #,'cmeans','cmedians'):
            if partname in violin_parts:
                vp = violin_parts[partname]
                vp.set_edgecolor(alt_colour)
                vp.set_linewidth(0.5)

        # Make the violin body blue with a red border:
        for vp in violin_parts['bodies']:
            vp.set_facecolor(alt_colour)
            vp.set_edgecolor(alt_colour)
            vp.set_linewidth(0.5)
            vp.set_alpha(violinalpha)
            
    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()
    # manually define a new patch 
    patch = mpatches.Patch(color=alt_colour, alpha = violinalpha, label='Multiple Exposure d$_a^2$')
    # handles is a list, so append manual patch
    handles.append(patch) 
        
    
    ax.legend(handles = handles)
    
    ax.set_ylim(0,max(multi_exposure_dict['violindata'], key=tuple).max() + 100)
    ax.set_xlim(multi_exposure_dict['timedata'].min() - 0.05, multi_exposure_dict['timedata'].max() + 0.05)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Diameter Squared / µm$^2$')
    plt.show()
    
    fig_density_ratio, ax = plt.subplots(figsize = figure_size)
    
    # plot density ratio 
    df_key = 'settling'
    

    ax.plot(gaussian_rolling_average(dict_of_dfs[df_key].Time_s, rolling_window)[0],
            gaussian_rolling_average(dict_of_dfs[df_key].density_ratio, rolling_window)[0],
            color = mid_colour)
    
    ax.errorbar(dict_of_dfs[df_key].Time_s,
                dict_of_dfs[df_key].density_ratio,
                dict_of_dfs[df_key].density_ratio_err,
                color = mid_colour,
                ms = markersize,
                fmt = 'h',
                elinewidth=ebarthickness,
                capsize=ecapsize, 
                capthick=ebarthickness,
                )
    
    ax.axhline(1, linewidth = 0.5, color = 'k')
    
    ax.set_ylim(0)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('d$_a^2$ / d$_e^2$')
    plt.show()
    
    if multi_exposure_dict == None:
        pass

    else:
        fig_multi_density_ratio, ax = plt.subplots(figsize = figure_size)

        # plot density ratio 
        df_key = 'settling'


        ax.plot(gaussian_rolling_average(dict_of_dfs[df_key].loc[dict_of_dfs[df_key].Time_s < multi_exposure_dict['timedata'].min()].Time_s, rolling_window)[0],
                gaussian_rolling_average(dict_of_dfs[df_key].loc[dict_of_dfs[df_key].Time_s < multi_exposure_dict['timedata'].min()].density_ratio, rolling_window)[0],
                color = mid_colour)

        ax.errorbar(dict_of_dfs[df_key].loc[dict_of_dfs[df_key].Time_s < multi_exposure_dict['timedata'].min()].Time_s,
                    dict_of_dfs[df_key].loc[dict_of_dfs[df_key].Time_s < multi_exposure_dict['timedata'].min()].density_ratio,
                    dict_of_dfs[df_key].loc[dict_of_dfs[df_key].Time_s < multi_exposure_dict['timedata'].min()].density_ratio_err,
                    color = mid_colour,
                    ms = markersize,
                    fmt = 'h',
                    elinewidth=ebarthickness,
                    capsize=ecapsize, 
                    capthick=ebarthickness,)
        
        
        ax.plot(gaussian_rolling_average(multi_exposure_dict['average'].time_s, rolling_window)[0],
                gaussian_rolling_average((multi_exposure_dict['average'].da_2/1e-12)/df_multi_period.d_e_2_um_2, rolling_window)[0],
                color = mid_colour)
        
        

        ax.axhline(1, linewidth = 0.5, color = 'k')

        ax.set_ylim(0)
        ax.set_xlabel('Time / s')
        ax.set_ylabel('d$_a^2$ / d$_e^2$')
        plt.show()
        
        
        
        
    # plot evaporation rate
    df_key = 'settling'
    
    fig_evap_rate, ax = plt.subplots(figsize = figure_size)

    ax.plot(gaussian_rolling_average(dict_of_dfs[df_key].Time_s, rolling_window)[0],
            gaussian_rolling_average(dict_of_dfs[df_key].evaporation_rate_um_2_per_s, rolling_window)[0],
            color = mid_colour)
    
    ax.scatter(dict_of_dfs[df_key].Time_s,
                dict_of_dfs[df_key].evaporation_rate_um_2_per_s,
                color = mid_colour,
                s = markersize * 10,
                marker = 'h',)
    
    
    ax.set_ylim(0)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Evaporation Rate / µm$^2$/s')
    plt.show()
    

    
    return


def plot_RH_comparison(data_dicts, RHs, multi_data_dicts = None, rolling_window = 3, colour_low_RH = 'red', colour_high_RH = 'blue'):
    '''
    For plotting results of experiements over a range of RH values.
    
    Takes list of output dictionaries from single exposusre analysis,
          list of RH values corresponding to experiemnts/output dicts
          list (optional) of multiple expsure results dictionaries. Default = None
          rolling window for plotting rolling average
          colour extremes for colour mapping across RHs
          
          returns nothing, shows plots.
    '''
    
    assert len(data_dicts) == len(RHs), 'Lengths of inputs are not correct. They must be the same.'
    
    if multi_data_dicts != None:
        assert len(data_dicts) == len(multi_data_dicts), 'Lengths of inputs are not correct. They must be the same.'

    
    cm1 = colors.LinearSegmentedColormap.from_list("RedBlue",[colour_low_RH, colour_high_RH])
    colours = cm1(normalise_for_plotting(RHs))
    
    df_key = 'settling'
    
    #plot d_v^2
    fig_RH_dv_comparison, ax = plt.subplots()

    for i, (df, rh) in enumerate(zip(data_dicts, RHs)):    

        ax.plot(gaussian_rolling_average((df[df_key].Time_s), rolling_window)[0],
                 gaussian_rolling_average(df[df_key].d_e_2_um_2, rolling_window)[0],
                 label = str(rh)+ ' % RH',
                 color = colours[i])

        ax.scatter((df[df_key].Time_s),
                     df[df_key].d_e_2_um_2,
                     color = colours[i], s = 10)

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('d$_v^2$ / µm$^2$')
    ax.legend()
    plt.show()
    
    
    #plot d_a^2
    for i, (df, rh) in enumerate(zip(data_dicts, RHs)):    

            ax.plot(gaussian_rolling_average((df[df_key].Time_s), rolling_window)[0],
                     gaussian_rolling_average(df[df_key].d_e_2_um_2, rolling_window)[0],
                     label = str(rh)+ ' % RH',
                     color = colours[i])

            ax.scatter((df[df_key].Time_s),
                         df[df_key].d_e_2_um_2,
                         color = colours[i], s = 10)

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('d$_v^2$ / µm$^2$')
    ax.legend()
    plt.show()

    fig_RH_dv_comparison, ax = plt.subplots()

    for i, (df, rh) in enumerate(zip(data_dicts, RHs)):    

        ax.plot(gaussian_rolling_average((df[df_key].Time_s), rolling_window)[0],
                 gaussian_rolling_average(df[df_key].d_a_2_um_2, rolling_window)[0],
                 label = str(rh)+ ' % RH',
                 color = colours[i])

        ax.scatter((df[df_key].Time_s),
                     df[df_key].d_a_2_um_2,
                     color = colours[i], s = 10)

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('d$_a^2$ / µm$^2$')
    ax.legend()
    plt.show()
    
    
    #plot d_v^2 normalised
    fig_RH_dv_comparison_normalised, ax = plt.subplots()

    for i, (df, rh) in enumerate(zip(data_dicts, RHs)):

        ax.plot(gaussian_rolling_average((df[df_key].Time_s - df[df_key].Time_s.min())/df[df_key].d_e_2_um_2.head(1).values[0], rolling_window)[0],
                 gaussian_rolling_average(df[df_key].d_e_2_um_2/df[df_key].d_e_2_um_2.head(1).values[0], rolling_window)[0],
                 label = str(rh)+ ' % RH',
                 color = colours[i])

        ax.scatter((df[df_key].Time_s - df[df_key].Time_s.min())/df[df_key].d_e_2_um_2.head(1).values[0],
                     df[df_key].d_e_2_um_2/df[df_key].d_e_2_um_2.head(1).values[0],
                     color = colours[i], s = 10)

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel('(Time / d$_0^2$) / (s  /µm$^2$ )')
    ax.set_ylabel('d$^2$ / d$_0^2$')
    ax.legend()
    plt.show()

    #plot d_a^2 normalised
    fig_RH_da_comparison_normalised, ax = plt.subplots()

    for i, (df, rh) in enumerate(zip(data_dicts, RHs)):

        ax.plot(gaussian_rolling_average((df[df_key].Time_s - df[df_key].Time_s.min())/df[df_key].d_a_2_um_2.head(1).values[0], rolling_window)[0],
                 gaussian_rolling_average(df[df_key].d_a_2_um_2/df[df_key].d_a_2_um_2.head(1).values[0], rolling_window)[0],
                 label = str(rh)+ ' % RH',
                 color = colours[i])

        ax.scatter((df[df_key].Time_s - df[df_key].Time_s.min())/df[df_key].d_a_2_um_2.head(1).values[0],
                     df[df_key].d_a_2_um_2/df[df_key].d_a_2_um_2.head(1).values[0],
                     color = colours[i], s = 10)

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('d$_a^2$ / µm$^2$')
    ax.legend()
    plt.show()
    
    
    #plotting multiple exposure data
    if multi_data_dicts == None:
        pass

    else:
        
        violinalpha = 0.3
        
        fig_multi_diameter_sq, ax = plt.subplots()
        
        for i, (df_dict, rh, multi_exposure_dict) in enumerate(zip(data_dicts, RHs, multi_data_dicts)):
            
            
            ax.plot(gaussian_rolling_average(df_dict[df_key].loc[df_dict[df_key].Time_s < multi_exposure_dict['timedata'].min()].Time_s, rolling_window)[0],
                    gaussian_rolling_average(df_dict[df_key].loc[df_dict[df_key].Time_s < multi_exposure_dict['timedata'].min()].d_a_2_um_2, rolling_window)[0],
                    label = str(rh)+ ' % RH',
                    color = colours[i])

            ax.scatter(gaussian_rolling_average(df_dict[df_key].loc[df_dict[df_key].Time_s < multi_exposure_dict['timedata'].min()].Time_s, rolling_window)[0],
                    gaussian_rolling_average(df_dict[df_key].loc[df_dict[df_key].Time_s < multi_exposure_dict['timedata'].min()].d_a_2_um_2, rolling_window)[0],
                    color = colours[i])
            
            df_multi_period = pd.merge(pd.DataFrame(multi_exposure_dict['timedata'],columns = ['time']),
                                   df_dict[df_key],
                                   left_on='time', right_on='Time_s')


            ax.plot(gaussian_rolling_average(multi_exposure_dict['average'].time_s, rolling_window)[0],
                    gaussian_rolling_average(multi_exposure_dict['average'].da_2/1e-12, rolling_window)[0],
                    color = colours[i],
                    lw = 0.5)

            violin_parts = ax.violinplot(multi_exposure_dict['violindata'],
                                         multi_exposure_dict['timedata'],
                                         widths=0.01,
                                         showmeans = 1,
                                         showmedians = 0,)
                                         #quantiles = len(multi_exposure_dict['violindata'])*[[0.25, 0.75]],)

            # Make all the violin statistics marks red:
            for partname in ('cbars','cmins','cmaxes', 'cmeans', 'cmedians', 'cquantiles'): #,'cmeans','cmedians'):
                if partname in violin_parts:
                    vp = violin_parts[partname]
                    vp.set_edgecolor(colours[i])
                    vp.set_linewidth(0.5)

            # Make the violin body blue with a red border:
            for vp in violin_parts['bodies']:
                vp.set_facecolor(colours[i])
                vp.set_edgecolor(colours[i])
                vp.set_linewidth(0.5)
                vp.set_alpha(violinalpha)

            # where some data has already been plotted to ax
            handles, labels = ax.get_legend_handles_labels()
            # manually define a new patch 
            patch = mpatches.Patch(color=colours[i], alpha = violinalpha, label='Multiple Exposure d$_a^2$')
            # handles is a list, so append manual patch
            handles.append(patch) 


        ax.legend(handles = handles)

        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_xlabel('Time / s')
        ax.set_ylabel('d$_a^2$ / µm$^2$')
        ax.legend()
        plt.show()
    
    #plotting multiple exposure data
    if multi_data_dicts == None:
        pass

    else:
        
        violinalpha = 0.3
        
        fig_multi_diameter_sq, ax = plt.subplots()
        
        for i, (df_dict, rh, multi_exposure_dict) in enumerate(zip(data_dicts, RHs, multi_data_dicts)):
            
            
            
            df_multi_period = pd.merge(pd.DataFrame(multi_exposure_dict['timedata'],columns = ['time']),
                                   df_dict[df_key],
                                   left_on='time', right_on='Time_s')

            ax.plot(gaussian_rolling_average((df_dict[df_key].loc[df_dict[df_key].Time_s < multi_exposure_dict['timedata'].min()].Time_s - df_dict[df_key].Time_s.min())/df_dict[df_key].d_a_2_um_2.head(1).values[0], rolling_window)[0],
                     gaussian_rolling_average(df_dict[df_key].loc[df_dict[df_key].Time_s < multi_exposure_dict['timedata'].min()].d_a_2_um_2/df_dict[df_key].d_a_2_um_2.head(1).values[0], rolling_window)[0],
                     label = str(rh)+ ' % RH',
                     color = colours[i])

            ax.scatter((df_dict[df_key].loc[df_dict[df_key].Time_s < multi_exposure_dict['timedata'].min()].Time_s - df_dict[df_key].Time_s.min())/df_dict[df_key].d_a_2_um_2.head(1).values[0],
                         df_dict[df_key].loc[df_dict[df_key].Time_s < multi_exposure_dict['timedata'].min()].d_a_2_um_2/df_dict[df_key].d_a_2_um_2.head(1).values[0],
                         color = colours[i], s = 10)


            ax.plot(gaussian_rolling_average((multi_exposure_dict['average'].time_s  - df_dict[df_key].Time_s.min())/df_dict[df_key].d_a_2_um_2.head(1).values[0],
                                                   rolling_window)[0],
                    gaussian_rolling_average(multi_exposure_dict['average'].da_2/1e-12/df_dict[df_key].d_a_2_um_2.head(1).values[0],
                                                  rolling_window)[0],
                    color = colours[i],
                    lw = 0.5)

            violin_parts = ax.violinplot(np.array(multi_exposure_dict['violindata'], dtype = 'object') / float(df_dict[df_key].d_a_2_um_2.head(1).values[0]),
                                         (multi_exposure_dict['timedata'] - df_dict[df_key].Time_s.min())/df_dict[df_key].d_a_2_um_2.head(1).values[0],
                                         widths=0.00001,
                                         showmeans = 1,
                                         showmedians = 0,)
                                         #quantiles = len(multi_exposure_dict['violindata'])*[[0.25, 0.75]],)

            # Make all the violin statistics marks red:
            for partname in ('cbars','cmins','cmaxes', 'cmeans', 'cmedians', 'cquantiles'): #,'cmeans','cmedians'):
                if partname in violin_parts:
                    vp = violin_parts[partname]
                    vp.set_edgecolor(colours[i])
                    vp.set_linewidth(0.5)

            # Make the violin body blue with a red border:
            for vp in violin_parts['bodies']:
                vp.set_facecolor(colours[i])
                vp.set_edgecolor(colours[i])
                vp.set_linewidth(0.5)
                vp.set_alpha(violinalpha)

            # where some data has already been plotted to ax
            handles, labels = ax.get_legend_handles_labels()
            # manually define a new patch 
            patch = mpatches.Patch(color=colours[i], alpha = violinalpha, label='Multiple Exposure d$_a^2$')
            # handles is a list, so append manual patch
            handles.append(patch) 


        ax.legend(handles = handles)

        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_xlabel('(Time / d$_0^2$) / (s  /µm$^2$ )')
        ax.set_ylabel('d$^2$ / d$_0^2$')
        ax.legend()
        plt.show()
    
    return

