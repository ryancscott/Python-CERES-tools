# ==============================================================================
# This script is used to compare the performance of various machine learning
# learning algorithms in terms of their ability to predict CRS downward longwave
# radiative fluxes at the surface.
# ==============================================================================


import sys
import numpy as np
import cerestools as ceres
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import mpl_scatter_density

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

# ----------------------------------------------------------

path = '/Users/rcscott2/Desktop/CRS/my_output/JAN-2019_/'

full_day = True  # False = an hourly granule instead of a full day

# ----------------------------------------------------------

if full_day is False:

    file = 'CER_CRS4_Terra-FM1-MODIS_GH4_1111TH.2019010100'
    file_ = path+file

    lat, lon, p_levs, obs_tim, sza = ceres.read_crs_geolocation_dev(file_)

    lat[abs(lat) > 90] = np.nan
    lon[abs(lon) > 360] = np.nan
    sza[abs(sza) > 180] = np.nan

    # target/label variable
    lwd, lwd_name, lwd_units, _ = \
        ceres.read_crs_var(file_path=file_,
                           var_name='Longwave flux - downward - total sky',
                           lev_arg=5,
                           fill=True)

    # training/feature variables
    t850, t850_name, t850_units, _ = \
        ceres.read_crs_var(file_path=file_,
                           var_name='Temperature profile',
                           lev_arg=4,
                           fill=True)

    cf1, cf1_name, cf1_units, _ = \
        ceres.read_crs_var(file_path=file_,
                           var_name='Cloud fraction',
                           lev_arg=0,
                           fill=True)

    cf2, cf2_name, cf2_units, _ = \
        ceres.read_crs_var(file_path=file_,
                           var_name='Cloud fraction',
                           lev_arg=1,
                           fill=True)

    ct1, ct1_name, ct1_units, _ = \
        ceres.read_crs_var(file_path=file_,
                           var_name='Cloud temperature',
                           lev_arg=0,
                           fill=True)

    ct2, ct2_name, ct2_units, _ = \
        ceres.read_crs_var(file_path=file_,
                           var_name='Cloud temperature',
                           lev_arg=1,
                           fill=True)

    cod1, cod1_name, cod11_units, _ = \
        ceres.read_crs_var(file_path=file_,
                           var_name='Cloud optical thickness',
                           lev_arg=0,
                           fill=True)

    cod2, cod2_name, cod2_units, _ = \
        ceres.read_crs_var(file_path=file_,
                           var_name='Cloud optical thickness',
                           lev_arg=1,
                           fill=True)

    pw, pw_name, pw_units, _ = \
        ceres.read_crs_var(file_path=file_,
                           var_name='Precipitable water',
                           lev_arg=-1,
                           fill=True)

    for i in range(len(lat)):
        if np.isnan(ct1[i]):
            ct1[i] = 0
        if np.isnan(ct2[i]):
            ct2[i] = 0
        if np.isnan(cf1[i]):
            cf1[i] = 0
        if np.isnan(cf2[i]):
            cf2[i] = 0
        if np.isnan(cod1[i]):
            cod1[i] = 0
        if np.isnan(cod2[i]):
            cod2[i] = 0

    cf = cf1 + cf2
    ct = (cf1 * ct1 + cf2 * ct2) / cf
    cod = (cf1 * cod1 + cf2 * cod2) / cf

    ct[np.isnan(ct)] = 0
    cod[np.isnan(cod)] = 0

    for i in range(10):
        # print(cf1[i], cf2[i], cf[i], ct1[i], ct2[i], ct[i], cod1[i], cod2[i], cod[i])
        print(cf[i], ct[i], cod[i])

# ----------------------------------------------------------

elif full_day is True:

    # target/label variable
    lwd, lon, lat, sza = \
        ceres.read_day_of_crs_files(
            path=path,
            file_struc='CER_CRS4_Terra-FM1-MODIS_GH4_1111TH.20190101',
            variable='Longwave flux - downward - total sky',
            lev_arg=5,
            fill=True,
            dev=True)

    # training/feature variables
    t850, _, _, _ = \
        ceres.read_day_of_crs_files(
            path=path,
            file_struc='CER_CRS4_Terra-FM1-MODIS_GH4_1111TH.20190101',
            variable='Temperature profile',
            lev_arg=5,
            fill=True,
            dev=True)

    cf1, _, _, _ = \
        ceres.read_day_of_crs_files(
            path=path,
            file_struc='CER_CRS4_Terra-FM1-MODIS_GH4_1111TH.20190101',
            variable='Cloud fraction',
            lev_arg=0,
            fill=True,
            dev=True)

    cf2, _, _, _ = \
        ceres.read_day_of_crs_files(
            path=path,
            file_struc='CER_CRS4_Terra-FM1-MODIS_GH4_1111TH.20190101',
            variable='Cloud fraction',
            lev_arg=1,
            fill=True,
            dev=True)

    ct1, _, _, _ = \
        ceres.read_day_of_crs_files(
            path=path,
            file_struc='CER_CRS4_Terra-FM1-MODIS_GH4_1111TH.20190101',
            variable='Cloud temperature',
            lev_arg=0,
            fill=True,
            dev=True)

    ct2, _, _, _ = \
        ceres.read_day_of_crs_files(
            path=path,
            file_struc='CER_CRS4_Terra-FM1-MODIS_GH4_1111TH.20190101',
            variable='Cloud temperature',
            lev_arg=1,
            fill=True,
            dev=True)

    cod1, _, _, _ = \
        ceres.read_day_of_crs_files(
            path=path,
            file_struc='CER_CRS4_Terra-FM1-MODIS_GH4_1111TH.20190101',
            variable='Cloud optical thickness',
            lev_arg=0,
            fill=True,
            dev=True)

    cod2, _, _, _ = \
        ceres.read_day_of_crs_files(
            path=path,
            file_struc='CER_CRS4_Terra-FM1-MODIS_GH4_1111TH.20190101',
            variable='Cloud optical thickness',
            lev_arg=1,
            fill=True,
            dev=True)

    pw, _, _, _ = \
        ceres.read_day_of_crs_files(
            path=path,
            file_struc='CER_CRS4_Terra-FM1-MODIS_GH4_1111TH.20190101',
            variable='Precipitable water',
            lev_arg=-1,
            fill=True,
            dev=True)

    lat[abs(lat) > 90] = np.nan
    lon[abs(lon) > 360] = np.nan
    sza[abs(sza) > 180] = np.nan

    for i in range(len(lat)):
        if np.isnan(ct1[i]):
            ct1[i] = 0
        if np.isnan(ct2[i]):
            ct2[i] = 0
        if np.isnan(cf1[i]):
            cf1[i] = 0
        if np.isnan(cf2[i]):
            cf2[i] = 0
        if np.isnan(cod1[i]):
            cod1[i] = 0
        if np.isnan(cod2[i]):
            cod2[i] = 0

    cf = cf1 + cf2
    cf[cf > 3e+38] = 0
    ct = (cf1 * ct1 + cf2 * ct2) / cf
    cod = (cf1 * cod1 + cf2 * cod2) / cf

    ct[np.isnan(ct)] = 0
    cod[np.isnan(cod)] = 0

    ct[ct > 1000] = 0
    cf[cf > 1000] = 0
    cod[cod > 1000] = 0
    pw[pw > 1000] = 0

    # for i in range(10):
    #     print(cf1[i], cf2[i], ct1[i], ct2[i], ct[i], cod1[i], cod2[i], cod[i])

    # sys.exit()

# ==========================================================
# Put the atmospheric variables into a pandas dataframe
# for subsequent processing - removing NaNs, scaling, etc.
# ==========================================================
print('\nShape of data\n')
print('Lat:', lat.shape)
print('LWd:', lwd.shape)
print('T850 :', t850.shape)
print('CF:', cf.shape)
print('CT:', ct.shape)
print('COD:', cod.shape)
print('PWV:', pw.shape, '\n')

# put the data into a pandas data frame
d = {'sza': sza,
     'cf': cf, 'ct': ct, 'cod': cod,
     't850': t850, 'pwv': pw, 'lwd': lwd}

data = pd.DataFrame(data=d)
print('\nDataframe:\n', data.head(10))

print('\nNum FOVs with NaN data:\n',
      data.isnull().sum(axis=0))

# drop rows with any NaN values (how to best handle cloud temperature)
data = data.dropna()
data = data.drop(index=0)

print('Dataframe shape:', data.shape)
print('\nPreview of dataframe:\n', data.head())

X = data[['sza', 'cf', 'ct', 'cod', 't850', 'pwv']]
y = data['lwd']

# transform X values to z scores prior to training
# do for just X (or pass the entire df to scale all the data)
s = StandardScaler().fit(X)
X = s.transform(X)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ---------------------------------------------------------------------

# # split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# print('X_test (not scaled yet): ', X_test)
#
# # transform X values to z scores prior to training
# # do for just X (or pass the entire df to scale all the data)
# s1 = StandardScaler().fit(X_train)
# X_train = s1.transform(X_train)
# s2 = StandardScaler().fit(X_test)
# X_test = s2.transform(X_test)
#
# # scaled versions
# print('X_train: ', X_train, '\n')
# print('X_test: ', X_test)

# ---------------------------------------------------------------------

# print out training / testing data shapes
print('\nX train shape: \n', X_train.shape)
print('y train shape: \n', y_train.shape)
print('X test shape: \n', X_test.shape)
print('y test shape: \n', y_test.shape)


# ==========================================================
# Train various regression models to predict CRS LW flux
# and compare their performance
# ==========================================================

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_y_pred = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, lin_y_pred)
lin_mae = mean_absolute_error(y_test, lin_y_pred)
lin_rmse = np.sqrt(lin_mse)
print('\nLinear Regression:')
print('MSE: ', lin_mse)
print('MAE: ', lin_mae)
print('RMSE: ', lin_rmse)

# Decision Tree Regressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
tree_y_pred = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_y_pred)
tree_mae = mean_absolute_error(y_test, tree_y_pred)
tree_rmse = np.sqrt(tree_mse)
print('\nDecision Tree:')
print('MSE: ', tree_mse)
print('MAE: ', tree_mae)
print('RMSE: ', tree_rmse)

# Random Forest Regressor
forest_reg = RandomForestRegressor(n_estimators=200,
                                   random_state=0)
forest_reg.fit(X_train, y_train)
forest_y_pred = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_test, forest_y_pred)
forest_mae = mean_absolute_error(y_test, forest_y_pred)
forest_rmse = np.sqrt(forest_mse)
print('\nRandom Forest:')
print('MSE: ', forest_mse)
print('MAE: ', forest_mae)
print('RMSE: ', forest_rmse)

# XGBoost Regressor
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror',
                           n_estimators=200,
                           seed=123,
                           booster="dart",
                           max_depth=7)
xgb_reg.fit(X_train, y_train)
xgb_y_pred = xgb_reg.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_mae = mean_absolute_error(y_test, xgb_y_pred)
xgb_rmse = np.sqrt(xgb_mse)
print('\nXGBoost:')
print('MSE: ', xgb_mse)
print('MAE: ', xgb_mae)
print('RMSE: ', xgb_rmse)

# # Support Vector Regressor - use GridSearch to optimize C, gamma
# svm_reg = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
# svm_reg.fit(X_train, y_train)
# svm_y_pred = svm_reg.predict(X_test)
# svm_mse = mean_squared_error(y_test, svm_y_pred)
# svm_mae = mean_absolute_error(y_test, svm_y_pred)
# svm_rmse = np.sqrt(svm_mse)
# print('\nSVM:')
# print('MSE: ', svm_mse)
# print('MAE: ', svm_mae)
# print('RMSE: ', svm_rmse)

# ==========================================================
# Plot of actual CRS LWd vs model predicted CRS LWd
# ==========================================================
min_lwd = 100
max_lwd = 500

bins = np.arange(min_lwd, max_lwd, 2.5)
x = np.linspace(min_lwd, max_lwd, 100)

fig, ax = plt.subplots(2, 2, figsize=(9, 8))
fig.suptitle('CER_CRS4_Terra-FM1-MODIS_GH4_1111TH.20190101:00-23h\n' +
             r'Training features: Cloud properties (frac, temp, OD), PWV, T$_{850}$, SZA')
ax[0, 0].plot(x, x, 'k', alpha=0.2)
ax[0, 0].scatter(y_test, lin_y_pred, s=0.01, alpha=1, zorder=2)#, c=y_test, cmap='viridis')
#_, _, _, im = ax[0, 0].hist2d(y_test, lin_y_pred, bins=[bins, bins], cmap='BuPu', vmin=0, vmax=25)
#cb_ax = fig.add_axes([0.905, 0.53, 0.015, 0.35])
#cb = plt.colorbar(im, cax=cb_ax)
ax[0, 0].set_title('Linear Regression')
#ax[0, 0].set_xlabel(r'Actual CRS LW$\downarrow$ [W m$^{-2}$]')
ax[0, 0].set_ylabel(r'Predicted CRS LW$\downarrow$ [W m$^{-2}$]')
ax[0, 0].set_xlim([min_lwd, max_lwd])
ax[0, 0].set_ylim([min_lwd, max_lwd])
text0 = 'N = ' + str(y_test.shape[0]) + '\n' + \
        'MSE: ' + str(np.around(lin_mse, 2)) + '\n' + \
        'MAE: ' + str(np.around(lin_mae, 2)) + '\n' + \
        'RMSE: ' + str(np.around(lin_rmse, 2))
ax[0, 0].text(0.05, 0.95, text0, transform=ax[0, 0].transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(facecolor='white', alpha=0.85))

ax[0, 1].plot(x, x, 'k', alpha=0.2)
ax[0, 1].scatter(y_test, tree_y_pred, s=0.01, alpha=1, zorder=2)#, c=y_test, cmap='viridis')
#ax[0, 1].hist2d(y_test, tree_y_pred, bins=[bins, bins], cmap='BuPu', vmin=0, vmax=25)
ax[0, 1].set_title('Decision Tree')
#ax[0, 1].set_xlabel(r'Actual CRS LW$\downarrow$ [W m$^{-2}$]')
ax[0, 1].set_ylabel(r'Predicted CRS LW$\downarrow$ [W m$^{-2}$]')
ax[0, 1].set_xlim([min_lwd, max_lwd])
ax[0, 1].set_ylim([min_lwd, max_lwd])
text1 = 'N = ' + str(y_test.shape[0]) + '\n' + \
        'MSE: ' + str(np.around(tree_mse, 2)) + '\n' + \
        'MAE: ' + str(np.around(tree_mae, 2)) + '\n' + \
        'RMSE: ' + str(np.around(tree_rmse, 2))
ax[0, 1].text(0.05, 0.95, text1, transform=ax[0, 1].transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(facecolor='white', alpha=0.85))
ax[1, 0].plot(x, x, 'k', alpha=0.2)
ax[1, 0].scatter(y_test, forest_y_pred, s=0.01, alpha=1, zorder=2)#, c=y_test, cmap='viridis')
# ax[1, 0].hist2d(y_test, forest_y_pred, bins=[bins, bins], cmap='BuPu', vmin=0, vmax=25)
ax[1, 0].set_title('Random Forest')
ax[1, 0].set_xlabel(r'Actual CRS LW$\downarrow$ [W m$^{-2}$]')
ax[1, 0].set_ylabel(r'Predicted CRS LW$\downarrow$ [W m$^{-2}$]')
ax[1, 0].set_xlim([100, 500])
ax[1, 0].set_ylim([100, 500])
text2 = 'N = ' + str(y_test.shape[0]) + '\n' + \
        'MSE: ' + str(np.around(forest_mse, 2)) + '\n' + \
        'MAE: ' + str(np.around(forest_mae, 2)) + '\n' + \
        'RMSE: ' + str(np.around(forest_rmse, 2))
ax[1, 0].text(0.05, 0.95, text2, transform=ax[1, 0].transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(facecolor='white', alpha=0.85))
ax[1, 1].plot(x, x, 'k', alpha=0.2)
ax[1, 1].scatter(y_test, xgb_y_pred, s=0.01, alpha=1, zorder=2)#, c=y_test, cmap='viridis')
# ax[1, 1].hist2d(y_test, xgb_y_pred, bins=[bins, bins], cmap='BuPu', vmin=0, vmax=25)
ax[1, 1].set_title('XGBoost')
ax[1, 1].set_xlabel(r'Actual CRS LW$\downarrow$ [W m$^{-2}$]')
ax[1, 1].set_ylabel(r'Predicted CRS LW$\downarrow$ [W m$^{-2}$]')
ax[1, 1].set_xlim([min_lwd, max_lwd])
ax[1, 1].set_ylim([min_lwd, max_lwd])
text3 = 'N = ' + str(y_test.shape[0]) + '\n' + \
        'MSE: ' + str(np.around(xgb_mse, 2)) + '\n' + \
        'MAE: ' + str(np.around(xgb_mae, 2)) + '\n' + \
        'RMSE: ' + str(np.around(xgb_rmse, 2))
ax[1, 1].text(0.05, 0.95, text3, transform=ax[1, 1].transAxes, fontsize=11,
              verticalalignment='top', bbox=dict(facecolor='white', alpha=0.85))
# plt.tight_layout()
plt.show()

# # print out actual vs prediction side by side
# for i, (el1, el2) in enumerate(list(zip(y_test, forest_y_pred))):
#     print(i, el1, el2)

