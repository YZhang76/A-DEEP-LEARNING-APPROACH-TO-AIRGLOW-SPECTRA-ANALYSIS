# The input is the raw data file, e.g. 2019-07-27.dat. The output is a .dat file of smoothed signals, e.g. 20190727smoothed_signals.dat. The output is background corrected, and background_flag=0. 


# python pipline.py
# --data_dir <path to where the .dat files are>
# --result_dir <where you want the result .dat file saved> 
# --date <date  you want to plot  in yyymmdd format>  
# --learning_rate <Learning rate of the netrual network, default=1e-100> 
# --epochs <Epochs of the netrual network, default=2000. It should be >= 2000, but not too big for avoiding overfit.> 
# --partial_smooth_n <It is only for testing via a small computer, which still runs for several hours. Please ignore this if you have a powerful computer. A small integer 0<=n<5, e.g. 0, 1, 2. Smooth only spectra from time n*128 to time (n+1)*128. Choose n such that (n+1)*128 doesn't exceed the length of time. Default=10 means that the programme smooth all spectra. Default=10.>




import numpy as np
import pandas as pd
from datetime import *
import argparse
from os import path, makedirs
import pylab
from scipy import sparse
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import optimizers
import time
import csv



# Functions provided by BAS. 


def scan_dtype():
    return np.dtype([
        ('time', 'datetime64[s]'),
        ('solar_zenith_angle', 'float32'),
        ('ccd_temperature', 'float32'),
        ('exposure_time', 'int32'),
        ('exposures_per_scan', 'int32'),
        ('background_flag', 'int32'),
        ('spectra', 'int32', 1024)])


def bstr2str(s):
    return s.decode('UTF8')


def load_data(data_dir, current_date):
    fn = path.join(data_dir, current_date.strftime('%Y-%m-%d') + '.dat')
    return np.loadtxt(fn, dtype=scan_dtype(), delimiter=',', converters={0: bstr2str})


def split_data(data):
    signal = data[data['background_flag'] == 0]
    background = data[data['background_flag'] == 1]
    return signal, background


def subtract_background(signal, backgrounds):
    background_validity_time = np.timedelta64(600, 's')
    idx = backgrounds['time'].searchsorted(signal['time'])
    # The assumption made is that the first background selected will match scan parameters
    # (ccd_temp, exposure_time, etc.), and so only need to check the second background matches the first.
    # Backgrounds with the same scan parameters but taken more than background_validity_time seconds ago
    # are ignored.
    if idx == 0:
        raise IndexError('Scan found with no previous background')
    elif idx == 1:
        mean_background = backgrounds[idx-1]['spectra']
    else:
        bg1 = backgrounds[idx-1]
        bg2 = backgrounds[idx-2]
        if (signal['time'] - bg1['time']) <= background_validity_time                 and (signal['time'] - bg2['time']) <= background_validity_time                 and abs(bg1['ccd_temperature']-bg2['ccd_temperature']) < 2                 and bg1['exposure_time'] == bg2['exposure_time']                 and bg1['exposures_per_scan'] == bg2['exposures_per_scan']:
            mean_background = np.array([bg1['spectra'], bg2['spectra']]).mean(0)
        else:
            mean_background = backgrounds[idx-1]['spectra']
    signal['spectra'][:] = signal['spectra'] - mean_background
    
       

def full_day_spectra(signals, current_date):
    averaging_period = 60 # seconds
    resolution = 60*24 # (60 * 60 * 24) / averaging_period
    spectra = np.zeros((1024, resolution)) * np.nan
    
    start_of_day = datetime.combine(current_date, time(13,45,0))
    end_of_day = start_of_day + timedelta(hours=24)
    start_of_day = np.datetime64(start_of_day)
    end_of_day = np.datetime64(end_of_day)
    
    for signal in signals:
        signal_length = int(signal['exposure_time']*signal['exposures_per_scan'])
        signal_start = signal['time']
        signal_end = signal_start + np.timedelta64(signal_length, 's')
        
        start_idx = int(np.floor((signal_start - start_of_day).item().seconds / averaging_period))
        end_idx = int(np.floor((signal_end - start_of_day).item().seconds / averaging_period) + 1)
        #print(start_idx, end_idx-1, signal_start, signal_end, signal_length)
        spectra[:, start_idx:end_idx] = np.expand_dims(signal['spectra'].T, axis=1)
    return spectra


def date_type(datestr):
    return datetime.strptime(datestr, '%Y%m%d').date()

def output_to_csv(args, nc_result):
    f = open(args,  'w')
    csv_file = csv.writer(f)
    for row in nc_result:
        time = row['time']
        sza = row['solar_zenith_angle']
        ct = row['ccd_temperature']
        et = row['exposure_time']
        eps = row['exposures_per_scan']
        bf = row['background_flag']
        lis = [time, sza, ct, et, eps, bf]
        for i in range(1024):
            lis.append(row['spectra'][i])
        csv_file.writerow(lis)
    f.close()



# Median stack as the input.


def local_median(df, time_interval):

    ti = int(np.floor(time_interval/2))
    for index in range(df.shape[0]):
        if index < ti:
            df.iloc[index,:] = df.iloc[:index+ti+1,:].median(axis=0)
        elif index > df.shape[0] - ti:
            df.iloc[index,:] = df.iloc[index-ti:,:].median(axis=0)
        else:
            df.iloc[index,:] = df.iloc[index-ti:index+ti+1,:].median(axis=0)
    return df


def median_stack(signals, time_interval, window):
    
    sig = signals.copy()
    df = pd.DataFrame(sig['spectra'], index=signals['time'])
    df_median = local_median(df, time_interval) 
    df_smoothed = local_median(df_median.T, window).T   
    sig['spectra'] = df_smoothed.values
    
    return sig 


# Neural Network.


def build_model(shape, learning_rate):
    
    input_img = Input(shape=shape)  

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    model = Model(input_img, decoded)
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer='adam',  loss='mean_absolute_error')
    
    return model


# Prediction. 


def prediction(signals, learning_rate, epochs):
    
    sig = signals.copy()
    shape = sig['spectra'].shape
    median_s = median_stack(sig, 30, 4)['spectra']
    
    num = 0
    while num*128 < shape[0]:
        
        if 128*(num+1) > shape[0]:
            holder = np.ones([128*(num+1), 1024])
            holder[:shape[0], :1024] = median_s
            ms = holder.copy()
            holder[:shape[0], :1024] = sig['spectra']
            sp = holder.copy()
        else:
            ms = median_s.copy()
            sp = sig['spectra'].copy()
 
    # wave-length band 600-800: 
        X0 = ms[128*num:128*(num+1), 600:856]
        y0 = sp[128*num:128*(num+1), 600:856]
        ma0 = np.max(X0)
        mi0 = np.min(y0)
        normalise0 = ma0 - mi0
        y0 = (y0 - mi0)/normalise0
        X0 = (X0 - mi0)/normalise0
        X0 = X0.reshape(1, 128, 256, 1)
        y0 = y0.reshape(1, 128, 256, 1)
        model0 = build_model((128, 256, 1), learning_rate=learning_rate)
        model0.fit(X0, y0,  epochs=epochs)
        prediction0 = model0.predict(X0).reshape(128, 256)*normalise0 + mi0
        sp[128*num:128*(num+1), 600:856] = prediction0
        
    # wave-length band 200-600:    
        X1 = ms[128*num:128*(num+1), 88:600]
        y1 = sp[128*num:128*(num+1), 88:600]
        ma1 = np.max(X1)
        mi1 = np.min(y1)
        normalise1 = ma1 - mi1
        y1 = (y1 - mi1)/normalise1
        X1 = (X1 - mi1)/normalise1
        X1 = X1.reshape(1, 128, 512, 1)
        y1 = y1.reshape(1, 128, 512, 1)
        model1 = build_model((128, 512, 1), learning_rate=learning_rate)
        model1.fit(X1, y1,  epochs=epochs)
        prediction1 = model1.predict(X1).reshape(128, 512)*normalise1 + mi1
        sp[128*num:128*(num+1), 88:600] = prediction1
        
        sig['spectra'] = sp[:shape[0], :1024]
        num += 1
       
    return sig 

# Partial prediction for running via a small computer.  

def partial_prediction(signals, learning_rate, epochs, n):
    
    sig = signals.copy()
    shape = sig['spectra'].shape
    median_s = median_stack(sig, 30, 4)['spectra']
    
    num = n    
        
    if 128*(num+1) > shape[0]:
        holder = np.ones([128*(num+1), 1024])
        holder[:shape[0], :1024] = median_s
        ms = holder.copy()
        holder[:shape[0], :1024] = sig['spectra']
        sp = holder.copy()
    else:
        ms = median_s.copy()
        sp = sig['spectra'].copy()
 
 # wave-length band 600-800: 
    X0 = ms[128*num:128*(num+1), 600:856]
    y0 = sp[128*num:128*(num+1), 600:856]
    ma0 = np.max(X0)
    mi0 = np.min(y0)
    normalise0 = ma0 - mi0
    y0 = (y0 - mi0)/normalise0
    X0 = (X0 - mi0)/normalise0
    X0 = X0.reshape(1, 128, 256, 1)
    y0 = y0.reshape(1, 128, 256, 1)
    model0 = build_model((128, 256, 1), learning_rate=learning_rate)
    model0.fit(X0, y0,  epochs=epochs)
    prediction0 = model0.predict(X0).reshape(128, 256)*normalise0 + mi0
    sp[128*num:128*(num+1), 600:856] = prediction0
        
 # wave-length band 200-600:    
    X1 = ms[128*num:128*(num+1), 88:600]
    y1 = sp[128*num:128*(num+1), 88:600]
    ma1 = np.max(X1)
    mi1 = np.min(y1)
    normalise1 = ma1 - mi1
    y1 = (y1 - mi1)/normalise1
    X1 = (X1 - mi1)/normalise1
    X1 = X1.reshape(1, 128, 512, 1)
    y1 = y1.reshape(1, 128, 512, 1)
    model1 = build_model((128, 512, 1), learning_rate=learning_rate)
    model1.fit(X1, y1,  epochs=epochs)
    prediction1 = model1.predict(X1).reshape(128, 512)*normalise1 + mi1
    sp[128*num:128*(num+1), 88:600] = prediction1
        
    sig['spectra'] = sp[:shape[0], :1024]
           
    return sig 


# Running the code. 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Directory in which to find raw data files')
    parser.add_argument('--result_dir', help='Directory in which results will be placed', default='.')
    parser.add_argument('--date', type=date_type, help='Specify date for which to produce plot (YYYYMMDD) Defaults to yesterday.',
                        default=date.today() - timedelta(days=1))
    parser.add_argument('--learning_rate', help='Learning rate of the netrual network, default=1e-100', default=1e-100)
    parser.add_argument('--epochs', help='Epochs of the netrual network, default=2000', default=2000)
    parser.add_argument('--partial_smooth_n', help='A number 0<=n<=5. Smooth spectra between n*128 to (n+1)*128. Default=10, smooth all    spectra.', default=10)
    args = parser.parse_args()
    
    scans = load_data(args.data_dir, args.date)
    signals, backgrounds = split_data(scans)
    for signal in signals:
        subtract_background(signal, backgrounds)
        signal['spectra'][:] = signal['spectra'] - np.mean(signal['spectra'])
    for signal in signals:
        signal['spectra'][:] = signal['spectra'] - np.min(signals['spectra'])

    if args.partial_smooth_n != 10: 
        predict = partial_prediction(signals, float(args.learning_rate), int(args.epochs), int(args.partial_smooth_n))
    else:
        predict = prediction(signals, float(args.learning_rate), int(args.epochs))
    
    output_to_csv(path.join(args.result_dir, args.date.strftime('%Y%m%d') + 'smoothed_signals.dat'), predict)
    
    
if __name__ == "__main__":
    main()





