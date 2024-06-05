""" nlf_nn.py -- Net Load Forecast Using Neural Nets

    The main script for creating and submitting forecasts to the remote server.
    USAGE:
    Train a model for the Georga Substation:
    $ python nlf_nn.py -s GA -tr   

    Plot the forecast:
    $ python nlf_nn.py -s GA --plot

    Create a forecast:
    $ python nlf_nn.py -s GA -cf

    Upload a forecast:
    $ python nlf_nn.py --upload-forecast --file data/forecast.nn_point.20230711_094825.90c2a42c-f0ad-11ed-94b4-5edf5e2b3336.csv
 
 """
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import joblib
import unittest
import logging as log
import pdb
# device = torch.device("mps")
device = torch.device("cpu")
print(f"Using {device} device")

import torch.nn as nn
import torch.nn.functional as F
import torch
torch.manual_seed(43)

CLI_SITE_ID = None
MODEL_NAME = f'nn_point'
site_id_map = {
        'GA': '90c2a42c-f0ad-11ed-94b4-5edf5e2b3336',
        'TX': '8568f10f-eb8f-11ed-a556-128dcacebd72',
        'OR': '5ebb4527-edbd-11ed-bf8d-128dcacebd72',
        'HI': 'c639b1f3-eb8f-11ed-802e-aec5a60999dc'
    }



best_models = [

            # GA
            'models/nn_point.90c2a42c-f0ad-11ed-94b4-5edf5e2b3336.2023-07-13_091602.51', # MSE: 0.003

            # TX
            'models/nn_point.8568f10f-eb8f-11ed-a556-128dcacebd72.2023-07-13_093123.61', # MSE: 0.001

            # OR
            'models/nn_point.5ebb4527-edbd-11ed-bf8d-128dcacebd72.2023-07-13_110623.34', # MSE: 0.001

            # HI
            'models/nn_point.c639b1f3-eb8f-11ed-802e-aec5a60999dc.2023-07-11_112831.55', # MSE: 0.017 HI

    ]
# Data windowing
label_columns = ['net_load']

input_columns = ['refc_Maximum/Composite radar reflectivity_atmosphere_dB', 'vis_Visibility_surface_m', 'gust_Wind speed (gust)_surface_m s**-1', 'sp_Surface pressure_surface_Pa', 't_Temperature_surface_K', 'tcc_Total Cloud Cover_boundaryLayerCloudLayer_%', 'lcc_Low cloud cover_lowCloudLayer_%', 'mcc_Medium cloud cover_middleCloudLayer_%', 'hcc_High cloud cover_highCloudLayer_%', 'tcc_Total Cloud Cover_atmosphere_%', 'ulwrf_Upward long-wave radiation flux_nominalTop_W m**-2', 'dswrf_Downward short-wave radiation flux_surface_W m**-2', 'dlwrf_Downward long-wave radiation flux_surface_W m**-2', 'uswrf_Upward short-wave radiation flux_surface_W m**-2', 'ulwrf_Upward long-wave radiation flux_surface_W m**-2', 'vbdsf_Visible Beam Downward Solar Flux_surface_W m**-2', 'vddsf_Visible Diffuse Downward Solar Flux_surface_W m**-2', 'uswrf_Upward short-wave radiation flux_nominalTop_W m**-2', 'r_Relative humidity_isothermZero_%', 'pres_Pressure_isothermZero_Pa', 'dayofweek', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos']
# input_columns = ['t_Temperature_surface_K', 'tcc_Total Cloud Cover_boundaryLayerCloudLayer_%', 'lcc_Low cloud cover_lowCloudLayer_%', 'mcc_Medium cloud cover_middleCloudLayer_%',  'tcc_Total Cloud Cover_atmosphere_%', 'r_Relative humidity_isothermZero_%',  'dayofweek', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos']
all_columns = ['timestamp'] + input_columns + label_columns 


class NeuralNetworkProb(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=32):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), 
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), 
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), 
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), 
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), 
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, device=device),
        )
        self.train_rmse_list = []
        self.test_rmse_list = []

    def forward(self, x):
        y_pred = self.linear_relu_stack(x).to(device) # range -inf to +inf

        return y_pred

def mean_std_loss(y_pred, y_actual):
    # y_pred.squeeze()
    mu_pred = y_pred[0][0].to(device)
    std_pred = y_pred[0][1].to(device)
    y_actual = y_actual.to(device)
    alpha = 1e-2
    loss = torch.mean((y_pred - y_actual)**2).sqrt() + torch.sigmoid(std_pred)
    return loss

def predict(model, dataset):
    model.eval()
    y_pred = []
    y_actual = []
    with torch.no_grad():
        for i in range(len(dataset)):
            X, y = dataset[i]
            pred = model(X.to(device)).to(device)
            y_actual.append(y.item())
            y_pred.append(pred.item())

    return np.array(y_actual), np.array(y_pred) 

def create_train_test_datasets(master_df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    if 'timestamp' not in master_df.columns:
        master_df = master_df.reset_index()
    n = len(master_df)

    # shuffle the data
    master_df = master_df.sample(frac=1, random_state=1).reset_index(drop=True)
    train_df = master_df.iloc[0:int(n*train_ratio)]
    test_df = master_df.iloc[int(n*train_ratio):int(n*(train_ratio+val_ratio))]
    val_df = master_df.iloc[int(n*(train_ratio+val_ratio)):]

    return train_df, val_df, test_df


class NetLoadForecastDataset(Dataset):
    def __init__(self, df, site_id, window, training=False, forecast=None, mu=None, std=None): 
        """ Dataframe is the entire dataframe, including features, non-features, and labels.
            Window is the window object that will be used to split the dataframe into input and labels.
            the input_df is the dataframe that will be used to create the input tensor.
            the labels_df is the dataframe that will be used to create the labels tensor.
            input - is the input tensor
            labels - is the labels tensor
        """
        self.window = window
        self.site_id = site_id
        self.X_mean = mu
        self.X_std  = std 
        self.training = training
        self.forecast = forecast
        self.df = df
        self.input_df, self.labels_df = window.split_window(self.df)
        self.input = self.input_df.reset_index(drop=True).values
        if self.forecast:
            self.labels = np.zeros((self.input_df.shape[0], self.window.label_width)) 
        else:
            self.labels = self.labels_df.reset_index(drop=True).values
        self.input_norm = self.normalize_input(self.input)
        self.input = torch.from_numpy(self.input_norm.astype('float32')).to(device)
        self.labels = torch.from_numpy(self.labels.astype('float32')).to(device)
        self.len_remaining = self.input_df.shape[0] - self.window.total_window_size

    def get_scalars(self):
        return {'mu': self.X_mean, 'std': self.X_std}

    def __len__(self):
        return self.input_df.shape[0] - self.window.total_window_size + 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx > self.__len__():
            raise StopIteration
        else:
            X = self.input[idx].to(device)
            y = self.labels[idx].to(device)
        return X, y 

    def normalize_input(self, X)->np.ndarray:
        if self.training:
            self.X_mean = X.mean(axis=0)
            self.X_std  = X.std(axis=0)

        if self.X_mean is None or self.X_std is None:
            raise Exception('Normalizing scalars is None')
            
        return (X - self.X_mean) / self.X_std

    def get_input_timestamp(self, idx):
        # return self.input_df['timestamp'].iloc[idx+self.window.input_indices[-1]]
        return self.df['timestamp'].iloc[idx+self.window.input_indices[-1]]

def save_scalers(scalars: dict, model_path: str):
    """ Save the mean and std scalars to the model path."""
    scalar_file = f'{model_path}.scalars'
    joblib.dump(scalars, scalar_file)
    print(f'Model Scaler saved to {scalar_file}')   

def load_scalars(model_path: str):
    """ Load the mean and std scalars from the model path."""
    scalar_file = f'{model_path}.scalars'
    return joblib.load(scalar_file)

def ETL(df_in: pd.DataFrame):
    """ Perform the ETL on the MASTER dataframe. """

    df = df_in.copy()
    df.reset_index(inplace=True, drop=True)

    label_columns = ['net_load']
    input_columns = ['refc_Maximum/Composite radar reflectivity_atmosphere_dB', 'vis_Visibility_surface_m', 'gust_Wind speed (gust)_surface_m s**-1', 'sp_Surface pressure_surface_Pa', 't_Temperature_surface_K', 'tcc_Total Cloud Cover_boundaryLayerCloudLayer_%', 'lcc_Low cloud cover_lowCloudLayer_%', 'mcc_Medium cloud cover_middleCloudLayer_%', 'hcc_High cloud cover_highCloudLayer_%', 'tcc_Total Cloud Cover_atmosphere_%', 'ulwrf_Upward long-wave radiation flux_nominalTop_W m**-2', 'dswrf_Downward short-wave radiation flux_surface_W m**-2', 'dlwrf_Downward long-wave radiation flux_surface_W m**-2', 'uswrf_Upward short-wave radiation flux_surface_W m**-2', 'ulwrf_Upward long-wave radiation flux_surface_W m**-2', 'vbdsf_Visible Beam Downward Solar Flux_surface_W m**-2', 'vddsf_Visible Diffuse Downward Solar Flux_surface_W m**-2', 'uswrf_Upward short-wave radiation flux_nominalTop_W m**-2', 'r_Relative humidity_isothermZero_%', 'pres_Pressure_isothermZero_Pa', 'dayofweek', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos']
    all_columns   = ['timestamp'] + input_columns + label_columns

    # drop rows with user flagged data
    if 'quality_flag' in df.columns:
        df.dropna(subset=['quality_flag'], inplace=True)
        df['quality_flag'] = df['quality_flag'].astype(int)
        df['user_flagged'] = df['quality_flag'].apply(lambda x: int(x) & 0x1)
        df = df[df['user_flagged'] == 0]

    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['year'] = pd.to_datetime(df['timestamp']).dt.year
    df['dayofyear'] = pd.to_datetime(df['timestamp']).dt.dayofyear
    
    # drop the rows with duplicate timestamps
    df.drop_duplicates(subset=['timestamp'], inplace=True)

    # remove any net load values that are greater than 3 standard deviations from the mean
    if 'net_load' in df.columns:
        df = df[ df['net_load'] != 0.00 ]
    
    # Compute the sin and cos of the hour to capture the cyclical nature of time
    hours_in_day = 24.0
    days_in_year = 365.2425
    df['hour_sin'] = np.sin(df.hour * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df.hour * (2 * np.pi / 24))
    df['year_sin'] = np.sin(df.dayofyear * hours_in_day * (2 * np.pi / (days_in_year * hours_in_day)))
    df['year_cos'] = np.cos(df.dayofyear * hours_in_day * (2 * np.pi / (days_in_year * hours_in_day)))

    if 'net_load' not in df.columns:
        all_columns = ['timestamp'] + input_columns
    df = df[all_columns]

    df.dropna(inplace=True)
    df.set_index('timestamp', inplace=True)

    print(f'df.shape: {df.shape}')
    if df.isna().sum().sum() > 0:
        raise ValueError(f'There are still {df.isna().sum().sum()} missing values in the dataframe!')
    else:
        print('There are no missing values in the dataframe!')
    return df


class WindowGenerator():
    """A class for windowing time-series data.
    A class for windowing time-series data.
    INPUT:
      input_width: The number of input time steps to use.
      label_width: The number of output time steps to use.
      shift: The number of time steps to shift the label relative to the last input time step.
      Example:

      # Align the label with the input directly. Indices should be features: [0], labels: [0] window_0= WindowGenerator(input_width=1, offset=-1, label_width=1, input_columns=['hour_sin', 'year_sin'], label_columns=['net_load', 'rolling_std']) print(window_0) Total window size: 1 Input indices: [0] Label indices: [0] Label column name(s): ['net_load', 'rolling_std'] # indices should be features: [0:9], labels: [10:14] window_10_0_5= WindowGenerator(input_width=10, offset=0, label_width=5, input_columns=['hour_sin', 'year_sin'], label_columns=['net_load', 'rolling_std']) print(window_10_0_5) Total window size: 15 Input indices: [0 1 2 3 4 5 6 7 8 9] Label indices: [10 11 12 13 14] Label column name(s): ['net_load', 'rolling_std']
      # indices should be features: [0:9], labels: [14]
      window_10_4_2= WindowGenerator(input_width=10, offset=4, label_width=2, input_columns=['hour_sin', 'year_sin'], label_columns=['net_load', 'rolling_std'])
      print(window_10_4_1)
      Total window size: 16
      Input indices: [0 1 2 3 4 5 6 7 8 9]
      Label indices: [14 15]
      Label column name(s): ['net_load', 'rolling_std']




        Input the last 24 hours of data and predict the next consecutive 6 hours of data.

        w1 = WindowGenerator(input_width=24, label_width=6, shift=6,
                            label_columns=['net_load'])

        Total window size: 30
        Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
        Label indices: [24 25 26 27 28 29]
        Label column name(s): ['net_load']

      train_df: The training dataframe.
      val_df: The validation dataframe.
      test_df: The test dataframe.
      label_columns: The column names of the labels.


      This class can:

      Handle the indexes and offsets as shown in the diagrams above.
      Split windows of features into (features, labels) pairs.
      Plot the content of the resulting windows.
      Efficiently generate batches of these windows from the training, evaluation, and test data, using tf.data.Datasets.

    """
    def __init__(self, input_width, label_width, offset, input_columns, label_columns):
      # Store the raw data.
      self.plot = None


      # Work out the label column indices.
      self.label_columns = label_columns
      self.input_columns = input_columns
      self.label_columns_indices = {name: i for i, name in
                                      enumerate(label_columns)}
      self.column_indices = {name: i for i, name in
                             enumerate(input_columns)}

      # Work out the window parameters.
      # indices should be features: [0:9], labels: [10:14]
      # 10, 4, 1 should yield: features: [0:9], labels: [14]
      self.input_width = input_width
      self.offset = offset 
      self.label_width = label_width

      self.total_window_size = input_width + offset + label_width 

      self.input_slice = slice(0, input_width) # slice(0, 9, 2) is equivalant to [0:9:2]
      self.input_indices = np.arange(self.total_window_size)[self.input_slice]

      self.label_start = self.total_window_size - self.label_width
      self.labels_slice = slice(self.label_start, self.total_window_size) # self.total_window_size - input_width - offset)
      self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
      return '\n'.join([
          f'Total window size: {self.total_window_size}',
          f'Input indices: {self.input_indices}',
          f'Label indices: {self.label_indices}',
          f'Label column name(s): {self.label_columns}'])

    def plot(self, model=None, plot_col='net_load', max_subplots=3):
      inputs, labels = self.example
      plt.figure(figsize=(12, 8))
      plot_col_index = self.column_indices[plot_col]
      max_n = min(max_subplots, len(inputs))
      for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
          label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
          label_col_index = plot_col_index

        if label_col_index is None:
          continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
          predictions = model(inputs)
          plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)

        if n == 0:
          plt.legend()

        plt.xlabel('Time [h]')

    def split_window(self, df):
      if 'timestamp' not in df.columns:
        # df = df.set_index('timestamp')
        df = df.reset_index()
      # inputs_df = df[self.input_columns][self.input_slice]
      # labels_df = df[self.label_columns][self.labels_slice]
      inputs_df = df[self.input_columns]
      try:
        labels_df = df[self.label_columns]
      except KeyError:
        labels_df = None

      return inputs_df, labels_df 



class SfaForecastNnModel():
    def __init__(self, site_id=None, model_path=None, model_name=None, training=False):
        if site_id is None or model_path is None:
            raise ValueError('site_id and model_path are required')
        self.site_id = site_id
        self.df = pd.read_csv('data/data.MASTER.csv')
        self.df = self.df[ self.df.site_id == site_id ]
        self.df = ETL(self.df) # create the main df with rows and cols
        self.train_df, self.test_df, self.val_df = create_train_test_datasets(self.df)
        self.window_1to1 = WindowGenerator(input_width=1, offset=-1, label_width=1, input_columns=input_columns, label_columns=label_columns)
        self.model = NeuralNetworkProb(input_dim=len(input_columns), 
                      output_dim=1, 
                      hidden_dim=32).to(device)
        self.model_path = model_path
        self.model_name = model_name
        if not training:
            self.model.load_state_dict(torch.load(model_path))

    def plot_prediction(self):
        
        print('Plotting prediction...')
        scalars = load_scalars(fx.model_path)
        df = self.df.copy()
        df.reset_index(inplace=True)
        df.timestamp = pd.to_datetime(df.timestamp)

        df = df[ df.timestamp >= pd.Timestamp('2023-06-27', tz='utc')  ] 
        df = df[ df.timestamp <= pd.Timestamp('2023-07-05', tz='utc')  ] 
        # register mu and std as model parameters so they are saved with the model
        fx_ds = NetLoadForecastDataset(df, self.site_id, fx.window_1to1, training=False, forecast=True, mu=scalars['mu'], std=scalars['std'])

        # make predictions
        print('*********** Inference ***********')
        fx_output = fx.inference(fx_ds) 
        fx_output = np.array(fx_output).squeeze()

        # create a forecast dataframe
        fx_df = pd.DataFrame(fx_output, columns=['p0', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p100'])
        fx_df['timestamp'] = df.timestamp
        fx_df = fx_df.set_index('timestamp')
        y_pred = fx_df['p50'].values
        # N = 200
        # start = np.random.randint(0, fx_df.shape[0]-N)
        # end   = start + N
        crps = crps_loss_dist(fx_df[['p0', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p100']].values, df['net_load'].values)
        crps_ref = get_crps_from_site_id(self.site_id)
        crpss = 1 - crps / crps_ref
        error_mean = (fx_df['p50'].values - df['net_load'].values).mean()
        error_std = (fx_df['p50'].values - df['net_load'].values).std()
        plt.figure(figsize=(10, 6))
        plt.title(f'Day-Ahead (24hr) Net Load Probabilistic Forecast', fontsize=16)
        plt.ylabel('Net Load (MW)', fontsize=16)
        
        # conver the timestamp to local time in OR
        df['local_time']= pd.DatetimeIndex(df.timestamp).tz_convert('US/Pacific')

        a = 0.1
        for p in [0, 10, 20, 30, 40]: 
            # plot the probability with increasing transparency, fill between the 0 and 100th percentile
            
            plt.fill_between(df.local_time, fx_df[f'p{p}'].values, fx_df[f'p{p+10}'].values, alpha=a, color='tab:blue')
            a += 0.1

        a -= 0.1 # keep the same color for 50+60
        for p in [ 50, 60, 70, 80, 90]:
            plt.fill_between(df.local_time, fx_df[f'p{p}'].values, fx_df[f'p{p+10}'].values, alpha=a, color='tab:blue')
            a -= 0.1

        plt.plot(df.local_time, df['net_load'].values, label='Actual', color='black')
        plt.plot(df.local_time, fx_df[f'p50'].values, label=f'24hr Forecast', color='tab:red', linestyle='--', alpha=0.5)

        # rotate the x-axis labels
        plt.xticks(rotation=45)

        plt.grid(True)
        plt.legend(fontsize=12)
        plt.savefig(f'fig/NLF_example.png', bbox_inches='tight', dpi=300)
        plt.show()

    def train_model(self, site_id=None, smoketest=False):
        model_path = None
        self.model.train()
        
        train_ds = NetLoadForecastDataset(self.train_df, self.site_id, self.window_1to1, training=True)
        scalars = train_ds.get_scalars()
        # register mu and std as model parameters so they are saved with the model
        test_ds = NetLoadForecastDataset(self.test_df, self.site_id, self.window_1to1, training=False, mu=scalars['mu'], std=scalars['std'])
        val_ds = NetLoadForecastDataset(self.val_df, self.site_id, self.window_1to1, training=False, mu=scalars['mu'], std=scalars['std']) 


        # BATCH SIZE
        batch_size = 32  # smaller batch sizes are better for training, but may take longer

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        learning_rate = 1e-3
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        def train_one_epoch(epoch_index, tb_writer):
            running_loss = 0.
            last_loss = 0.

            # Here, we use enumerate(training_loader) instead of
            # iter(training_loader) so that we can track the batch
            # index and do some intra-epoch reporting
            for i, data in enumerate(train_dl):
                # Every data instance is an input + label pair
                inputs, labels = data

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = self.model(inputs).to(device)

                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                # N = batch_size * 100
                N = 1000 
                if i % N == (N - 1):
                    last_loss = running_loss / float(N) # loss per batch
                    print('  batch {} loss: {}'.format((i + 1)*batch_size, last_loss))
                    tb_x = epoch_index * len(train_dl) + i + 1
                    tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                    running_loss = 0.

            return last_loss

        # Per EPOCH activity
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp_hms = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        date = datetime.now().strftime('%Y-%m-%d')
        writer = SummaryWriter(f'runs/nn.{site_id}.{timestamp_hms}')
        epoch_number = 0

        if smoketest:
            EPOCHS = 1
        else:
            EPOCHS = 64

        best_vloss = 1_000_000.

        train_crps = []
        val_crps = []
        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = train_one_epoch(epoch_number, writer)


            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            n = 0
            with torch.no_grad():
                for i, vdata in enumerate(test_dl):
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss
                    n += 1

                avg_vloss = running_vloss / n 

            train_crps.append(avg_loss)
            val_crps.append(avg_vloss)
            print('LOSS: training {} validation {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = f'models/{self.model_name}.{site_id}.{timestamp_hms}.{epoch_number}'
                
                if not smoketest:
                    torch.save(self.model.state_dict(), model_path)

            epoch_number += 1



        self.model_path = model_path
        crps_ref = get_crps_from_site_id(site_id)
        if smoketest:
            print(f'Best CRPS loss: {best_vloss}')
            print(f'CRPS skill: {1 - best_vloss/crps_ref}')
        elif self.model_path is not None:
            # print(f"'{self.model_path}', # CRPSS: {best_vloss:0.3f}/CRPSS: {1 - best_vloss/crps_ref:0.3f}")
            print(f"'{self.model_path}', # MSE: {best_vloss:0.3f}")
            print(f'Note that this model needs to be added to the model registry manually')
            save_scalers(train_ds.get_scalars(), self.model_path)
            self.model.load_state_dict(torch.load(self.model_path)) # load the best model for subsequent plotting.
        else:
            raise Exception('No model was trained. Model path is None')

        plt.plot(train_crps, label='train')
        plt.plot(val_crps, label='val')
        plt.legend()
        plt.grid(True)
        plt.title('Training Loss vs Validation Loss:')
        plt.ylabel('CRPS')
        plt.xlabel('Epoch')
        plt.show()


    def get_model_path(self):
        """ Returns the path to the best model for this site """
        return self.model_path

    def compute_validation_error(self)->tuple[np.ndarray, np.ndarray]:
        """ Computes the validation error for the best model for this site 
        Returns:
            (error_mean, error_std)
        """
        scalars = load_scalars(self.model_path)
        # register mu and std as model parameters so they are saved with the model
        val_ds = NetLoadForecastDataset(self.val_df, self.site_id, self.window_1to1, training=False, forecast=True, mu=scalars['mu'], std=scalars['std'])
        y_pred, y_true = predict(self.model, val_ds)

        # get the validattion data timstamps
        timestamps = self.val_df.timestamp.values # TODO: add a get_timestamps() method to the dataset class

        df = pd.DataFrame(y_pred.squeeze(), columns=['y_pred'])
        df['y_true'] = y_true.squeeze()
        df['timestamp'] = timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['site_id'] = self.site_id 
        df['error'] = df['y_pred'] - df['y_true']
        df['abs_error'] = np.abs(df['error'])
        df['abs_pct_error'] = np.abs(df['error']) / df['y_true']

        hourly_error_mean = df.groupby(df.timestamp.dt.hour)['error'].mean()
        hourly_error_std = df.groupby(df.timestamp.dt.hour)['error'].std()
        if False:
            # create a subplot historgram for every hour and plot the error distribution
            fig, axs = plt.subplots(4, 6, figsize=(20, 10))
            for i, ax in enumerate(axs.flatten()):
                ax.hist(df[df.timestamp.dt.hour==i]['error'], bins=20, density=True)
                ax.set_title(f'Hour {i}')
            plt.show()
        return hourly_error_mean, hourly_error_std 

    def inference(self, dataset): # site_id, model, dataset):
        """ Expecting of output of model is a single point measurement: 
                mu = model(inputs)
            Output is a tensor of shape [batch, 11] predicting the net loat at each probability
        """
        fx_dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        self.model.eval()
        fx_output = []

        # get the emperical mean and std for the validation data and use it to estimate the gaussian prediction error
        error_mean, error_std = self.compute_validation_error()
        
        with torch.no_grad():
            for i, data in enumerate(fx_dl):
                # Every data instance is an input + label pair
                inputs, labels = data

                # Make predictions for this batch
                y_pred = self.model(inputs).to(device)

                hour = dataset.get_input_timestamp(i).hour
                dist = torch.distributions.normal.Normal(y_pred[:, 0], error_std[hour], validate_args=False) # y_pred may be batched i.e [batch, 2]
                probs = torch.tensor([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]).to(device) # (11, ) 

                probs = probs.repeat(y_pred.shape[0], 1) # [batch, 11]
                fx   = dist.icdf(probs.T).to(device).T # the output is a tensor of shape [batch, 11] predicting the net loat at each probability
                fx_output.append(fx.cpu().detach().numpy())

        return np.array(fx_output)



def get_parameter(model, name: str):
    """ Returns the value of the model parameter with the given name """
    param = model.get_parameter(name).cpu().clone().detach().numpy()
    print(f'get_parameter({name}) = {param}')
    return param

def continuous_ranked_probability_score( obs, fx, fx_prob):
    """Continuous Ranked Probability Score (CRPS) designed for pytorch tensors.
    Expanded from https://solarforecastarbiter-core.readthedocs.io/en/latest/_modules/solarforecastarbiter/metrics/probabilistic.html#continuous_ranked_probability_score

    .. math::

        \\text{CRPS} = \\frac{1}{n} \\sum_{i=1}^n \\int_{-\\infty}^{\\infty}
        (F_i(x) - \\mathbf{1} \\{x \\geq y_i \\})^2 dx

    where :math:`F_i(x)` is the CDF of the forecast at time :math:`i`,
    :math:`y_i` is the observation at time :math:`i`, and :math:`\\mathbf{1}`
    is the indicator function that transforms the observation into a step
    function (1 if :math:`x \\geq y`, 0 if :math:`x < y`). In other words, the
    CRPS measures the difference between the forecast CDF and the empirical CDF
    of the observation. The CRPS has the same units as the observation. Lower
    CRPS values indicate more accurate forecasts, where a CRPS of 0 indicates a
    perfect forecast. [1]_ [2]_ [3]_

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n, d) array_like
        Forecasts (physical units) of the right-hand-side of a CDF with d
        intervals (d >= 2), e.g., fx = [10 MW, 20 MW, 30 MW] is interpreted as
        <= 10 MW, <= 20 MW, <= 30 MW.
    fx_prob : (n, d) array_like
        Probability [%] associated with the forecasts.

    Returns
    -------
    crps : float
        The Continuous Ranked Probability Score, with the same units as the
        observation.

    Raises
    ------
    ValueError
        If the forecasts have incorrect dimensions; either a) the forecasts are
        for a single sample (n=1) with d CDF intervals but are given as a 1D
        array with d values or b) the forecasts are given as 2D arrays (n,d)
        but do not contain at least 2 CDF intervals (i.e. d < 2).
"""
    # match observations to fx shape: (n,) => (n, d)
    if np.ndim(fx) < 2:
        raise ValueError("forecasts must be 2D arrays (expected (n,d), got"
                         f"{np.shape(fx)})")
    elif np.shape(fx)[1] < 2:
        raise ValueError("forecasts must have d >= 2 CDF intervals "
                         f"(expected >= 2, got {np.shape(fx)[1]})")

    n = len(fx)

    ## extend CDF min to ensure obs within forecast support
    ## fx.shape = (n, d) ==> (n, d + 1) # example
    # fx_min = np.minimum(obs, fx[:, 0])
    obs = torch.Tensor(obs)
    fx = torch.Tensor(fx)
    fx_prob = torch.Tensor(fx_prob)
    fx_min = torch.min(obs, fx[:, 0]) 

    fx = torch.hstack([fx_min[:, np.newaxis], fx]) # gcp, I tried but this failed

    fx_prob = torch.hstack([torch.zeros([n, 1]), fx_prob])

    # extend CDF max to ensure obs within forecast support
    # fx.shape = (n, d + 1) ==> (n, d + 2) # example
    idx = (fx[:, -1] < obs)
    fx_max = torch.max(obs, fx[:, -1])

    fx = torch.hstack([fx, fx_max[:, np.newaxis]])

    fx_prob = torch.hstack([fx_prob, torch.full([n, 1], 100)])

    # indicator function:
    # - left of the obs is 0.0
    # - obs and right of the obs is 1.0
    # o = np.where(fx >= obs[:, np.newaxis], 1.0, 0.0)
    try:
        o = torch.where(fx >= obs[:, np.newaxis], 1.0, 0.0)
    except:
        o = torch.where(fx >= obs, 1.0, 0.0)


    # correct behavior when obs > max fx:
    # - should be 0 over range: max fx < x < obs
    o[idx, -1] = 0.0

    # forecast probabilities [unitless]
    f = fx_prob / 100.0

    # integrate along each sample, then average all samples
    # crps = np.mean(np.trapz((f - o) ** 2, x=fx, axis=1))
    crps = torch.mean(torch.trapz((f - o) ** 2, x=fx, dim=1))

    return crps

def ETL_input_data(site_id: str, start: pd.Timestamp, end: pd.Timestamp):
    """ Extract the hrrr data, add features, transform it."""
    hrrr = pd.read_csv('../hrrr/data/hrrr.forecast.MASTER.csv', parse_dates=True)
    hrrr = hrrr[ hrrr['site_id'] == site_id ]
    hrrr['timestamp'] = pd.to_datetime(hrrr['timestamp'])

    if start is not None:
        hrrr = hrrr[ hrrr['timestamp'] >= start ]
    if end is not None:
        hrrr = hrrr[ hrrr['timestamp'] <= end ]

    hrrr = hrrr.drop_duplicates(subset=['timestamp'])
    hrrr_hourly = hrrr.copy()
    hrrr_hourly = hrrr.set_index('timestamp')
    hrrr_hourly = hrrr_hourly.resample('H').interpolate(method='linear')
    df = hrrr_hourly.reset_index()

    # add features
    ####################
    
    input_columns = ['refc_Maximum/Composite radar reflectivity_atmosphere_dB', 'vis_Visibility_surface_m', 'gust_Wind speed (gust)_surface_m s**-1', 'sp_Surface pressure_surface_Pa', 't_Temperature_surface_K', 'tcc_Total Cloud Cover_boundaryLayerCloudLayer_%', 'lcc_Low cloud cover_lowCloudLayer_%', 'mcc_Medium cloud cover_middleCloudLayer_%', 'hcc_High cloud cover_highCloudLayer_%', 'tcc_Total Cloud Cover_atmosphere_%', 'ulwrf_Upward long-wave radiation flux_nominalTop_W m**-2', 'dswrf_Downward short-wave radiation flux_surface_W m**-2', 'dlwrf_Downward long-wave radiation flux_surface_W m**-2', 'uswrf_Upward short-wave radiation flux_surface_W m**-2', 'ulwrf_Upward long-wave radiation flux_surface_W m**-2', 'vbdsf_Visible Beam Downward Solar Flux_surface_W m**-2', 'vddsf_Visible Diffuse Downward Solar Flux_surface_W m**-2', 'uswrf_Upward short-wave radiation flux_nominalTop_W m**-2', 'r_Relative humidity_isothermZero_%', 'pres_Pressure_isothermZero_Pa', 'dayofweek', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos']
    all_columns = ['timestamp'] + input_columns

    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['year'] = pd.to_datetime(df['timestamp']).dt.year
    df['dayofyear'] = pd.to_datetime(df['timestamp']).dt.dayofyear
    
    # Compute the sin and cos of the hour to capture the cyclical nature of time
    hours_in_day = 24.0
    days_in_year = 365.2425
    df['hour_sin'] = np.sin(df.hour * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df.hour * (2 * np.pi / 24))
    df['year_sin'] = np.sin(df.dayofyear * hours_in_day * (2 * np.pi / (days_in_year * hours_in_day)))
    df['year_cos'] = np.cos(df.dayofyear * hours_in_day * (2 * np.pi / (days_in_year * hours_in_day)))

    # if self.forecast:
        # all_columns = ['timestamp'] + input_columns
    df = df[all_columns]
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.isna().sum().sum() > 0:
        raise ValueError(f'There are still {df.isna().sum().sum()} missing values in the dataframe!')
    else:
        print('*******************************************')
        print('There are no missing values in the dataframe!')
        print('df.shape: ', df.shape)
        print('df.timestamp.min(): ', df.index.min())
        print('df.timestamp.max(): ', df.index.max())
        print('*******************************************')
    return df 

def crps_loss(y_pred, y):
    """ compute the CRPS loss for a batch of forecasts.
        y_pred = [mu, std]
        y = [obs]
    """
    fx, fx_prob = create_gaussian_forecast(y_pred[0], y_pred[1])
    fx_prob = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    fx_prob = np.broadcast_to(fx_prob, fx.shape)
    try:
        crps = continuous_ranked_probability_score( y, fx, fx_prob) 
    except:
        crps = 0.5
    return crps

def crps_loss_dist(fx, y):
    """ compute the CRPS loss for a batch of forecasts.
        fx = an array of net load forecasts for 11 quantiles [batch, 11]
        y = observations [batch, 1]
    """
    probs = torch.tensor([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]) # .to(device) # (11, ) 
    probs = probs.repeat(fx.shape[0], 1) # [batch, 11]
    fx_prob = probs*10.
    y = y.squeeze() # turn [batch, 1] into [batch, ]
    crps = continuous_ranked_probability_score( y, fx, fx_prob) 
    return crps

def get_crps_from_site_id(site_id)->float:
    df = pd.read_csv('data/forecast.reference.all.crps.csv')
    df = df[df['site_id'] == site_id]
    timestamp_max = df['timestamp'].max()
    df = df[df['timestamp'] == timestamp_max]
    return df['crps_ref'].item()

def get_forecast_starttime(site_id)->pd.Timestamp:
    """ Get the start time for the forecast (in UTC). 
        This is the time that the hrrr forecast should be observed.
        The final forecast submitted to SFA should be +1 hour than this.
        OUTPUT: pd.Timestamp (UTC) The output is the starttime of the hrrr observation, one day ahead, adjusted for local timezone.
    """
    tz_offset = {
            '90c2a42c-f0ad-11ed-94b4-5edf5e2b3336': 4, # 'GA': 
            '8568f10f-eb8f-11ed-a556-128dcacebd72': 5, # 'TX': 
            '5ebb4527-edbd-11ed-bf8d-128dcacebd72': 7, # 'OR': 
            'c639b1f3-eb8f-11ed-802e-aec5a60999dc': 10, # 'HI': 
        } 

    # get the current date in UTC at 00:00:00Z
    date = pd.Timestamp.now(tz='UTC').floor('D')
    return date + pd.Timedelta(days=1) + pd.Timedelta(hours=tz_offset[site_id])

  
def create_forecast(site_id):
    """ Create a forecast for the day ahead.
        # site_id = '90c2a42c-f0ad-11ed-94b4-5edf5e2b3336' # GA
        # site_id = '8568f10f-eb8f-11ed-a556-128dcacebd72' # TX
        # site_id = '5ebb4527-edbd-11ed-bf8d-128dcacebd72' # OR
        # site_id = 'c639b1f3-eb8f-11ed-802e-aec5a60999dc' # HI
    """
    model_path = None
    for path in best_models:
        if site_id in path:
            model_path = path
            break

    print('Best model: ', model_path)

    fx = SfaForecastNnModel(site_id=site_id, model_path=model_path, model_name=MODEL_NAME)


    # create a DateTimeIndex for the forecast
    start = get_forecast_starttime(site_id=site_id)
    end = start + pd.Timedelta(hours=23)
    print(f'forecast start = {start}, end = {end}')

    # create a df from the master data and try to m
    # data = pd.read_csv('data/data.MASTER.csv')
    data = pd.read_csv('../hrrr/data/hrrr.forecast.MASTER.csv')
    data = data[data['site_id'] == site_id]
    data = ETL(data)
    if 'timestamp' not in data.columns:
        data.reset_index(inplace=True)
    data.timestamp = pd.to_datetime(data.timestamp)
    data = data[data['timestamp'] >= start]
    data = data[data['timestamp'] <= end]

    scalars = load_scalars(fx.model_path)
    # register mu and std as model parameters so they are saved with the model
    fx_ds = NetLoadForecastDataset(data, site_id, fx.window_1to1, training=False, forecast=True, mu=scalars['mu'], std=scalars['std'])
    assert (len(fx_ds) == 24)

    # make predictions
    print('*********** Inference ***********')
    # fx_output = inference(fx.site_id, fx.model, fx_ds)
    fx_output = fx.inference(fx_ds)

    fx_output = np.array(fx_output).squeeze()
    print(f'fx_output.shape = {fx_output.shape}')
    print()
    assert (fx_output.shape == (24,11))

    # create a forecast dataframe
    fx_df = pd.DataFrame(fx_output, columns=['p0', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p100'])
    fx_df['timestamp'] = data.timestamp.values + pd.Timedelta(hours=1)
    fx_df['timestamp'] = pd.to_datetime(fx_df['timestamp'], utc=True)
    fx_df['site_id'] = site_id
    print('*********** Forecast Dataframe***********') 
    print(fx_df)

    date = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f'data/forecast.{fx.model_name}.{date}.{site_id}.csv'
    fx_df.to_csv(filename, index=False)
    print(f'NN Forecast saved to {filename}')
    print('Enter the following command to upload the forecast to the database:')
    print(f'$ python sfa.py --upload-forecast --file {filename}')

def predict(model, dataset):
    """ Expecting of output of model is a single point measurement: 
            mu = model(inputs)
        Output is a tensor of shape [batch, 11] predicting the net loat at each probability
    """
    fx_dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    fx_output = []
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for i, data in enumerate(fx_dl):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Make predictions for this batch
            y_pred = model(inputs).to(device)
            y_true = labels.to(device)

            y_pred_list.append(y_pred.cpu().detach().numpy())
            y_true_list.append(y_true.cpu().detach().numpy())

    return np.array(y_pred_list), np.array(y_true_list)


def inference_gaussian(model, dataset):
    """ Expecting of model is mu, std = model(inputs)
        Output is a tensor of shape [batch, 11] predicting the net loat at each probability
    """
    fx_dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    fx_output = []
    with torch.no_grad():
        for i, data in enumerate(fx_dl):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Make predictions for this batch
            y_pred = model(inputs).to(device)
            dist = torch.distributions.normal.Normal(y_pred[:, 0], y_pred[:, 1], validate_args=False) # y_pred may be batched i.e [batch, 2]
            probs = torch.tensor([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]).to(device) # (11, ) 
            probs = probs.repeat(y_pred.shape[0], 1) # [batch, 11]
            fx   = dist.icdf(probs.T).to(device).T # the output is a tensor of shape [batch, 11] predicting the net loat at each probability
            fx_output.append(fx.cpu().detach().numpy())

    return np.array(fx_output)

class TestNetLoadForecast(unittest.TestCase):

    def setUp(self):
        # self.site_id = CLI_SITE_ID
        self.site_id = '90c2a42c-f0ad-11ed-94b4-5edf5e2b3336' # GA
        # site_id = '8568f10f-eb8f-11ed-a556-128dcacebd72' # TX
        # site_id = '5ebb4527-edbd-11ed-bf8d-128dcacebd72' # OR
        # site_id = 'c639b1f3-eb8f-11ed-802e-aec5a60999dc' # HI

        model_path = None
        for path in best_models:
            if self.site_id in path:
                model_path = path
                break

        print('Best model: ', model_path)

        self.fx = SfaForecastNnModel(site_id=self.site_id, model_path=model_path, model_name=MODEL_NAME)

    def test_create_dataset(self):
        """ Create a dataset"""
        train_ds = Dataset()
        train_ds = NetLoadForecastDataset(self.fx.train_df, self.fx.site_id, self.fx.window_1to1, training=True)
        scalars = train_ds.get_scalars()
        # test that the array, mu, is not empty
        self.assertTrue(len(scalars['mu']) > 0)
        self.assertTrue(len(scalars['std']) > 0)
        self.assertEquals(train_ds.input.shape[1], len(scalars['mu']))
        print(f'scalars = {scalars}')

    def test_create_window(self):
        # # indices should be features: [0], labels: [0]
        window_0= WindowGenerator(input_width=1, offset=-1, label_width=1, input_columns=['value', 'quality_flag'], label_columns=['net_load', 'rolling_std'])
        print(window_0)
        self.assertEqual(window_0.input_indices,[0]) 
        self.assertEqual(window_0.label_indices,[0]) 

    def test_continuous_rank_probability_score(self):
        df = pd.read_csv('data/forecast.reference.all.crps.csv')
        df = df[df['site_id'] == self.site_id]
        obs = df['value'].values
        fx = df[['0.0', '10.0', '20.0', '30.0', '40.0', '50.0', '60.0', '70.0', '80.0', '90.0', '100.0']].values
        fx_prob = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        fx_prob = np.broadcast_to(fx_prob, fx.shape)
        print(f'obs.shape = {obs.shape}')
        print(f'fx.shape = {fx.shape}')
        print(f'fx_prob.shape = {fx_prob.shape}')

        self.assertTrue(obs.shape == (fx.shape[0], ) )
        self.assertTrue(fx.shape == fx_prob.shape)
        crps = continuous_ranked_probability_score( obs, fx, fx_prob)
        print(f'CRPS = {crps}')
        self.assertTrue(np.isclose(crps, 0.0, atol=1e-1))

    def test_create_probabalistic_forecast(self):
        """ Create a probabalistic forecast for any time series """

        # get the available observation data
        df = pd.read_csv('data/forecast.reference.all.crps.csv')
        df = df[df['site_id'] == self.site_id]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # timestamp = pd.Timestamp('2023-06-08 11:00:00Z') # this works
        timestamp = pd.Timestamp('2023-06-08 12:00:00Z') # this works with data interpolation

        # interpolate master dataframe
        # for each observation, get the input data
        obs = df[ df.timestamp == timestamp]['value'].values
        self.assertTrue(len(obs) == 1)

        # create a forecast with mu and std for each observation
        # hrrr = pd.read_csv('../hrrr/data/hrrr.forecast.MASTER.csv')
        data = pd.read_csv('data/data.MASTER.csv')
        data = data[data['site_id'] == self.site_id]
        data = ETL(data)
        if 'timestamp' not in data.columns:
            data.reset_index(inplace=True)
            data.timestamp = pd.to_datetime(data.timestamp)
            
        data = data[data['timestamp'] == timestamp]
        print(f'data = {data}')
        self.assertTrue(len(data) == 1)
        
        # create a probabalistic forecast for each observation
        scalars = load_scalars(model_path=self.fx.model_path)
        ds = NetLoadForecastDataset(data, self.site_id, self.fx.window_1to1, training=False, mu=scalars['mu'], std=scalars['std'])
        y_actual, y_pred = predict(self.fx.model, ds)
        print(f'y_actual={y_actual}, y_pred = {y_pred}, rel_error = {(y_pred - y_actual)/y_actual}')
        self.assertTrue(len(y_pred) == 1)

        mu = y_pred
        std = 0.03 # for testing only
        fx, fx_prob = create_gaussian_forecast(mu, std)

        # create a CRPS score for each observation
        fx_prob = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        fx_prob = np.broadcast_to(fx_prob, fx.shape)
        print(f'obs.shape = {obs.shape}')
        print(f'fx.shape = {fx.shape}')
        print(f'fx_prob.shape = {fx_prob.shape}')

        self.assertTrue(obs.shape == (fx.shape[0], ) )
        self.assertTrue(fx.shape == fx_prob.shape)
        import pdb; pdb.set_trace()
        crps = continuous_ranked_probability_score( obs, fx, fx_prob) 
        self.assertTrue(np.isclose(crps, 0.0, atol=1e-1))
        print('Forecasted CRPS = ', crps)

        # get the skill score for the forecast
        ref_fx = pd.read_csv('data/forecast.reference.all.crps.csv')
        ref_fx = ref_fx[ref_fx['site_id'] == self.site_id]
        ref_fx['timestamp'] = pd.to_datetime(ref_fx['timestamp'])
        ref_fx = ref_fx[ref_fx['timestamp'] == timestamp]
        ref_fx = ref_fx[['0.0', '10.0', '20.0', '30.0', '40.0', '50.0', '60.0', '70.0', '80.0', '90.0', '100.0']].values

        cprs_ref = continuous_ranked_probability_score( obs, ref_fx, fx_prob)
        skill = 1.0 - (crps/cprs_ref)
        print(f'CRPS = {crps}, CRPS_ref = {cprs_ref}, skill = {skill}')
        self.assertTrue(skill < 1.0)

    def test_create_forecast(self):
        
        # create a DateTimeIndex for the forecast
        start = get_forecast_starttime(site_id=self.site_id)
        end = start + pd.Timedelta(hours=23)
        print(f'forecast start = {start}, end = {end}')

        # create a df from the master data and try to m
        # data = pd.read_csv('data/data.MASTER.csv')
        data = pd.read_csv('../hrrr/data/hrrr.forecast.MASTER.csv')
        data = data[data['site_id'] == self.site_id]
        data = ETL(data)
        if 'timestamp' not in data.columns:
            data.reset_index(inplace=True)
        data.timestamp = pd.to_datetime(data.timestamp)
        data = data[data['timestamp'] >= start]
        data = data[data['timestamp'] <= end]
        print(f'data = {data}')

        if False:
            self.fx.train_model()
        scalars = load_scalars(self.fx.model_path)
        # register mu and std as model parameters so they are saved with the model
        fx_ds = NetLoadForecastDataset(data, self.site_id, self.fx.window_1to1, training=False, forecast=True, mu=scalars['mu'], std=scalars['std'])
        self.assertTrue(len(fx_ds) == 24)
        # should batch size be 1?

        # make predictions
        print('*********** Inference ***********')
        fx_output = self.fx.inference(fx_ds)
        # _, fx_output = predict(self.fx.model, fx_ds)

        fx_output = np.array(fx_output).squeeze()
        print(f'fx_output.shape = {fx_output.shape}')
        print()
        self.assertTrue(fx_output.shape == (24,11))

        # create a forecast dataframe
        fx_df = pd.DataFrame(fx_output, columns=['p0', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p100'])
        fx_df['timestamp'] = data.timestamp.values + pd.Timedelta(hours=1)
        fx_df['timestamp'] = pd.to_datetime(fx_df['timestamp'], utc=True)
        fx_df['site_id'] = self.site_id
        print('*********** Forecast Dataframe***********') 
        print(fx_df)

        date = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/forecast.{self.fx.model_name}.{date}.{self.site_id}.csv'
        fx_df.to_csv(filename, index=False)
        print(f'NN Forecast saved to {filename}')
        print('Enter the following command to upload the forecast to the database:')
        print(f'$ python sfa.py --upload-forecast --file {filename}')

        return fx_df 

    def test_train_model(self):
        """ Train a model """
        if False:
            self.fx.train_model(site_id=self.site_id, smoketest=True)

    def test_get_crps_score(self):
        print('*********** Test Get CRPS_REF ***********')
        crps = get_crps_from_site_id(self.site_id)
        self.assertTrue(crps > -6.)
        self.assertTrue(crps < 2.)

    def test_get_validation_error(self):
        print('*********** Test Get Validation Error ***********')
        model = self.fx.model
        error_mean, error_std = self.fx.compute_validation_error()
        print(f'error_mean = {error_mean}, error_std = {error_std}')
        error_std_mean = np.median(error_std)
        error_std_std = np.std(error_std)
        std_75 = np.percentile(error_std, 75)
        print(f'error_std_mean = {error_std_mean}, error_std_std = {error_std_std}, std_75 = {std_75}')
        plt.hist(error_std, bins=20)
        plt.show()
        self.assertTrue(error_mean is not None)
        self.assertTrue(error_std is not None)

class TestOneShot(unittest.TestCase):

    def setUp(self):
        # self.site_id = CLI_SITE_ID
        self.site_id = '90c2a42c-f0ad-11ed-94b4-5edf5e2b3336' # GA
        # site_id = '8568f10f-eb8f-11ed-a556-128dcacebd72' # TX
        # site_id = '5ebb4527-edbd-11ed-bf8d-128dcacebd72' # OR
        # site_id = 'c639b1f3-eb8f-11ed-802e-aec5a60999dc' # HI

        model_path = None
        for path in best_models:
            if self.site_id in path:
                model_path = path
                break

        print('Best model: ', model_path)

        self.fx = SfaForecastNnModel(site_id=self.site_id, model_path=model_path, model_name=MODEL_NAME)

    def test_get_timestamps(self):
        """ Get the timestamps of the label data."""
                # create a DateTimeIndex for the forecast
        start = pd.Timestamp('2023-07-11 4:00:00Z') + pd.Timedelta(hours=1)
        end = start + pd.Timedelta(hours=4)
        print(f'forecast start = {start}, end = {end}')

        data = pd.read_csv('../hrrr/data/hrrr.forecast.MASTER.csv')
        data = data[data['site_id'] == self.site_id]
        data = ETL(data)
        if 'timestamp' not in data.columns:
            data.reset_index(inplace=True)
        data.timestamp = pd.to_datetime(data.timestamp)
        data = data[data['timestamp'] >= start]
        data = data[data['timestamp'] <= end]
        print(f'data = {data}')

        scalars = load_scalars(self.fx.model_path)
        # register mu and std as model parameters so they are saved with the model
        fx_ds = NetLoadForecastDataset(data, self.site_id, self.fx.window_1to1, training=False, forecast=True, mu=scalars['mu'], std=scalars['std'])
        self.assertTrue( fx_ds.get_input_timestamp(0) == start)
        self.assertTrue( fx_ds.get_input_timestamp(-1) == end)

if __name__ == '__main__':
        # create a list of command line arguments passed to the script
    import argparse
    import sys
    args = sys.argv
    parser = argparse.ArgumentParser(description='Bistory.AI SolarForecaster_API', 
                                     epilog='Example: bitstory/sfa $ python nlf_nn.py -s GA -tr  \n $ python sfa_nn.py -s OR --plot')
    parser.add_argument('-d', '--dev', action='store_true', help='Run Development Tests')
    parser.add_argument('-t', '--test', action='store_true', help='Run Unitests')

    # create a command line arguement like "--test GA" to run the test for GA
    parser.add_argument('-s', '--site', type=str, default='GA', help='Site Name to train model for')
    parser.add_argument('-tr', '--train', action='store_true', help='Train model for site_id')
    parser.add_argument('--smoketest', action='store_true', help='Run smoketest')
    parser.add_argument('-cf', '--create-forecast', action='store_true', help='Create forecast for site_id')
    parser.add_argument('--plot', action='store_true', help='Plot forecast for site_id (python nlf_nn.py -s OR --plot)')

    args = parser.parse_args()

    if args.test:
        if args.site:
            CLI_SITE_ID = args.site
            print('Running Unit Tests of SFA...')
            runner = unittest.TextTestRunner()
            system_tests = unittest.TestLoader().loadTestsFromTestCase(TestNetLoadForecast)
            one_shot_tests = unittest.TestLoader().loadTestsFromTestCase(TestOneShot)
            # runner.run(system_tests)
            runner.run(one_shot_tests)
        else:
            log.error('Please specify a site_id to run the unit tests on.')

    if args.create_forecast and args.site:
        site_id = site_id_map[args.site]
        create_forecast(site_id=site_id)

    if args.train and args.site:

        site_id = site_id_map[args.site]
        print(f'Training model for site_id={site_id}')

        model_path = None


        for path in best_models:
            if site_id in path:
                model_path = path
                break

        print('Best model: ', model_path)

        fx = SfaForecastNnModel(site_id=site_id, model_path=model_path, model_name=MODEL_NAME, training=True)
        if args.smoketest:
            fx.train_model(site_id=site_id, smoketest=True)
        else:
            fx.train_model(site_id=site_id, smoketest=False)

        fx.plot_prediction()

    if args.plot and args.site:

        site_id = site_id_map[args.site]
        for path in best_models:
            if site_id in path:
                model_path = path
                break

        print('Best model: ', model_path)


        fx = SfaForecastNnModel(site_id=site_id, model_path=model_path, model_name=MODEL_NAME)
        fx.plot_prediction()

    else:
        print('You must provide a site_id to plot')

