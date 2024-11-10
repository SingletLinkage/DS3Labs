import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def I_1(df: pd.DataFrame):
    temp = sorted(df.loc[:, 'temperature'])
    len_temp = len(temp)
    # mean, min, max, median, std_dev
    mean_temp = 0
    std_dev_temp = 0
    min_temp = temp[0]
    max_temp = temp[0]
    median_temp = 0

    for _ in temp:
        mean_temp += _
        if _ < min_temp:
            min_temp = _
        if _ > max_temp:
            max_temp = _
        
        std_dev_temp += _** 2

    mean_temp /= len_temp
    std_dev_temp = (std_dev_temp / len_temp - mean_temp**2)**0.5
    if len_temp % 2 == 0:
        median_temp = (temp[len_temp // 2] + temp[len_temp // 2 - 1]) / 2
    else:
        median_temp = temp[len_temp // 2]

    print(f'''
    The statistical measures of Temperature attribute are: 
    mean=\t{mean_temp:.2f},
    median=\t{median_temp:.2f},
    maximum=\t{min_temp:.2f}, 
    minimum=\t{max_temp:.2f}, 
    STD=\t{std_dev_temp:.2f}.
    ''')

def Pearson_Correlation(x:np.ndarray , y: np.ndarray):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return np.sum((x - mean_x) * (y - mean_y)) / np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))

def I_2(df: pd.DataFrame):
    # temperature,humidity,pressure,rain,lightavg,lightmax,moisture
    attributes = ['temperature', 'humidity', 'pressure', 'rain', 'lightavg', 'lightmax', 'moisture']
    corr_df = pd.DataFrame(columns=attributes, index=attributes)
    for y_att in attributes:
        for x_att in attributes:
            x = df.loc[:, x_att].to_numpy()
            y = df.loc[:, y_att].to_numpy()
            corr_df.loc[x_att, y_att] = Pearson_Correlation(x, y)
    
    print(corr_df.to_string())

    l_avg = corr_df.loc[:, 'lightavg']
    # between 0.6 and 1
    print("Redundant Attributes wrt lightavg: ", l_avg[(abs(l_avg) >= 0.6) & (abs(l_avg) < 1)].index.to_list())

def I_3(df: pd.DataFrame):
    t12_humidity = df[df['stationid'] == 't12']['humidity'].to_numpy()
    min_humidity = np.min(t12_humidity)
    max_humidity = np.max(t12_humidity)
    bin_size = 5
    bins = np.arange(min_humidity, max_humidity+bin_size, bin_size)

    # print(bins)
    heights = np.zeros_like(bins)
    for h in t12_humidity:
        heights[int((h - min_humidity) // bin_size)] += 1
    
    plt.bar(bins+bin_size/2, heights, width=bin_size, edgecolor='black')
    plt.xticks(bins)
    plt.xlabel('Humidity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Humidity for station t12')
    plt.show()


def II_1(df: pd.DataFrame, display: bool=True):
    attributes = ['temperature', 'humidity', 'pressure', 'rain', 'lightavg', 'lightmax', 'moisture']
    df.dropna(subset=['stationid'], inplace=True)  # dropping only rows with missing stationid
    df.dropna(subset=attributes, thresh=int(2*len(attributes)/3), inplace=True)  # dropping rows with missing values in any of the attributes
    # df.reset_index(drop=True, inplace=True)  # to reset index - otherwise index will have missing values
    if display:
        print(df.to_string())
        df.to_csv('replaced_df.csv')

def II_2(df: pd.DataFrame, display: bool=True):
    II_1(df, False)
    indices = df.index
    # indices = iter(df.index)

    prev_index = -1
    for idx, row in df.iterrows():
        # row['dates'] = '08-07-2018' -> split('-') = ['08', '07', '2018'] -> map(int, ...) and [::-1] = [2018, 7, 8] -> unpack
        # date = datetime.date(*map(int, row['dates'].split('-')[::-1]))

        for col in row[pd.isnull(list(row))].index:
            cur_date = datetime.date(*map(int, row['dates'].split('-')[::-1]))

            if idx == 0 or df.loc[idx, 'stationid'] != df.loc[prev_index, 'stationid']:
                prev_val = None
            else:
                prev_val = df.loc[prev_index, col]
                prev_date = datetime.date(*map(int, df.loc[prev_index, 'dates'].split('-')[::-1]))
            
            next_idx = idx+1
            while next_idx < len(df) and (next_idx not in indices or pd.isnull(df.loc[next_idx, col])):
                next_idx += 1
            
            # next_idx = next(indices, None)
            # while next_idx is not None and pd.isnull(df.loc[next_idx, col]):
            #     next_idx = next(indices, None)


            if next_idx >= len(df) or df.loc[idx, 'stationid'] != df.loc[next_idx, 'stationid']:
                next_val = None
            else:
                next_val = df.loc[next_idx, col]
                next_date = datetime.date(*map(int, df.loc[next_idx, 'dates'].split('-')[::-1]))

            if prev_val is None:
                df.loc[idx, col] = next_val
            elif next_val is None:
                df.loc[idx, col] = prev_val
            else:
                df.loc[idx, col] = linear_interpolate(prev_date, prev_val, next_date, next_val, cur_date)
            
        prev_index = idx
    
    # Linear interpolation done
    if not display:
        return df
    
    # print(df.to_string())
    # Some actual stuff now
    ori_df = pd.read_csv('landslide_data_original.csv')
    attributes = ['temperature', 'humidity', 'pressure', 'rain', 'lightavg', 'lightmax', 'moisture']
    parameters = ['mean', 'median', 'std_dev']
    ori_stats = pd.DataFrame(columns=attributes, index=parameters)
    miss_stats = pd.DataFrame(columns=attributes, index=parameters)

    for col in attributes:
        ori_vals = ori_df.loc[:, col]
        miss_vals = df.loc[:, col]

        ori_stats.loc[:, col] = calculate_stats(ori_vals)
        miss_stats.loc[:, col] = calculate_stats(miss_vals)
    

    print("Stats from original file: \n", ori_stats.to_string(), '\n')

    print("Stats from file with missing data: \n", miss_stats.to_string())

    # np.set_printoptions(precision=3, threshold=np.inf, suppress=True)
    # (np.hstack([np.reshape(ori_df.loc[df.index, 'lightavg'], (-1, 1)), np.reshape(df.loc[:, 'lightavg'], (-1, 1))]))
    # for i in df.index:
    #     if ori_df.loc[i, 'lightavg'] != df.loc[i, 'lightavg']:
    #         print(i, ori_df.loc[i, 'lightavg'], df.loc[i, 'lightavg'])

    rmse_vals = dict()
    for col in attributes:
        rmse_vals[col] = rmse(ori_df.loc[:, col], df.loc[:, col])
    
    print(rmse_vals)
    
    plt.bar(rmse_vals.keys(), rmse_vals.values())
    plt.xlabel('Attributes')
    plt.ylabel('RMSE')
    plt.title('RMSE of attributes')
    plt.show()

def rmse(y_true, y_pred):
    rmse = 0
    for i in y_pred.index:
        rmse += (y_true[i] - y_pred[i])**2
    return (rmse/len(y_true))**0.5

def calculate_stats(x: np.ndarray):
    mean = 0
    median = 0
    std_dev = 0
    len_x = len(x)

    for x_i in x:
        mean += x_i
        std_dev += x_i**2

    mean /= len_x
    std_dev= (std_dev/ len_x - mean**2)**0.5
    if len_x % 2 == 0:
        median = (x[len_x // 2] + x[len_x // 2 - 1]) / 2
    else:
        median = x[len_x // 2]
    
    return mean, median, std_dev

def linear_interpolate(x1, y1, x2, y2, x):
    return y1 + (y2-y1)*(x-x1)/(x2-x1)

def III_1(df: pd.DataFrame, display:bool=True):
    II_2(df, display=False)

    attributes = ['temperature', 'humidity', 'pressure', 'rain', 'lightavg', 'lightmax', 'moisture']
    if display:
        fig, ax = plt.subplots(3,3)
        for idx, col in enumerate(attributes):
            ax[idx//3][idx%3].boxplot(df[col])
            ax[idx//3][idx%3].set_title(col)
        plt.show()

def III_2(df:pd.DataFrame, display:bool=True):
    III_1(df, False)

    attributes = ['temperature', 'humidity', 'pressure', 'rain', 'lightavg', 'lightmax', 'moisture']
    stats = df.describe()

    for col in attributes:
        q1 = stats.at['25%', col]
        q3 = stats.at['75%', col]
        iqr = q3-q1

        for idx in df[col].index:
            if q1 - 1.5*iqr < df.at[idx, col] < q3 + 1.5*iqr:
                pass
            else:
                df.at[idx, col] = stats.at['50%', col]

    if display:
        fig, ax = plt.subplots(3,3)
        for idx, col in enumerate(attributes):
            ax[idx//3][idx%3].boxplot(df[col])
            ax[idx//3][idx%3].set_title(col)
        plt.show()

    # Yes Even after changing outliers to median, new outliers are created... this is because the q3 gets shifted to a new position leading to change in iqr and hence, change of q3 + 1.5*iqr limits
    # The resultant outliers are a result of new limits

def IV_1(df: pd.DataFrame):
    # Outlier corrected data
    III_2(df, False)
    min_val = 5
    max_val = 12
    attributes = ['temperature', 'humidity', 'pressure', 'rain', 'lightavg', 'lightmax', 'moisture']

    for col in attributes:
        _temp = df.loc[:, col].to_numpy()
        df.loc[:, col] = (_temp - min(_temp))/(max(_temp) - min(_temp)) * (max_val - min_val) + min_val
    
    print(df.describe())
    
def IV_2(df: pd.DataFrame):
    # Outlier corrected data
    III_2(df, False)

    # Before Z score normalization
    print("Before Normalization: \n", df.describe().loc[['mean', 'std'], :])
    attributes = ['temperature', 'humidity', 'pressure', 'rain', 'lightavg', 'lightmax', 'moisture']

    for col in attributes:
        _temp = df.loc[:, col].to_numpy()
        df.loc[:, col] = (_temp - np.mean(_temp))/np.std(_temp)
    
    np.set_printoptions(precision=5, suppress=True)
    print("After normalization: \n", df.describe())


if __name__ == '__main__':
    df = pd.read_csv('landslide_data_original.csv')
    # I_3(df)
    df = pd.read_csv('landslide_data_miss.csv')
    # III_1(df)
    IV_1(df)