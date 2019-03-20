
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import math
import random
import datetime
import matplotlib.pyplot as plt


# In[ ]:


data_dict = {i:pd.read_csv('data'+str(i)+'.csv') for i in range(len(data_ids))}


# In[ ]:


for m in range(len(data_dict)):
    print(len(data_dict[m].index))


# In[ ]:


if data_source == 'nelson':
  for m in range(77,83):
    data_dict[m] = data_dict[m].drop_duplicates(subset=['DTS', 'FUTURE_PRICE',
           'VOLUME_BID', 'SKEWNESS_BID', 'KURTOSIS_BID',
           'VOLUME_ASK', 'SKEWNESS_ASK', 'KURTOSIS_ASK'])


# In[ ]:


for i in range(len(data_dict.keys())):
  try:
    data_dict[i] = data_dict[i][data_dict[0].columns]
  except KeyError:
    data_dict[i].columns = data_dict[0].columns


# In[ ]:


for i,df in data_dict.items():
  if data_source == 'guru':
    data_dict[i]['Datetimestamp'] = pd.to_datetime(df['Datetimestamp'])
    data_dict[i].set_index('Datetimestamp', drop=True, append=False, inplace=True)
    data_dict[i] = df.between_time('09:00','11:30')
  else:
    data_dict[i]['DTS'] = pd.to_datetime(df['DTS'])
    data_dict[i].set_index('DTS', drop=True, append=False, inplace=True)
    data_dict[i] = df.between_time('08:00','10:30')
  data_dict[i].reset_index(inplace=True)


# In[ ]:


def normalize(df):
  index_lst = []
  index_pairs = []
  for i in range(len(df.index)):
    if (df['Kurtosis_BID'].iloc[i] == -3.0) and (df['Kurtosis_BID'].iloc[i-1] != -3.0):
      index_lst.append(i)
  for index in index_lst:
    index_pair = None
    count = 1
    while index_pair == None:
      if (df['Kurtosis_BID'].iloc[index+count] < 5.0) and (df['Kurtosis_BID'].iloc[index+count-1] > 5.0):
        index_pair = index+count
        index_pairs.append((index,index_pair))
      count+=1
  
  for pairs in index_pairs[::-1]:
    for n in range(pairs[0],pairs[1]):
      for col in ['TotalMarketDepth','Depth_Bid','Skewness_BID','Kurtosis_BID','Depth_Ask','Skewness_ASK','Kurtosis_ASK']:
        df.at[n, col] = df[col].iloc[pairs[0]-1]     
  return df


# In[ ]:


#Load data into dataframe
for i,data in data_dict.items():
  if data_source == 'guru':
    data_dict[i].columns = ['DTS', 'FUTURE_PRICE', 'BID_VOLUME', 'ASK_VOLUME', 'TICKCOUNT', 'OBV','CV_BID', 'CV_ASK', 'VOLUME_BID','VOLUME_ASK',
                            'SKEWNESS_BID', 'KURTOSIS_BID', 'SKEWNESS_ASK', 'KURTOSIS_ASK'] 
  data_dict[i] = data_dict[i][['DTS','FUTURE_PRICE','BID_VOLUME','ASK_VOLUME','VOLUME_BID','SKEWNESS_BID','KURTOSIS_BID','VOLUME_ASK','SKEWNESS_ASK','KURTOSIS_ASK']]
  data_dict[i].columns = ['DateTimeStamp','Price','Bid_Volume','Ask_Volume','Depth_Bid','Skewness_BID','Kurtosis_BID','Depth_Ask','Skewness_ASK','Kurtosis_ASK']
  data_dict[i]['TotalMarketDepth'] = data_dict[i]['Depth_Bid'] + data_dict[i]['Depth_Ask']
  if data_source == 'guru':
    data_dict[i] = normalize(data_dict[i])
  
  data_dict[i]['BidAskRatio'] = data_dict[i]['Depth_Bid'] / data_dict[i]['Depth_Ask']
  data_dict[i]['AskBidRatio'] = data_dict[i]['Depth_Ask'] / data_dict[i]['Depth_Bid']

  data_dict[i] = data_dict[i][['DateTimeStamp','Price','Bid_Volume','Ask_Volume','BidAskRatio','AskBidRatio','TotalMarketDepth','Depth_Bid','Depth_Ask','Skewness_BID','Skewness_ASK','Kurtosis_BID','Kurtosis_ASK']]
  data_dict[i]['DateTimeStamp'] = pd.to_datetime(data_dict[i]['DateTimeStamp'], format='%Y-%m-%d %H:%M:%S')
  
print(data_dict[0].head())


# In[ ]:


for i in data_dict.keys():
  data_dict[i] = data_dict[i].rename(index=str, columns={'DateTimeStamp':'DateTimeStamp','Price':'Price', 'BidAskRatio':'BidAsk_Ratio','AskBidRatio':'AskBid_Ratio','TotalMarketDepth':'TotalDepth','Depth_Bid':'Depth_Bid','Depth_Ask':'Depth_Ask',
                                                         'Skewness_BID':'MD_SKEW_BID','Skewness_ASK':'MD_SKEW_ASK','Kurtosis_BID':'MD_KURTOSIS_BID','Kurtosis_ASK':'MD_KURTOSIS_ASK'})


# In[ ]:


ticks = 1


# In[ ]:


#add LogReturn and Volatility features
for i in data_dict.keys():
  data_dict[i]['LogReturn_Incremental'] = data_dict[i]['Price'].rolling(2).apply(lambda x: np.log(x[-1]/x[0]))
  data_dict[i]['Depth_LogReturn_Incremental'] = data_dict[i]['TotalDepth'].rolling(2).apply(lambda x: np.log(x[-1]/x[0]))
  data_dict[i]['Depth_Bid_LogReturn_Incremental'] = data_dict[i]['Depth_Bid'].rolling(2).apply(lambda x: np.log(x[-1]/x[0]))
  data_dict[i]['Depth_Ask_LogReturn_Incremental'] = data_dict[i]['Depth_Ask'].rolling(2).apply(lambda x: np.log(x[-1]/x[0]))
  data_dict[i].fillna(0, inplace=True)


# In[ ]:


for i in data_dict.keys():
  print(i,len(data_dict[i].index))
      
for i in data_dict.keys():
  data_dict[i].reset_index(drop = False, inplace = True)
  data_dict[i] = data_dict[i].loc[data_dict[i].index % ticks == 0]
  
print('')
for i in data_dict.keys():
  print(i,len(data_dict[i].index))


# In[ ]:


all_data = [data_dict[i] for i in range(len(data_dict))]


# In[ ]:


for i,df in enumerate(all_data):
  df['DateTimeStamp'] = pd.to_datetime(df['DateTimeStamp'])
  df.set_index('DateTimeStamp',drop=True,inplace=True)
  if data_source == 'guru':
    all_data[i] = df.between_time('09:00','16:45')
  else:
    all_data[i] = df.between_time('08:00','10:30')


# In[ ]:


from scipy.signal import savgol_filter

for m in range(len(all_data)):
  try:
    #all_data[m]['Smooth_Price'] = savgol_filter(all_data[m]['Price'], 101, 2)
    all_data[m]['Smooth_Price'] = savgol_filter(all_data[m]['Price'], min(501,len(all_data[m].index)-1), 1)
    all_data[m]['Smooth_Depth_Bid'] = savgol_filter(all_data[m]['Depth_Bid'], 51, 3)
    all_data[m]['Smooth_Depth_Ask'] = savgol_filter(all_data[m]['Depth_Ask'], 51, 3)
    all_data[m]['Smooth_TotalDepth'] = all_data[m]['Smooth_Depth_Bid'] + all_data[m]['Smooth_Depth_Ask']
    all_data[m]['Smooth_BidAsk_Ratio'] = all_data[m]['Smooth_Depth_Bid'] / all_data[m]['Smooth_Depth_Ask']
    all_data[m]['Smooth_AskBid_Ratio'] = all_data[m]['Smooth_Depth_Ask'] / all_data[m]['Smooth_Depth_Bid']
    all_data[m]['Smooth_MD_SKEW_BID'] = savgol_filter(all_data[m]['MD_SKEW_BID'], 51, 3)
    all_data[m]['Smooth_MD_SKEW_ASK'] = savgol_filter(all_data[m]['MD_SKEW_ASK'], 51, 3)
    all_data[m]['Smooth_MD_KURTOSIS_BID'] = savgol_filter(all_data[m]['MD_KURTOSIS_BID'], 51, 3)
    all_data[m]['Smooth_MD_KURTOSIS_ASK'] = savgol_filter(all_data[m]['MD_KURTOSIS_ASK'], 51, 3)
  except:
    all_data[m]['Smooth_Price'] = []
    all_data[m]['Smooth_Depth_Bid'] = []
    all_data[m]['Smooth_Depth_Ask'] = []
    all_data[m]['Smooth_TotalDepth'] = []
    all_data[m]['Smooth_BidAsk_Ratio'] = []
    all_data[m]['Smooth_AskBid_Ratio'] = []
    all_data[m]['Smooth_MD_SKEW_BID'] = []
    all_data[m]['Smooth_MD_SKEW_ASK'] = []
    all_data[m]['Smooth_MD_KURTOSIS_BID'] = []
    all_data[m]['Smooth_MD_KURTOSIS_ASK'] = []


# In[ ]:


unscaled_prices = [all_data[i]['Smooth_Price'] for i in range(len(all_data))]


# In[ ]:


for i in range(len(all_data)):
  all_data[i].reset_index(inplace=True)
  all_data[i] = all_data[i].replace(np.inf, 0)
  all_data[i].fillna(0, inplace=True)


# In[ ]:


def soft_vote_policy(qvalues,position):
  action_index,qval = qvalues.index(max(qvalues)),max(qvalues)

  if position == 0:
    if qval >= 0:
      action=action_index
    else:
      action=2
  elif position == 1:

    if action_index == 0:
      action = 0
    else:
      action = 2

  else:

    if action_index == 0:
      action = 1
    else:
      action = 2

  return action


# In[ ]:


class Reward():
  def get_reward(self,price, new_price, position, new_position):
    
    if position == 0 and new_position == 2:
      profit = ((new_price - price) * 1000) - 50

    elif position == 0 and new_position == 1:
      profit = ((price - new_price) * 1000) - 50

    elif position == 1 and new_position == 1:
      profit = ((price - new_price) * 1000)

    elif position == 2 and new_position == 2:
      profit = ((new_price - price) * 1000)

    elif position == 1 and new_position == 0:
      profit = 0

    elif position == 2 and new_position == 0:
      profit = 0

    else:
      profit = 0

    return profit

reward = Reward()

def get_position(position, action):
  if action == 0:
    if position == 0: return 2
    elif position == 1: return 0
  elif action == 1:
    if position == 0: return 1
    elif position == 2: return 0
  else:
    return position
  
def get_profit_from_matrix(all_data,unscaled_prices,days):
  profit = 0
  for m in range(days[0],days[1]):
    position = 0
    for n in range(len(all_data[m].index)-1):
      current_position = position
      qvalues = all_data[m].iloc[n][current_position]
      agent_action = soft_vote_policy(qvalues,current_position)
      position = update_position(agent_action,position)

      profit += update_profit(current_position, unscaled_prices[m][n], position, unscaled_prices[m][n+1])
      
  print(profit)

def update_profit(position, price, new_position, new_price):
  if position == 0 and new_position == 2:
      return ((new_price - price) * 1000) - 3.03
  elif position == 0 and new_position == 1:
      return ((price - new_price) * 1000) - 3.03
  elif position == 1 and new_position == 1:
      return ((price - new_price) * 1000)
  elif position == 2 and new_position == 2:
      return ((new_price - price) * 1000)

  elif position == 1 and new_position == 0:
      return - 3.03

  elif position == 2 and new_position == 0:
      return - 3.03
  else:
    return 0  
  
def update_position(action,position):
  if action == 2:
    return position
  elif action == 0:
    if position == 0: return 2
    elif position == 1: return 0
    else:
      print('oh no')
      x = 1/0
  elif action == 1:
    if position == 0: return 1
    elif position == 2: return 0
    else:
      print('uh oh')
      x = 1/0


# In[ ]:


for m in range(len(month_data)):
  for q in range(7):
     all_data[m]['reward'+str(q)] = 0.0


# In[ ]:


def add_rewards(all_data, unscaled_prices):
  for m in range(len(all_data)):
    print(m, end=" ", flush=True)
    all_rewards = []
    for n in range(len(all_data[m].index)-1):
      reward_vals = []
      position_action = [[0,1,2],[0,2],[1,2]]

      for position in range(len(position_action)):
        reward_up = [0 for _ in range(len(position_action[position]))] 
        for action_index, action in enumerate(position_action[position]):
          new_position = get_position(position,action)
          current_reward = reward.get_reward(unscaled_prices[m].iloc[n], unscaled_prices[m].iloc[n+1], position, new_position)
          reward_vals.append(current_reward)
      all_rewards.append(reward_vals)
    for q in range(7):
      all_data[m]['reward'+str(q)] = [rewards_[q] for rewards_ in all_rewards] + [0.0]
  return all_data

def get_qvalues(all_data,days,iterations):
  for i in range(days[0],days[1]):
    print(i, end=" ", flush=True)
    if iterations == 'all':
      for itr in range(len(all_data[i].index)):
        for q in range(7):
          all_data[i]['next'+str(q)] = list(all_data[i]['q'+str(q)][1:].values)+[0]
        all_data[i]['q0'] = all_data[i]['reward0'] + all_data[i][['next'+str(i) for i in range(5,7)]].max(axis=1)
        all_data[i]['q1'] = all_data[i]['reward1'] + all_data[i][['next'+str(i) for i in range(3,5)]].max(axis=1)
        all_data[i]['q2'] = all_data[i]['reward2'] + all_data[i][['next'+str(i) for i in range(0,3)]].max(axis=1)
        all_data[i]['q3'] = all_data[i]['reward3'] + all_data[i][['next'+str(i) for i in range(0,3)]].max(axis=1)
        all_data[i]['q4'] = all_data[i]['reward4'] + all_data[i][['next'+str(i) for i in range(3,5)]].max(axis=1)
        all_data[i]['q5'] = all_data[i]['reward5'] + all_data[i][['next'+str(i) for i in range(0,3)]].max(axis=1)
        all_data[i]['q6'] = all_data[i]['reward6'] + all_data[i][['next'+str(i) for i in range(5,7)]].max(axis=1)
    else:
      for itr in range(iterations):
        for q in range(7):
          all_data[i]['next'+str(q)] = list(all_data[i]['q'+str(q)][1:].values)+[0]
        all_data[i]['q0'] = all_data[i]['reward0'] + all_data[i][['next'+str(i) for i in range(5,7)]].max(axis=1)
        all_data[i]['q1'] = all_data[i]['reward1'] + all_data[i][['next'+str(i) for i in range(3,5)]].max(axis=1)
        all_data[i]['q2'] = all_data[i]['reward2'] + all_data[i][['next'+str(i) for i in range(0,3)]].max(axis=1)
        all_data[i]['q3'] = all_data[i]['reward3'] + all_data[i][['next'+str(i) for i in range(0,3)]].max(axis=1)
        all_data[i]['q4'] = all_data[i]['reward4'] + all_data[i][['next'+str(i) for i in range(3,5)]].max(axis=1)
        all_data[i]['q5'] = all_data[i]['reward5'] + all_data[i][['next'+str(i) for i in range(0,3)]].max(axis=1)
        all_data[i]['q6'] = all_data[i]['reward6'] + all_data[i][['next'+str(i) for i in range(5,7)]].max(axis=1)
  return all_data


# In[ ]:


for i in range(len(all_data)):
  for q in range(7):
    all_data[i]['q'+str(q)] = [0 for _ in range(len(all_data[i].index))]


# In[ ]:


for i in range(len(all_data)):
  print(i,len(all_data[i].index))


# In[ ]:


all_data = add_rewards(all_data, unscaled_prices)


# In[ ]:


all_data = get_qvalues(month_data,(0,len(all_data)),'all')


# In[ ]:


for m in range(0,len(all_data)):
    all_data[m].to_csv('day_'+str(m)+'_processed.csv')

