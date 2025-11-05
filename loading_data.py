import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#import tensorflow as tf
import scipy.io
import h5py
# from tensorflow import keras
# from tensorflow.keras.layers import Dense, LSTM
# from tensorflow.keras.models import Sequential
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.optimizers import Adam
# import tensorflow.keras.backend as K
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from IPython.display import clear_output
from scipy.signal import savgol_filter
import gc

def load_data(file_name,pickle_name,bn):
  f = h5py.File(file_name,"r")
  print(list(f.keys()))
  batch=f.get("batch")
  num_cells = batch['summary'].shape[0]
  bat_dict = {}
  for i in range(num_cells):
    cl = f[batch['cycle_life'][i,0]][()] #ici on a remplacé ".value" par "[()]", et dans toutes les lignes suivantes
    policy = f[batch['policy_readable'][i,0]][()].tobytes()[::2].decode()
    summary_IR = np.hstack(f[batch['summary'][i,0]]['IR'][0,:].tolist())
    summary_QC = np.hstack(f[batch['summary'][i,0]]['QCharge'][0,:].tolist())
    summary_QD = np.hstack(f[batch['summary'][i,0]]['QDischarge'][0,:].tolist())
    summary_TA = np.hstack(f[batch['summary'][i,0]]['Tavg'][0,:].tolist())
    summary_TM = np.hstack(f[batch['summary'][i,0]]['Tmin'][0,:].tolist())
    summary_TX = np.hstack(f[batch['summary'][i,0]]['Tmax'][0,:].tolist())
    summary_CT = np.hstack(f[batch['summary'][i,0]]['chargetime'][0,:].tolist())
    summary_CY = np.hstack(f[batch['summary'][i,0]]['cycle'][0,:].tolist())
    summary = ({'IR': summary_IR, #création d'un 1er dict avec les données "résumées", càd 1 donnée par cycle
               'QC': summary_QC,
               'QD': summary_QD,
               'Tavg': summary_TA,
               'Tmin': summary_TM,
               'Tmax': summary_TX,
               'chargetime': summary_CT,
               'cycle': summary_CY})
    cycles = f[batch['cycles'][i,0]]
    cycle_dict = {}
    for j in range(cycles['I'].shape[0]):
        I = np.hstack((f[cycles['I'][j,0]][()]))
        Qc = np.hstack((f[cycles['Qc'][j,0]][()]))
        Qd = np.hstack((f[cycles['Qd'][j,0]][()]))
        Qdlin = np.hstack((f[cycles['Qdlin'][j,0]][()]))
        T = np.hstack((f[cycles['T'][j,0]][()]))
        Tdlin = np.hstack((f[cycles['Tdlin'][j,0]][()]))
        V = np.hstack((f[cycles['V'][j,0]][()]))
        dQdV = np.hstack((f[cycles['discharge_dQdV'][j,0]][()]))
        t = np.hstack((f[cycles['t'][j,0]][()]))
        cd = ({'I': I,
               'Qc': Qc,
               'Qd': Qd,
               'Qdlin': Qdlin,
               'T': T,
               'Tdlin': Tdlin,
               'V':V,
               'dQdV': dQdV,
               't':t})
        cycle_dict[str(j)] = cd

    cell_dict = ({'cycle_life': cl,
                 'charge_policy':policy,
                  'summary': summary,
                  'cycles': cycle_dict})
    key = "b"+bn+"c" + str(i)
    bat_dict[key]=   cell_dict


  df=pd.DataFrame(bat_dict)

  with open(pickle_name,'wb') as fp:
        pickle.dump(bat_dict,fp)
  return df

path_to_file="C:\\Users\\Ivan\\Desktop\\DipData\\"

# b1=load_data(path_to_file+"2017-05-12_batchdata_updated_struct_errorcorrect.mat","batch1.pkl","1")
# b2=load_data(path_to_file+"2017-06-30_batchdata_updated_struct_errorcorrect.mat","batch2.pkl","2")
# b3=load_data(path_to_file+"2018-04-12_batchdata_updated_struct_errorcorrect.mat","batch3.pkl","3")

batch1 = pickle.load(open(r'batch1.pkl', 'rb'))
numBat1 = len(batch1.keys())
batch2 = pickle.load(open(r'batch2.pkl','rb'))
numBat2 = len(batch2.keys())
batch3 = pickle.load(open(r'batch3.pkl','rb'))
numBat3 = len(batch3.keys())
numBat = numBat1 + numBat2 + numBat3

bat_dict = {**batch1, **batch2, **batch3}
bat_dict_keys = bat_dict.keys()

Ic = []
Vc = []
Tc = []
Id = []
Vd = []
Td = []
SOH = []
qd = []
charge_policy = []
ce = []
for i in bat_dict_keys:
    clear_output(wait=True)
    #print('cell:', i)
    cell = bat_dict[i]
    num_cycle = len(cell['summary']['cycle'])
    for j in range(1, num_cycle):
        start_discharge = np.where(cell['cycles'][str(j)]['I'] < -0.05)
        start_discharge = start_discharge[0][0]

        end_discharge = np.where(cell['cycles'][str(j)]['I'] < -3.9)
        end_discharge = end_discharge[0][-1]

        start_charge = np.where(cell['cycles'][str(j)]['I'] > 0)
        end_charge = start_charge[0][-1]
        start_charge = start_charge[0][0]

        Id.append(cell['cycles'][str(j)]['I'][start_discharge:end_discharge])
        Vd.append(cell['cycles'][str(j)]['V'][start_discharge:end_discharge])
        Td.append(cell['cycles'][str(j)]['T'][start_discharge:end_discharge])

        Ic.append(cell['cycles'][str(j)]['I'][start_charge:end_charge])
        Vc.append(cell['cycles'][str(j)]['V'][start_charge:end_charge])
        Tc.append(cell['cycles'][str(j)]['T'][start_charge:end_charge])
        charge_policy.append(cell['charge_policy'])

        QC = cell['summary']['QC'][j]
        QD = cell['summary']['QD'][j]
        qd.append(QD)
        initial_Qd = cell['summary']['QD'][1]
        SOH.append((QD) / initial_Qd)
        dict_key = "{}_cycle_{}".format(i, j)
        ce.append(dict_key)
print('all cycles from batches 1, 2 and 3 were browsed')

df=pd.DataFrame({"Ic":Ic,"Id":Id,"Vc":Vc,"Vd":Vd,"Tc":Tc,"Td":Td,"charge_policy":charge_policy,"SOH":SOH,"QD":qd})
df.index=ce
# print('\ndf.index:')
# print(df.index)
cycle_dataset_df=df.T
# print('\ncycle_dataset_df:')
# print(cycle_dataset_df)
cycle_dataset_df=cycle_dataset_df.T
cycle_dataset_df=cycle_dataset_df.T

cells=[]
k=["1","2","3"]
for j in k:
    for i in range(48):
      filter_col=[]
      data=[]
      filter_col = [col for col in cycle_dataset_df if col.startswith("b"+j+"c"+str(i)+"_cycle")]
      data=cycle_dataset_df[filter_col].T
      data=data.reset_index(drop=True)
      if (len(data)>0):
            cells.append(data)
#print(f'\nlen(cells): {len(cells)}')
cells[0]["SOH"].plot()
for i in range(len(cells)):
    cells[i]["SOH"]=savgol_filter(cells[i]["SOH"], 90, 2)

cells_array = np.array(cells, dtype=object)
#np.save("celles.npy", cells_array, allow_pickle=True)
train_size=int(0.6*len(cells))
train_data=cells[0:train_size]
test_data=cells[train_size:len(cells)]
test_data_array = np.array(test_data, dtype=object)
#np.save("test_data",test_data_array, allow_pickle=True)

cycle_dataset_df=train_data
minic=min(cycle_dataset_df[0]["Ic"][0])
maxic=max(cycle_dataset_df[0]["Ic"][0])
minid=min(cycle_dataset_df[0]["Id"][0])
maxid=max(cycle_dataset_df[0]["Id"][0])
minvc=min(cycle_dataset_df[0]["Vc"][0])
maxvc=max(cycle_dataset_df[0]["Vc"][0])
minvd=min(cycle_dataset_df[0]["Vd"][0])
maxvd=max(cycle_dataset_df[0]["Vd"][0])
mintc=min(cycle_dataset_df[0]["Tc"][0])
maxtc=max(cycle_dataset_df[0]["Tc"][0])
mintd=min(cycle_dataset_df[0]["Td"][0])
maxtd=max(cycle_dataset_df[0]["Td"][0])

for i in range(len(cycle_dataset_df)):
    for j in range(len(train_data[i])):
        if(min(cycle_dataset_df[i]["Ic"][j])<minic):
            minic=min(cycle_dataset_df[i]["Ic"][j])
        if(min(cycle_dataset_df[i]["Id"][j])<minid):
            minid=min(cycle_dataset_df[i]["Id"][j])

        if(max(cycle_dataset_df[i]["Ic"][j])>maxic):
            maxic=max(cycle_dataset_df[i]["Ic"][j])
        if(min(cycle_dataset_df[i]["Id"][j])>maxid):
            maxid=max(cycle_dataset_df[i]["Id"][j])


        if(min(cycle_dataset_df[i]["Vc"][j])<minvc):
            minvc=min(cycle_dataset_df[i]["Vc"][j])
        if(min(cycle_dataset_df[i]["Vd"][j])<minvd):
            minvd=min(cycle_dataset_df[i]["Vd"][j])
        if(max(cycle_dataset_df[i]["Vc"][j])>maxvc):
            maxvc=max(cycle_dataset_df[i]["Vc"][j])
        if(min(cycle_dataset_df[i]["Vd"][j])>maxvd):
            maxvd=max(cycle_dataset_df[i]["Vd"][j])


        if(min(cycle_dataset_df[i]["Tc"][j])<mintc):
            mintc=min(cycle_dataset_df[i]["Tc"][j])
        if(min(cycle_dataset_df[i]["Td"][j])<mintd):
            mintd=min(cycle_dataset_df[i]["Td"][j])
        if(max(cycle_dataset_df[i]["Tc"][j])>maxtc):
            maxtc=max(cycle_dataset_df[i]["Tc"][j])
        if(min(cycle_dataset_df[i]["Td"][j])>maxtd):
            maxtd=max(cycle_dataset_df[i]["Td"][j])

for i in range(len(cycle_dataset_df)):
    for j in range(len(cycle_dataset_df[i])):
        cycle_dataset_df[i]["Ic"][j]=(cycle_dataset_df[i]["Ic"][j]-minic)/(maxic-minic)
        cycle_dataset_df[i]["Id"][j]=(cycle_dataset_df[i]["Id"][j]-minid)/(maxid-minid)

        cycle_dataset_df[i]["Vc"][j]=(cycle_dataset_df[i]["Vc"][j]-minvc)/(maxvc-minvc)
        cycle_dataset_df[i]["Vd"][j]=(cycle_dataset_df[i]["Vd"][j]-minvd)/(maxvd-minvd)

        cycle_dataset_df[i]["Tc"][j]=(cycle_dataset_df[i]["Tc"][j]-mintc)/(maxtc-mintc)
        cycle_dataset_df[i]["Td"][j]=(cycle_dataset_df[i]["Td"][j]-mintd)/(maxtd-mintd)

train_data=cycle_dataset_df
cycle_dataset_df=test_data
for i in range(len(cycle_dataset_df)):
    for j in range(len(cycle_dataset_df[i])):
        cycle_dataset_df[i]["Ic"][j]=(cycle_dataset_df[i]["Ic"][j]-minic)/(maxic-minic)
        cycle_dataset_df[i]["Id"][j]=(cycle_dataset_df[i]["Id"][j]-minid)/(maxid-minid)

        cycle_dataset_df[i]["Vc"][j]=(cycle_dataset_df[i]["Vc"][j]-minvc)/(maxvc-minvc)
        cycle_dataset_df[i]["Vd"][j]=(cycle_dataset_df[i]["Vd"][j]-minvd)/(maxvd-minvd)

        cycle_dataset_df[i]["Tc"][j]=(cycle_dataset_df[i]["Tc"][j]-mintc)/(maxtc-mintc)
        cycle_dataset_df[i]["Td"][j]=(cycle_dataset_df[i]["Td"][j]-mintd)/(maxtd-mintd)

test_data=cycle_dataset_df
min_max=np.array([minic,maxic,minid,maxid,minvc,maxvc,minvd,maxvd,mintc,maxtc,mintd,maxtd])
print(f'min_max: {min_max}')
#np.save("min_max",min_max)

train_data_array = np.array(train_data, dtype=object)
#np.save("train_data_withnormalization",train_data_array, allow_pickle=True)

test_data_array = np.array(test_data, dtype=object)
np.save("test_data_withnormalization",test_data_array, allow_pickle=True)


# split a univariate sequence into samples
def split_sequence(X, y, sequence, n_steps_in, n_steps_out):
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x.drop(["SOH", "charge_policy"], axis=1).to_numpy())
        y.append(seq_y["SOH"].to_numpy())
    return X, y


def generatedata(train_data, test_data, input_step, output_step):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i in range(len(train_data)):
        X_train, Y_train = split_sequence(X_train, Y_train, train_data[i], input_step, output_step)
    for i in range(len(test_data)):
        X_test, Y_test = split_sequence(X_test, Y_test, test_data[i], input_step, output_step)
    X1 = np.array(X_train)
    Y1 = np.array(Y_train)
    X2 = np.array(X_test)
    Y2 = np.array(Y_test)
    t = np.zeros((len(X1), input_step, 6, 500))
    for k in range(len(X1)):
        for i in range(input_step):
            for x in range(6):
                for j in range(len(X1[k][i][x])):
                    if (j >= 500):
                        break
                    else:
                        t[k][i][x][j] = X1[k][i][x][j]
    X1 = t
    np.save("X1_" + str(input_step), X1)
    del X1
    del t
    gc.collect()

    t = np.zeros((len(X2), input_step, 6, 500))
    for k in range(len(X2)):
        for i in range(input_step):
            for x in range(6):
                for j in range(len(X2[k][i][x])):
                    if (j >= 500):
                        break
                    else:
                        t[k][i][x][j] = X2[k][i][x][j]
    X2 = t
    np.save("X2_" + str(input_step), X2)
    del X2
    del t
    gc.collect()
    np.save("Y2_" + str(output_step), Y2)
    np.save("Y1_" + str(output_step), Y1)

# input_step=10
# output_step=10
# generatedata(train_data,test_data,input_step,output_step)
#
# input_step=25
# output_step=25
# generatedata(train_data,test_data,input_step,output_step)
#
# input_step=25
# output_step=50
# generatedata(train_data,test_data,input_step,output_step)


print('5.9C(60%)-3.1C-newstructure')