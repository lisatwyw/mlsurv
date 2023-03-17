# OHE=1; exec( open('mlsurv_bench2.py').read()  )

model_path = '~/scratch/opensource/eicu/mdls_val/'

import sys, os
import pandas as pd
import numpy as np
SEED=1
np.random.seed(SEED)

import random
import argparse
from sklearn.metrics import roc_curve, auc,confusion_matrix, average_precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

#from scipy import interp
from numpy import interp
from sklearn.model_selection import KFold
from importlib import reload

from lifelines.utils import concordance_index    
import mlsurv_utils
reload(mlsurv_utils)

from data_extraction import utils
reload(utils)
from models import models
reload(models)
from models import data_reader
reload(data_reader)
import config 
reload(config)

parser = argparse.ArgumentParser(description="Create data for root")
parser.add_argument('--eicu_dir', type=str, help="Path to root folder containing all the patietns data")
parser.add_argument('--output_dir', type=str, help="Directory where the created data should be stored.")
args, _ = parser.parse_known_args()

from sklearn.metrics import f1_score   
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from sksurv.metrics  import *

def make_surv_array(t,f,breaks):
    """Transforms censored survival data into vector format that can be used in Keras.
    Arguments
        t: Array of failure/censoring times.
        f: Censoring indicator. 1 if failed, 0 if censored.
        breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Returns
        Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
    """
    n_samples=t.shape[0]
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5*timegap
    y_train = np.zeros((n_samples,n_intervals*2))
    for i in range(n_samples):
        if f[i]: #if failed (not censored)
            y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks[1:]) #give credit for surviving each time interval where failure time >= upper limit
            if t[i]<breaks[-1]: #if failure time is greater than end of last time interval, no time interval will have failure marked
                y_train[i,n_intervals+np.where(t[i]<breaks[1:])[0][0]]=1 #mark failure at first bin where survival time < upper break-point
        else: #if censored
            y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks_midpoint) #if censored and lived more than half-way through interval, give credit for surviving the interval.
    return y_train


    
def cd_auc( n_intervals ):
    def loss(y_true, y_pred):      
        
        yy=y_true[:,0:n_intervals]
        
        #todo: convert y_true to structured array
        y_st = mlsurv_utils.convert_to_structured( Ys[ tids[se] ][0], Ys[tids[se]][1]  )                     
        e=mlsurv_utils.cumulative_dynamic_auc(  y_st, y_st, -np.prod(y_pred,1), n_intervals, verbose=False, tied_tol=1e-8, ci=.95, debug=False )            
        return e[1]
    
    return loss 

def surv_likelihood_rnn(n_intervals):
    """Create custom Keras loss function for neural network survival model. Used for recurrent neural networks with time-distributed output.
       This function is very similar to surv_likelihood but deals with the extra dimension of y_true and y_pred that exists because of the time-distributed output.
    """
    def loss(y_true, y_pred):
        
        print( y_true.shape ) 
        cens_uncens = 1. + y_true[0,:,0:n_intervals] * (y_pred-1.) #component for all patients
        uncens = 1. - y_true[0,:,n_intervals:2*n_intervals] * y_pred #component for only uncensored patients
        return K.sum(-K.log(K.clip(K.concatenate((cens_uncens,uncens)),K.epsilon(),None)),axis=-1) #return -log likelihood
    return loss

try:
    os.chdir( '/home/lisat/scratch/opensource/eicu/eICU_Benchmark/' )
except:
    os.chdir( '/home/lisat/scratch/eicu/eICU_Benchmark/' )
            

def batch_generator_v2(config, X, Y, Y_time2ev, batch_size=1024, rng=np.random.RandomState(0), train=True, phen=True,base=True):
    if train:
        while True:
            # print( Y, Y.shape )
            c1 = np.where(Y.squeeze() ==1 )[0]
            c2 = np.where(Y.squeeze() ==0 )[0]

            print( 'size of class1:', len(c1),'size of class2:', len(c2) )                 
            all_index = list(range(X.shape[0]))
            while len(all_index) > (batch_size*0.2):

                idx1 = rng.choice( c1, int(batch_size/2))
                idx2 = rng.choice( c2, int(batch_size/2))
                idx = np.hstack( (idx1, idx2))

                x_batch = X[idx]
                y_batch = Y[idx]                
                y_t2ev_batch = Y_time2ev[idx]
                
                idx = list(set(all_index) - set(idx))
                data_selection = list(zip(x_batch, y_batch, y_t2ev_batch))
                random.shuffle(data_selection)
                x_batch, y_batch, y_t2ev_batch = zip(*data_selection)

                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch)
                y_t2ev_batch = np.array(y_t2ev_batch)
                                
                if not phen:
                    y_batch = np.expand_dims(y_batch, axis=-1)
                    y_t2ev_batch = np.expand_dims(y_t2ev_batch, axis=-1)
                
                
                
                
                
                if config.num and config.cat:
                    x_nc = x_batch[:, :, NCAT:]
                    x_cat = x_batch[:,:, :NCAT].astype(int)
                    if config.ohe:
                        one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429), dtype=int)
                        one_hot = (np.eye(429)[x_cat].sum(2) > 0).astype(int)
                        x_cat = one_hot
                    yield [x_nc, x_cat], (y_batch, y_t2ev_batch)

                elif not config.num and config.cat:
                    
                    x_cat = x_batch[:,:, :NCAT].astype(int)
                    if config.ohe:
                        one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429), dtype=int)
                        one_hot = (np.eye(429)[x_cat].sum(2) > 0).astype(int)
                        x_cat = one_hot
                    yield x_cat, (y_batch, y_t2ev_batch)
                else:
                    yield x_batch, (y_batch, y_t2ev_batch)
    else:
        while True:
            X = np.array(X)
            Y = np.array(Y)
            Y_time2ev = np.array(Y_time2ev)
            
            if not phen:
                Y = np.expand_dims(Y, axis=-1)
                Y = np.expand_dims(Y_time2ev, axis=-1)
                
            for i in range(0, len(Y), batch_size):
                st_idx = i
                end_idx = st_idx + batch_size

                if config.num and config.cat:
                    x_nc = X[:, :, NCAT:]
                    x_cat = X[:, :, :NCAT]
                    yield [x_nc[st_idx:end_idx], x_cat[st_idx:end_idx]], (Y[st_idx:end_idx], Y_time2ev[st_idx:end_idx])
                else:
                    yield X[st_idx:end_idx], (Y[st_idx:end_idx], Y_time2ev[st_idx:end_idx])

                    

def batch_generator(config, X, Y, batch_size=1024, rng=np.random.RandomState(0), train=True, phen=True,base=True):
    
    """
    Assumes X stores all categorical variables first, followed by numeric starting at index NCAT       
    """
    
    if train:
        while True:
            # print( Y, Y.shape )
            c1 = np.where(Y.squeeze() ==1 )[0]
            c2 = np.where(Y.squeeze() ==0 )[0]

            print( 'size of class1:', len(c1),'size of class2:', len(c2) )                 
            all_index = list(range(X.shape[0]))
            while len(all_index) > (batch_size*0.2):

                idx1 = rng.choice( c1, int(batch_size/2))
                idx2 = rng.choice( c2, int(batch_size/2))
                idx = np.hstack( (idx1, idx2))

                x_batch = X[idx]
                y_batch = Y[idx]
                idx = list(set(all_index) - set(idx))
                data_selection = list(zip(x_batch, y_batch))
                random.shuffle(data_selection)
                x_batch, y_batch = zip(*data_selection)

                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch)
                if not phen:
                    y_batch = np.expand_dims(y_batch, axis=-1)
                # if base:
                #     if config.task =='dec':
                #         y_batch = np.squeeze(y_batch,axis=-1)

                if config.num and config.cat:
                    x_nc = x_batch[:, :, NCAT:]
                    x_cat = x_batch[:,:, :NCAT].astype(int)
                    if config.ohe:
                        one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429), dtype=int)
                        one_hot = (np.eye(429)[x_cat].sum(2) > 0).astype(int)
                        x_cat = one_hot
                    yield [x_nc, x_cat], y_batch

                elif not config.num and config.cat:
                    
                    x_cat = x_batch[:,:, :NCAT].astype(int)
                    if config.ohe:
                        one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429), dtype=int)
                        one_hot = (np.eye(429)[x_cat].sum(2) > 0).astype(int)
                        x_cat = one_hot
                    yield x_cat, y_batch
                else:
                    yield x_batch, y_batch
    else:
        while True:
            X = np.array(X)
            Y = np.array(Y)
            if not phen:
                Y = np.expand_dims(Y, axis=-1)
            for i in range(0, len(Y), batch_size):
                st_idx = i
                end_idx = st_idx + batch_size

                if config.num and config.cat:
                    x_nc = X[:, :, NCAT:]
                    x_cat = X[:, :, :NCAT]
                    yield [x_nc[st_idx:end_idx], x_cat[st_idx:end_idx]], Y[st_idx:end_idx]
                else:
                    yield X[st_idx:end_idx], Y[st_idx:end_idx]

                    
                    
                    
                    
                    
def read_data_v2(config, train, test, VAL ):
    
    trn_inds = config.trn_inds
    val_inds = config.val_inds
        
     
    nrows_train0 = train[1]
    nrows_train = list(np.asarray( nrows_train0 )[ trn_inds ] )
    nrows_val   = list(np.asarray( nrows_train0 )[ val_inds ] )
    
    nrows_test = test[1]
    BASE = True
    
    PH=config.task=='phen' 
    
    
    if PH:
        n_labels = len(config.col_phe)
    elif config.task in ['dec', 'mort', 'rlos']:
        n_labels = 1
    elif config.task == 'surv':    
        n_labels = 2
    
    X_train = X_train0 = train[0][trn_inds, :, 1:-n_labels] # column 0 is patientunitstayid
    X_val   = X_val0 = train[0][val_inds, :, 1:-n_labels]   # column 0 is patientunitstayid    
    X_test  = test[0][:, :, 1:-n_labels] # column 0 patientunitstayid

    if config.num and not config.cat:
        X_train = X_train
        X_test = X_test
        X_val = X_val

    elif not config.num and config.cat:
        X_train = X_train[:,:,:NCAT]
        X_val   = X_val[:,:,:NCAT]
        X_test  = X_test[:,:,:NCAT]    
        
    Y_train = train[0][trn_inds, 0, -2:]
    Y_test  = test[0][:, 0, -2:] 
    Y_val   = train[0][val_inds, 0, -2:]         
    #print( "read_data_v2 > validation Y:", Y_val.shape )
    
    Ys= {} 
    Ys['trn'] = (Y_train[:, 1], Y_train[:, 0])
    Ys['val'] = (Y_val[:, 1], Y_val[:, 0])
    Ys['tst'] = (Y_test[:, 1], Y_test[:, 0])
     

    if config.task == 'surv':            
        
        Y_train = make_surv_array( Y_train[:, 1], Y_train[:, 0], breaks)
        Y_val  = make_surv_array( Y_val[:, 1], Y_val[:, 0], breaks)
        Y_test = make_surv_array( Y_test[:, 1], Y_test[:, 0], breaks)
        print( "make_surv_array > validation Y:", Y_val )
        
    elif config.task in ['mort', 'phen']:   
         
        Y_train = train[0][trn_inds, 0, -n_labels:]
        Y_test  = test[0][:, 0, -n_labels:] 
        Y_val   = train[0][val_inds, 0, -n_labels:] 
    else:
        Y_train = train[0][trn_inds, :, -n_labels:]
        Y_val   = train[0][val_inds, :, -n_labels:]        
        Y_test  = test[0][:, :, -n_labels:]

    nt=train[0].shape[0]
            
    X_train = list(zip(X_train, nrows_train))    
    X_train, nrows_train = zip(*X_train)
    X_train = np.array(X_train)

    X_val = list(zip(X_val, nrows_val))    
    X_val, nrows_val = zip(*X_val)
    X_val = np.array(X_val)    
        
    if config.task != 'surv':       
        
        Y_train = Y_train.astype(int)
        Y_val   = Y_val.astype(int)
        Y_test  = Y_test.astype(int)
    
    #train_gen = batch_generator_v2(config, X_train, Y_train, Ys['trn'][0], batch_size=config.batch_size, train=True, phen=PH,base=BASE)
    train_gen = batch_generator(config, X_train, Y_train, batch_size=config.batch_size, train=True, phen=PH,base=BASE)
    train_steps = np.ceil(len(X_train)/config.batch_size)

    val_gen   = batch_generator(config, X_val, Y_val, batch_size=config.batch_size, train=True,phen=PH,base=BASE)
    #val_gen   = batch_generator_v2(config, X_val, Y_val, Ys['val'][0], batch_size=config.batch_size, train=True,phen=PH,base=BASE)
    val_steps = np.ceil(len(X_val)/config.batch_size)
    
    max_time_step = nrows_test
    
    return train_gen, train_steps, val_gen, val_steps.astype(int), (X_test, Y_test), max_time_step, (X_train0, Y_train), (X_val0, Y_val), Ys 




args.task = ''
conf = config.Config(args)
conf.eicu_dir = '/home/lisat/scratch/opensource/eicu/'
conf.output_dir = '/home/lisat/scratch/opensource/eicu/out/'








vars_to_consider = sorted(['glucose', 'Invasive BP Diastolic', 'Invasive BP Systolic',
               'O2 Saturation', 'Respiratory Rate', 'Motor', 'Eyes', 'MAP (mmHg)',
               'Heart Rate', 'GCS Total', 'Verbal', 'pH', 'FiO2', 'Temperature (C)',
               'WBC''s in body fluid','WBC''s in cerebrospinal fluid', 'WBC''s in pericardial fluid', 'WBC''s in peritoneal fluid', \
                           'WBC''s in pleural fluid', 'WBC''s in synovial fluid', 'WBC''s in urine',
               'platelets x 1000', 'PT - INR', 'PTT ratio', 'PTT', 'total bilirubin','creatinine', '24 h urine urea nitrogen',
               'lactate', 'potassium', 'magnesium', 'sodium',  'chloride', 'Ferritin',
               'paCO2', 'paO2' ])# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7937754/ suggests Arterial blood gas pH, etc
        
if ( 'Pat' in globals())==False:
    Pat= pd.read_csv( conf.eicu_dir + 'patient.csv.gz')
    
if ( 'df_data' in globals())==False:
    # df_data = pd.read_csv('/home/lisat/scratch/opensource/eicu/out/all_data.csv')    
    # all_df0 = utils.embedding( conf.output_dir )    
    all_df = utils.prepare_categorical_variables(conf.output_dir, Pat )
        
    def filter_mortality_data(all_df):
        all_df = all_df[all_df.gender != 0]
        all_df = all_df[all_df.hospitaldischargestatus!=2]
        all_df['unitdischargeoffset'] = all_df['unitdischargeoffset']/(1440)
        all_df['itemoffsetday'] = (all_df['itemoffset']/24)
        all_df.drop(columns='itemoffsetday',inplace=True)
                
        # took out ethnicity!
        mort_cols = ['patientunitstayid', 'itemoffset', 'apacheadmissiondx', 'gender',
                    'GCS Total', 'Eyes', 'Motor', 'Verbal',
                    'admissionheight','admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)',
                    'Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
                    'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH', 
                    'unitdischargeoffset','hospitaldischargestatus', 'time2exit_days']                

        all_mort = all_df[mort_cols]
        all_mort = all_mort[all_mort['unitdischargeoffset'] >=2]
        all_mort = all_mort[all_mort['itemoffset']> 0]
        return all_mort
    
    all_mort = filter_mortality_data( all_df )
    df_data = all_mort[ all_mort['itemoffset']<= conf.mort_window ]    
    all_idx = np.array(sorted(list(df_data['patientunitstayid'].unique())))
    
    print( 'NaN survival times?', np.where( np.isnan(df_data.time2exit_days ) )[0] )
    
    

print('\nInput arguments:')
for i in sys.argv:
    print(i)
try:    
    # python mlsurv_bench2.py 2 5 300 16 10 -1 400 1 bilstm 3 64 1 bce 0.3 0 0.001 mort ACT   
    ctn=1
    fd = int( sys.argv[ctn] ); ctn+=1
    FD = int( sys.argv[ctn] ); ctn+=1
    conf.epochs = EP = int( sys.argv[ctn] ); ctn+=1
    conf.batch_size = BS = int( sys.argv[ctn] ); ctn+=1
    VAL= int( sys.argv[ctn] ); ctn+=1
    NT = int( sys.argv[ctn] ); ctn+=1
    MXL= int( sys.argv[ctn] ); ctn+=1
    tk = int( sys.argv[ctn] ); ctn+=1
    ARC= sys.argv[ctn]; ctn+=1
    NL = int( sys.argv[ctn] ); ctn+=1
    NU = int( sys.argv[ctn] ); ctn+=1
    BN = int( sys.argv[ctn] ); ctn+=1
    LOSS = sys.argv[ctn]; ctn+=1
    DO = float( sys.argv[ctn] ); ctn+=1
    OHE= int( sys.argv[ctn] ); ctn+=1    
    LR = float( sys.argv[ctn] ); ctn+=1    
    OUT= sys.argv[ctn]; ctn+=1
    ACT= sys.argv[ctn]; ctn+=1
except:
       
    if ( 'fd' in globals())==False:
        fd = 0
    if ( 'FD' in globals())==False:
        FD = 5 
        
    if ( 'EP' in globals())==False:
        EP = 300    
        
    if ( 'BS' in globals())==False:
        BS = 16        
    if ( 'VAL' in globals())==False:
        VAL = 10 
    if ( 'NT' in globals())==False:
        NT = 16 
    if ( 'MXL' in globals())==False:
        MXL = 400
    if ( 'tk' in globals())==False:
        tk = 1
    if ( 'ARC' in globals())==False:
        ARC ='bilstm'
    if ( 'NL' in globals())==False:
        NL = 3        
    if ( 'NU' in globals())==False:
        NU = 64        
    if ( 'BN' in globals())==False:
        BN = 0     
    if ( 'LOSS' in globals())==False:
        if OUT=='surv':
            LOSS ='surv';
        else:
            LOSS ='bce'                                       
    if ( 'DO' in globals())==False:
        DO = 0.3          
    if ( 'OHE' in globals())==False:
        OHE = 1        
    if ( 'LR' in globals())==False:
        LR = 0.01                   
    if ( 'OUT' in globals())==False:
        OUT = 'mort'
    if ( 'LOSS' in globals())==False:
        ACT=''
        
if ( 'iSR' in globals())==False:
    iSR=1
    if OHE:
        iSR =5
        
skf = KFold(n_splits= FD)
conf.ohe = OHE     
conf.rnn_layers = NL
conf.epochs = EP
conf.batch_size = BS    
conf.pad_maxlen = MXL
conf.metric = ''
conf.arc = ARC 
conf.rnn_units = [NU] *NL
conf.dropout  = DO
conf.BN = BN
conf.LR = LR

if tk==1:
    TK = [1,2]
else:
    TK = [tk]
    
if VAL:            
    MON='val_'
else:
    MON=''    

# <----------------------------------------------------------- switch   --------------------------------------------------------    
if OUT=='surv':
    conf.task = args.task = 'surv'   
else:
    conf.task = args.task = 'mort'  
     


conf.dec_cat = ['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal']
conf.dec_cat.remove('ethnicity')
conf.dec_num = ['admissionheight', 'admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)','Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation', 'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH']

NCAT = len(conf.dec_cat)
print('\n\n\nThere are',NCAT,'categrical vars!\n\n\n')


Tr, Ts= {}, {}
for fold_id, (train_idx, test_idx) in enumerate(skf.split(all_idx)):
    Tr[fold_id] = train_idx
    Ts[fold_id] = test_idx    
    
    
train_idx = Tr[fd]
test_idx  = Ts[fd]
print('Running Fold {}...'.format(fd))

train_idx = all_idx[train_idx]                  
test_idx = all_idx[test_idx]

if ( 'train' in globals())==False:
    '''
    ['patientunitstayid', 'itemoffset', 'apacheadmissiondx', 'gender',
       'GCS Total', 'Eyes', 'Motor', 'Verbal', 'admissionheight',
       'admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)',
       'Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
       'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH',
       'unitdischargeoffset', 'hospitaldischargestatus', 'time2exit_days']
    '''
    train0, test0, col_norm = utils.normalize_data_mort_df( conf, df_data, train_idx, test_idx)   
    train= utils.df_to_list(train0)   # group by patient unit stay ID , i.e. time series per patient
    test = utils.df_to_list(test0)
    
    train, nrows_train = utils.pad(train, conf.pad_maxlen )
    test,  nrows_test  = utils.pad(test, conf.pad_maxlen )    
    print( 'ntrn', len(nrows_train), 'ntst', len(nrows_test),   )

    train = (train, nrows_train); test = (test, nrows_test)
    
    

    
    

if conf.task == 'surv': 
    
    halflife=365.*2    
    mn1=train0.time2exit_days.min()
    mx1=train0.time2exit_days.max()
    
    qq =np.nanquantile( train0.time2exit_days, .75 ) 
        
    breaks=np.arange(0., qq, qq/ NT )
    # breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) 
    n_intervals=len(breaks)-1
    
    checkpoint_filepath = final_modelpath = model_path + \
'%s_NT%d_%dFD%d_BS%d_VAL%d_MXL%d_OHE%d_%s_NL%d_NU%d_BN%d_DO%.1f_%s_LR%.4f' %(conf.task, NT, fd,FD, BS,VAL,MXL, OHE, ARC, NL, NU, BN, DO, LOSS, LR)
else:
    NT = n_intervals = 1
    checkpoint_filepath = final_modelpath = model_path + \
'%s_%dFD%d_BS%d_VAL%d_MXL%d_OHE%d_%s_NL%d_NU%d_BN%d_DO%.1f_%s_LR%.4f_%s' %(conf.task, fd,FD, BS,VAL,MXL, OHE, ARC, NL, NU, BN, DO, LOSS, LR, ACT)
    
    
N = train[0].shape[0]
trn_inds = np.arange( N,dtype=int)
conf.val_inds = val_inds = trn_inds[::VAL ]
conf.trn_inds =  trn_inds = np.setdiff1d( trn_inds, val_inds ).astype( int) 
    
if VAL:
    train_gen, train_steps, val_gen, val_steps, (X_test, Y_test), max_time_step, (X_train, Y_train), (X_val, Y_val), Ys = read_data_v2( conf, train, test, VAL )
else:
    train_gen, train_steps, (X_test, Y_test), max_time_step, (X_train, Y_train), Ys = read_data( conf, train, test, val=False )

xx = train_gen.__next__()

ntimesteps  = train[0].shape[1]        
input1_var2 = xx[0][0].shape[2]        


#============= Model construction ============

model = models.build_network_v2(conf, ntimesteps, input1_var2, NCAT, output_dim=n_intervals, ACT=ACT, activation='sigmoid')
optim = models.get_optimizer(lr=conf.lr) # Adam(lr=0.0001, beta_1 = 0.9)


"""
[1., 1., 0., 0., 0., 0., 0.] 
[0., 0., 1., 0., 0., 0., 0.]  # <-- missing these people

[1., 1., 1., 1., 1., 1., 0.  # living
0., 0., 0., 0., 0., 0., 1.] # event happened?
"""


def calc_approx_cindex( n_intervals ):
    def loss(y_true, y_pred):              
                
        true_event = y_true[:,n_intervals:].numpy()
        when =breaks[ np.argmax( true_event==1, axis=1) ]
        event_time = when
        event = (when>0).astype(bool) 
                
        censored_event = y_true[:,0:n_intervals].numpy()        
        when =breaks[ np.argmax( censored_event==0, axis=1) ]
        event_time += when
        
        # fill in 
        event_time[event_time==0] = breaks[-1]        
        
        if 0:
            print( '\n?y_true:\n', y_true )
            print( 'event:\n',event)
            print( 'Approx true event times:', event_time )
        
        #yp = np.cumprod( y_pred[:,0:np.nonzero(breaks>breaks[-1])[0][0]], axis=1)[:,-1]       
        
        yp = np.cumprod( y_pred, axis=1)[:,-1] # look at end
        measures =concordance_index_censored( event, event_time, yp  )
                
        return measures[0] 
    return loss   
    
class CustomCallback( tf.keras.callbacks.Callback ):   
    
    def __init__(self, ohe, SR, patience=0):
        super( CustomCallback, self).__init__()                
        self.y_st_n=mlsurv_utils.convert_to_structured( Ys[ 'trn' ][0], Ys[ 'trn'][1] )      
        self.y_st_v=mlsurv_utils.convert_to_structured( Ys[ 'val' ][0], Ys[ 'val'][1] )              
        
        self.y_st_tt=mlsurv_utils.convert_to_structured( Ys[ 'tst' ][0][::SR,], Ys[ 'tst'][1][::SR, ] )      
        
        self.Cindices=[]
        self.All=[]
        
        self.Cindices_tst=[]
        self.All_tst=[]
        
        self.ohe = ohe 
        self.SR = SR
        
    def on_epoch_end(self, epoch, logs={} ):
        
        if (epoch % 5)==0:
            SR = self.SR
            if self.ohe:           
                x_cat = X_val[:, :, :NCAT].astype(int)
                x_nc = X_val[:,:,NCAT:]            
                one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429), dtype=int)
                x_cat = (np.eye(conf.n_cat_class)[x_cat].sum(2) > 0).astype(int)            
                yp = self.model.predict([x_nc, x_cat])
            else:        
                yp = self.model.predict( [X_val[:,:,NCAT:],X_val[:,:,:NCAT]] )

            yp = -np.cumprod(yp,axis=1)[:,-1] 

            c1=concordance_index_ipcw( self.y_st_n, self.y_st_v, yp) 
            self.Cindices.append(c1[0])
            self.All.append(c1)

            if self.ohe:           
                x_cat = X_test[::SR, :, :NCAT].astype(int)
                x_nc = X_test[::SR,:,NCAT:]            
                one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429), dtype=int)
                x_cat = (np.eye(conf.n_cat_class)[x_cat].sum(2) > 0).astype(int)            
                yp = self.model.predict([x_nc, x_cat])
            else:        
                yp = self.model.predict( [X_test[::SR,:,NCAT:],X_test[::SR,:,:NCAT]] )        

            yp = -np.cumprod(yp,axis=1)[:,-1] 
            c2=concordance_index_ipcw( self.y_st_n, self.y_st_tt, yp)         
            self.Cindices_tst.append(c2[0])
            self.All_tst.append(c2)
            print( 'cindex=', c1[0], '-', c2[0], end=' ' )       

            

def surv_v2( y_true, y_pred):                   
    n_intervals= y_true.numpy().shape[1]//2    
        
    #print( 'hellos', output_stream=sys.stdout )
    uncens =  y_true[:,n_intervals:2*n_intervals].numpy() - y_pred.numpy() # inndividuals w/ events

    i = (np.sum( y_true[:,n_intervals:2*n_intervals] ,axis=1) == 0).astype(int)
        
    #print( uncens, output_stream=sys.stdout )
    return np.sum( np.abs(uncens), axis=-1)*i #return -log likelihood


def calc_tn( y_true, y_pred ):
    n_intervals= y_true.numpy().shape[1]//2
    
    neg_true = (np.sum( y_true[:,n_intervals:].numpy(), axis=1)==0 ).astype(int)             
    neg_pred = (np.sum( y_pred.numpy(), axis=1)==0 ).astype(int)             
    
    nneg = np.sum(neg_true)+1e-10       
    return np.sum( neg_true == neg_pred )/ nneg # 1- K.sum( true_neg * (np.sum(y_pred, axis=1)/n_intervals  ) )

def calc_tp( y_true, y_pred ):
    n_intervals= y_true.numpy().shape[1]//2
    A= y_true[:,n_intervals:].numpy()
    B= y_pred.numpy()
    pos_true = (np.sum( A, axis=1)==1 ).astype(int)                 
    ctns = pos_true * np.sum( A*B, axis=1 )                
    npos = np.sum(pos_true)+1e-10     
    return ctns/ npos # 1- K.sum( true_neg * (np.sum(y_pred, axis=1)/n_intervals  ) )

def calc_tp2( y_true, y_pred ):    
    n_intervals= y_true.numpy().shape[1]//2
    true_event = y_true[:,n_intervals:] #.numpy()                        
    return K.sum(true_event * y_pred) /( K.sum(true_event)+1e-10)                

def surv_likelihood(n_intervals):
    def loss(y_true, y_pred):           
        #a=y_true.numpy()  # (BS, n_intervals*2, 1) 
        #b=y_pred.numpy()  # (BS, MXL, n_intervals )
        #print( '???\n\n', a.shape, b.shape ) 
        #tf.print(y_true, output_stream=sys.stdout)
        #tf.print(y_pred, output_stream=sys.stdout)        
        
        cens_uncens = 1. + y_true[:,0:n_intervals] * (y_pred-1.) # individuals w/o events
        uncens = 1. - y_true[:,n_intervals:2*n_intervals] * y_pred # inndividuals w/ events
        
        return K.sum(-K.log(K.clip(K.concatenate((cens_uncens,uncens)),K.epsilon(),None)),axis=-1) #return -log likelihood
    return loss


def emd( n_intervals ):
    def loss(y_true, y_pred):        
        yt = y_true[:,n_intervals:]        # 
        yp = K.log( y_pred )>0       
        return 1 # tf.reduce_mean(tf.square(tf.cumsum( yt, axis=-1) - tf.cumsum( yp, axis=-1)), axis=-1)
    return loss

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def joint_loss():
    def loss( y_true, y_pred ):                       
        #print(y_true, y_true2, y_pred )  # int64, float32, float32             
        crossentropy_loss = bce(y_true, y_pred)
        
        y_true = tf.cast(y_true, tf.float32)
        sen = utils.sensitivity( y_true, y_pred )
        
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)) + K.epsilon() )      # before summing: length = BS 
        FP = K.sum(K.round(K.clip((1-y_true) * y_pred, 0, 1)) + K.epsilon() )

        TN = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)) + K.epsilon() ) 
        FN = K.sum(K.round(K.clip((y_true) * (1-y_pred), 0, 1)) + K.epsilon() )
         
        PPV = TP/(TP+FP)
        NPV = TN/(TN+FN)        
        #print( 'SEN=',sen,'PPV=', PPV, 'BCE=', crossentropy_loss  )        
        return  crossentropy_loss + PPV + NPV + sen
    return loss
    

EARLY='loss'
if conf.task == 'mort':
    #EARLY='f1'
    Metrics= [models.metrics.f1, utils.ppv, utils.npv, utils.sensitivity]    
    Metrics= [models.metrics.f1, utils.ppv, utils.npv, utils.sensitivity, tf.keras.metrics.BinaryCrossentropy(from_logits=False) ]
    
    if LOSS=='joint':
        model.compile(loss=joint_loss(), optimizer=optim, metrics=Metrics, run_eagerly=True )
    elif LOSS=='bce':    
        model.compile(loss="binary_crossentropy", optimizer=optim, metrics=Metrics,  run_eagerly=True)
elif conf.task == 'rlos':
    model.compile(loss='mean_squared_error', optimizer=optim, metrics=['mse'],  run_eagerly=True )

elif conf.task in ['phen', 'dec']:
    model.compile(loss="binary_crossentropy" ,optimizer=optim, metrics=[models.metrics.f1,'accuracy'], run_eagerly=True)
else:    
    model.compile(loss =[surv_likelihood(n_intervals)], optimizer = optim, metrics=[surv_v2, calc_tp, calc_tn], run_eagerly=True) 
       
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor=MON+EARLY,
    filepath=checkpoint_filepath + 'ckpt',        
    mode='auto', save_best_only=True, save_weights_only=True,)
earlystop=tf.keras.callbacks.EarlyStopping(
    monitor=MON+EARLY,
    min_delta=0,
    patience=7, verbose=0, mode='auto', baseline=None, restore_best_weights=True)    
print('\n\nWriting results to', final_modelpath, '\n\n'); print( '\n\n\nCustomCallback at iSR =', iSR ); eval_callback=CustomCallback( ohe=OHE, SR= iSR );CC=[eval_callback, earlystop, model_checkpoint_callback]

#============= Train / Eval model ============
just_fitted=0

for MODE in TK:    
    if (MODE==1) or (MODE==3):         
        if VAL:
            hist = model.fit(train_gen, validation_data= val_gen, initial_epoch=0, validation_steps=val_steps, callbacks=CC, steps_per_epoch=25, use_multiprocessing=False, epochs=conf.epochs)
        else:
             hist = model.fit_generator(train_gen, use_multiprocessing=False, callbacks=CC, steps_per_epoch=25, epochs=conf.epochs,verbose=1,shuffle=True)
                
        print('Model-fitting done; saving the final model wts...')         
        hist = hist.history        
        try:
            #model.save( final_modelpath + "_final.h5" )
            model.save_weights( final_modelpath + "_final_wts.h5"  )
            just_fitted =1
        except:
            pass         

    elif MODE==2:   
        del train
        del test 
        
        import matplotlib.pyplot as plt
        if ('hist' in globals())==False:
            hist = 0
        else:
            try:
                f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(20,8) )                 
                for k in hist.keys():                    
                    if 'loss' not in k:
                        if  'val' in k:
                            ax1.plot( hist[k], label=k )        
                        else:
                            ax1.plot( hist[k], ':', label=k )
                        
                ax1.legend( ncol=2 ) # bbox_to_anchor=(0, -1.05) ); 
                ax1.set_title('Training progress'); 
            except:
                pass
                if 0:
                    ax1.plot( hist['loss'], label='training set' )
                    ax1.plot( hist['val_loss'], label='validation set'  )        
                    ax1.legend( nocl=3 );                 
                    ax1.set_title('Loss'); 
                
            Cindices_val = eval_callback.Cindices
            Cindices_tst = eval_callback.Cindices_tst 
            ax2.plot( eval_callback.Cindices , label='validation set')
            ax2.plot( eval_callback.Cindices_tst, label='subset of TEST set' )
            ax2.legend(  ); 
            ax2.set_title('C-indices'); 
            #ax2.scatter(x, y)
            plt.suptitle('Metrics')
            plt.tight_layout()
            plt.savefig( final_modelpath + '_ep%d.png' %  len(hist[k]) )        

            if just_fitted==0: # latest = tf.train.latest_checkpoint( final_modelpath )
                
                try:
                    model.load_weights( final_modelpath + "_final_wts.h5")
                    print('Model wts loaded from final. Proceed with evaluation...')
                except: #model = tf.keras.models.load_model(  final_modelpath + '_final_wts.fd', compile=False ) 
                    checkpoint = tf.train.Checkpoint(model)
                    checkpoint.restore( checkpoint_filepath + 'ckpt' )
                    print('Model wts loaded from checkpoint. Proceed with evaluation...')

         
            
        cdauc={}
        tids = ['trn','val','tst']
        if ('Res' in globals() )==False:
            Res = {}
            for r in ['CD-AUC', 'uno', 'harrell', 'f1','sen', 'spec', 'ppv', 'npv','aucpr','tpr', 'auc', 'mcc', 'specat90',  'cvscore' ]:
                Res[r] = []
            mean_fpr_mort = np.linspace(0, 1, 100)
            i_mort = 0
        
        evalsets = [0,1,2]
        if OHE==0:
            SR = 1
        else:
            SR = 2
            evalsets = [1,2]
        
        def eval_( Res ):

            print( '\n\n\nEvaluating at SR =', SR )
        
            for se in evalsets:
                if se==0:
                    Y1=Y_train; XX = X_train
                elif se==1:
                    Y1=Y_val; XX =X_val             
                else:
                    Y1=Y_test; XX = X_test
                                        
                if conf.num and conf.cat:
                    if conf.ohe:
                        x_cat = XX[::SR, :, :NCAT].astype(int)
                        x_nc = XX[::SR,:,NCAT:]

                        one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429), dtype=int)                    
                        x_cat = (np.eye(conf.n_cat_class)[x_cat].sum(2) > 0).astype(int)                    
                        probas_mort = model.predict([x_nc, x_cat])      
                    else:
                        probas_mort = model.predict([XX[::SR,:,NCAT:], XX[::SR,:,:NCAT]])                                                                                         

                        
                        
                        
                        
                        
                if conf.task == 'surv':                             
                    y_st0 = mlsurv_utils.convert_to_structured( Ys[ tids[0] ][0][::SR], Ys[tids[0]][1] [::SR] )      
                    y_st = mlsurv_utils.convert_to_structured( Ys[ tids[se] ][0][::SR], Ys[tids[se]][1][::SR] )      
                    p = Ys[ tids[se] ][0][::SR]                    
                    #  [np.interp(2,breaks,np.concatenate(([1],np.cumprod(probas_mort[i,:]))))  for i in range(probas_mort.shape[0]) ]
                    
                    yp = np.cumprod( probas_mort,axis=1)[:,-1]                    
                    
                    # perfect: 
                    # concordance_index( Ys[ tids[se] ][0],Ys[ tids[se] ][0], Ys[ tids[se] ][1] )                     
                    c1=concordance_index( Ys[ tids[se] ][0][::SR], yp, Ys[ tids[se] ][1][::SR] ) 
                    Res['harrell'].append(c1)
                                          
                    try:
                        # perfect: 
                        # concordance_index_ipcw( y_st0, y_st, -Ys[ tids[se] ][0][::SR] ) 
                        c2=concordance_index_ipcw( y_st0, y_st, -yp )[0]
                        Res['uno'].append(c2)
                    except:
                        pass                                        
                    try:
                        tp1=breaks[ n_intervals//2 ]                        
                        tp2=breaks[ -2 ]
                        yp1 = np.cumprod( probas_mort[:,0:np.nonzero(breaks>tp1)[0][0]], axis=1)[:,-1]                    
                        yp2 = np.cumprod( probas_mort[:,0:np.nonzero(breaks>tp2)[0][0]], axis=1)[:,-1]            
                    except:
                        #Ys['trn'][0][Ys['trn'][0]>0].min()
                        tp1,tp2= np.cumsum(np.diff(breaks)/5*4)
                        yp1 = np.cumprod( probas_mort[:,0:np.nonzero(breaks>tp1)[0][0]], axis=1)[:,-1]                    
                        yp2 = np.cumprod( probas_mort[:,0:np.nonzero(breaks>tp2)[0][0]], axis=1)[:,-1]            
    
                    a=-np.vstack((yp1,yp2)).transpose()
                    o1= mlsurv_utils.cumulative_dynamic_auc_v2( y_st0, y_st, a, [tp1,tp2], tied_tol=1e-8, ci=.95, debug=False )
                    o2= mlsurv_utils.sen_spec(  y_st0, y_st, a, [tp1,tp2], tied_tol=1e-8, ci=.95, debug=False )

            
                    # perfect: 
                    # a=Ys[ tids[0] ][0][::SR]; mlsurv_utils.cumulative_dynamic_auc_v2( y_st0, y_st0, -np.vstack((a,a)).transpose(),[tp1,tp2])
            
                    Res['CD-AUC'].append( o1[0][-1] )
                    Res['sen'].append( o2['SEN'][-1,1] )
                    Res['spec'].append( o2['SPEC'][-1,1] )
                    Res['npv'].append( o2['NPV'][-1,1] )
                    Res['ppv'].append( o2['PPV'][-1,1] )
                else:
                    
                    if ('th' in globals() )==False:
                        th = probas_mort.max()/2
                    fpr_mort, tpr_mort, thresholds = roc_curve( Y1[::SR], probas_mort )            
                    
                    Res['f1'].append(  f1_score( Y1[::SR], probas_mort> th, average='macro' ) )
                    Res['tpr'].append( interp(mean_fpr_mort, fpr_mort, tpr_mort))

                    Res['tpr'][-1][0] = 0.0
                    roc_auc_mort = auc(fpr_mort, tpr_mort)
                    Res['auc'].append(roc_auc_mort)  
                    
                    if 0:
                        TN,FP,FN,TP = confusion_matrix( Y1[::SR], probas_mort.round()).ravel()
                        PPV = TP/(TP+FP)
                        NPV = TN/(TN+FN)                   
                        
                    NPV =utils.npv_np( Y1[::SR], probas_mort.round() )
                    PPV =utils.ppv_np( Y1[::SR], probas_mort.round() )
                    sen =utils.sensitivity_np( Y1[::SR], probas_mort.round() )
                    spec=utils.specificity_np( Y1[::SR], probas_mort.round() )

                    Res['spec'].append(spec)
                    Res['sen' ].append(sen)
                    Res['ppv' ].append(PPV)
                    Res['npv' ].append(NPV)

                    average_precision_mort = average_precision_score( Y1[::SR],probas_mort)
                    Res['aucpr'].append(average_precision_mort)
                    Res['mcc'].append(matthews_corrcoef( Y1[::SR], probas_mort.round()))
                    Res['specat90'].append(1- fpr_mort[ tpr_mort>=0.90][0])

                print('===============\nEvaluation set:',se)
                for rk in Res.keys():
                    try:
                        if rk != 'tpr':
                            print( rk.upper(), '%.3f'% Res[rk][-1], end=', ')
                    except:
                        pass                    
            return Res, cdauc
            
        Res, cdauc = eval_( Res )        
        import pickle
        pickle.dump( {'Res':Res, 'hist':hist, 'SR':SR, 'Cindices_val':Cindices_val, 'Cindices_tst':Cindices_tst }, open( final_modelpath+'_progress.pkl', "wb"))            
        
    print( '\n', final_modelpath )
    
def show_res():  
    from glob import glob        
    import pickle
    import myutils
    
    ResBat={}
    for g in sorted( glob(model_path + '/mort*.pkl')):

        d=myutils.readpkl( g )    

        f= os.path.basename(g).split('_progress')[0]

        str1=''; str2=''
        try:
            ResBat[ f[6:]  ]
        except:
            ResBat[ f[6:]  ]= ['']*int(f[8]) 
        try:
            sr=  d['SR'] 
        except:
            sr= np.NaN

            
        for k in ['f1','sen','spec','ppv', 'npv', 'auc']: #d['Res'].keys() :
            try:
                str1+=k.upper() +  '=%.3f | '%d['Res'][k][-1] 
            except:
                pass
        
        #print('\nTest:\t', end=' |')
        
        for k in ['f1','sen','spec','ppv', 'npv', 'auc']: #d['Res'].keys() :
            try:
                str2+= k.upper() + '=%.3f | '%d['Res'][k][-2]
            except:
                pass                        
        
        ResBat[ f[6:]  ][ int(f[5]) ] = '\n'+str1 + '\n' + str2
        
    for k in ResBat.keys():
        print( '\n================\n%s | SR=%s'%(k, sr ))
        
        for f in range( len( ResBat[k] ) ): 
            s=ResBat[k][f] 
            if len(s)>1:                
                print( f ,end='')
                print( s )
        
show_res()            
