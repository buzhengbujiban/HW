import numpy as np
import operators as op
from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import Ridge
import pickle



# For neural network
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, LeakyReLU, BatchNormalization, PReLU, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.constraints import maxnorm
from keras import regularizers, optimizers, losses
##### to limit number of cores used by TensorFlow ####
from keras import backend as K

#import tensorflow 
import tensorflow.compat.v1 as tensorflow
tensorflow.disable_v2_behavior()

import keras

#from keras.models import model_from_json

tt1=tensorflow.config.threading.get_inter_op_parallelism_threads()
tt2=tensorflow.config.threading.get_intra_op_parallelism_threads()
print('threads to use, before', tt1,tt2)
num_threads=1
print('num_threads',num_threads)
tensorflow.config.threading.set_inter_op_parallelism_threads(num_threads)
tensorflow.config.threading.set_intra_op_parallelism_threads(num_threads)
tt1=tensorflow.config.threading.get_inter_op_parallelism_threads()
tt2=tensorflow.config.threading.get_intra_op_parallelism_threads()
print('threads to use, after', tt1,tt2)

class ModelNN:
    def __init__(self, input_dict_NN):
        #NN model
        tensorflow.set_random_seed(0)
        np.random.seed(0)
        
        self.epochs=input_dict_NN['epochs']
        self.batch_size=input_dict_NN['batch_size']
        number_of_features = input_dict_NN['number_of_features']
        reg_lam = input_dict_NN['reg_lam']
        neurons1 = input_dict_NN['neurons1']
        neurons2 = input_dict_NN['neurons2']
        neurons3 = input_dict_NN['neurons3']
        LeakyReLU_slope = input_dict_NN['LeakyReLU_slope']
        dropout_rate1 = input_dict_NN['dropout_rate1']
        dropout_rate2 = input_dict_NN['dropout_rate2']
        dropout_rate3 = input_dict_NN['dropout_rate3']
        learning_rate = input_dict_NN['learning_rate']
        
        # Create 3 layers
        Inp = Input(shape=(number_of_features,))
        L = Dense(neurons1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(reg_lam))(Inp)
        L = Dropout(dropout_rate1)(L)
        L = layers.LeakyReLU(LeakyReLU_slope)(L)
        
        L = Dense(neurons2, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(reg_lam))(L)
        L = Dropout(dropout_rate2)(L)
        L = layers.LeakyReLU(LeakyReLU_slope)(L)
        
        L = Dense(neurons3, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(reg_lam))(L)
        L = Dropout(dropout_rate3)(L)
        L = layers.LeakyReLU(LeakyReLU_slope)(L)
        L = BatchNormalization()(L)
        
        Output_final_layer = Dense(1, use_bias=False)(L)
        
        adam1=tensorflow.keras.optimizers.RMSprop(learning_rate=learning_rate, decay=0.0001)#, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
        
        self.model = Model(inputs=Inp, outputs=[Output_final_layer])
        self.model.compile(loss='mean_squared_error', optimizer=adam1)

    def fit(self, X, y, **kwargs):
        tensorflow.set_random_seed(0)
        np.random.seed(0)
    
        print('training model')
        early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                       verbose=2, callbacks=[early_stop], **kwargs)

    def predict(self, X):
        tensorflow.set_random_seed(0)
        np.random.seed(0)
        y_pred = self.model.predict(X)
        y_pred = op.at_nan2zero(y_pred)
        #y_pred = self.normalize(y_pred)
        return y_pred
    
    def get_weights_architecture(self):
        tensorflow.set_random_seed(0)
        np.random.seed(0)
        print('getting network weights and architecture')
        weights=self.model.get_weights()
        architecture=self.model.to_json()
        return weights, architecture
    
    def set_weights_architecture(self,weights,architecture):
        tensorflow.set_random_seed(0)
        np.random.seed(0)
        print('setting network weights and architecture')
        self.model = tensorflow.keras.models.model_from_json(architecture)#TF2
        self.model.set_weights(weights)  
        return weights, architecture
    


class fit_function:
  
    def __init__(self):
        
        ### This will determine the shape of the postA in ffw mode.
        ### If = 10, postA will have 10 days, if == 1, postA will have the (1 + delay) days
        ### Remember the last column is stale in delay 1, thus last - 1 column should be used when construcing preA
        self.prod_alpha_postA_length = 3 ## Either 1 or 10

    
    def fit(self, data, filter_matrix):
        
        print('NEURAL NETWORK FIT')

        # 1-D integer numpy array of alpha ids, sorted in ascending order
        alpha_list = data['alpha_list']

        numalphas = alpha_list.size
        numdates = data['numdates']
        numstocks = data['numstocks']
        dates = data['dates']
        delay = data['delay']

        # 1-D numpy array of integer alpha birthdays, order matching alpha_list
        birthday_arr = data['birthday_arr']

        # 2-D boolean numpy matrix, True if an alpha is born on a given day, False otherwise
        # it has the shape of (numalphas, numdates)
        is_born_mat = data['is_born_mat']

        # 1-D boolen numpy array of numdates, True if it is a rebalance date, False otherwise
        rebalance_dates_mask = data['rebalance_dates_mask']

        # Function handle to load an alpha postA. It has single argument as alpha id.
        # The postA is properly shifted in D0 mode.
        # postA = load_alpha(alpha_list[1]) loads the postA of alpha with id alpha_list[1]
        load_alpha = data['load_alpha']

        # Function handle to load a simvar. It has single argument as variable name.
        # The variable is properly shifted in D0 mode.
        # ret1_excess = load_simvar('ret1_excess') loads the ret1_excess
        load_simvar = data['load_simvar']
        
        pp_param = data['function_params']['fit_params'] if 'function_params' in data else {}
        n_buckets = pp_param.get('n_buckets',2)
        num_sa = pp_param.get('num_sa',100)
        return_cap = pp_param.get('return_cap',0.10)
        target_return_days = pp_param.get('target_return_days',1)
        lookback_to_use = pp_param.get('lookback',1250) 
        #bucketing_key can take = {'tvr', 'liq2'}
        bucketing_key = pp_param.get('bucketing_key','tvr')

        print('xxx parameters set')
        
        ##### Create alpha weights matrix #####
        alpha_weight_matrix = np.full_like(filter_matrix, np.nan, dtype=np.float32)

        ### Assert that filter_matrix does not contain nan's or inf's
        assert(np.all(np.isfinite(filter_matrix)))

        ### Set fit startdate, do not fit before this date ###
        lookback = lookback_to_use
        fit_startdate = 20210101
        fit_start_idx = np.where(data['dates'] <= fit_startdate)[0][-1]
        alpha_first_index = fit_start_idx - lookback - 10
        assert(alpha_first_index > 0)

        ### Get the ids of alphas that are ever selected within the fit window
        ever_used_alpha_ids = np.where(np.any(filter_matrix[:, fit_start_idx:].astype(np.bool), axis=1))[0]
        print('Number of alphas to load is %d' % ever_used_alpha_ids.size)
    
        simres_map = {aid: data['load_simres'](aid, verbose=False) for aid in alpha_list[ever_used_alpha_ids]}
    
        ### load alpha features (all available features listed at bottom of file)
        feature_list = [
                'dailypnl_top3000top1200',
                'dailypnl_top1000top600',
                'dailypnl_top500top250',
    
                'dailytvr_top3000top1200',
                'dailytvr_top1000top600',
                'dailytvr_top500top250',
    
                'dailyliq2_top500top250',
                'dailyliq2_top1000top600',
                'dailyliq2_top3000top1200',
    
                'dailyneb_top500top250',
                'dailyneb_top1000top600',
                'dailyneb_top3000top1200',    
                ]
        features_map = dict()
        if data['region'] != 'GLOBAL':
            
            for aid in alpha_list[ever_used_alpha_ids]:
                temp = data['load_features'](aid)
                features_map[aid] = {f: temp[f] for f in feature_list}
    
        ### done loading alpha features

        ### ------------------------------------------------------ ###
        ### Load ever used alphas into a dictionary ###

        ################################################
        # ### This is the slow way of loading the alphas
        # alpha_dict = {}
        # for idx in ever_used_alpha_ids:
        #     alpha = load_alpha(alpha_list[idx], verbose=True)
        #     ## In order to reduce memory footprint, cut out part that will never be used
        #     alpha_dict[idx] = np.copy(alpha[:, alpha_first_index:], order='F')


        ################################################
        ### Faster way of loading the alphas
        eind = data['numdates'] - 1
        #alpha_dict = {}
        ### Initialize the model dictionary
        model_dict = {}
        #for idx in ever_used_alpha_ids:
            # We are loading the alpha in (sind, eind) where eind is inclusive
            # If eind is not provided or is None, it would default to (data['numdates'] - 1)
            # If sind is not provided or is None, it would default to 0
            #alpha_dict[idx] = load_alpha(alpha_list[idx], sind=alpha_first_index, eind=eind, verbose=True)
        # Note that alpha load time is linear with the load range, i.e. the shorter the range, the faster the load
        
        ### ------------------------------------------------------ ###

        for di, date in enumerate(data['dates']):
            if di <= lookback or date <= fit_startdate:
                continue
            
            selected_idx_di_init = np.where(filter_matrix[:, di - delay] == True)[0]
            selected_alphas=selected_idx_di_init.size  
            #print('di',di,date,selected_alphas)

            if rebalance_dates_mask[di] and selected_alphas>3:
                print('⏩⏩⏩⏩ Fitting on %d' % date)

                idx_start = di - delay - lookback
                idx_end = di - delay
                

                # array of selected alpha indices
                selected_idx_di = np.where(filter_matrix[:, di - delay] == True)[0]
                
                alpha_list_born=alpha_list[selected_idx_di]
    
                ### define extra inputs needed for the below functions
                input_dict={}
                input_dict['n_buckets']=n_buckets
                input_dict['di']=di
                input_dict['lookback']=lookback
                input_dict['features_map']=features_map
                input_dict['simres_map']=simres_map
                input_dict['idx_end']=idx_end
                input_dict['idx_start']=idx_start
                input_dict['return_cap_for_target']=return_cap
                input_dict['target_var']='ret1_excess'
                input_dict['target_ndays']=target_return_days
                
                ### [STEP 1] bucket the alphas
                bucket_out_dict=bucket_basic_tvr(data,alpha_list_born,input_dict, Bucket_key = bucketing_key)
                bucket_keys=list(bucket_out_dict.keys())
                print('bucket_keys',bucket_keys) 
                
        
                model={}
                
                ### [STEP 2] go through buckets and filter from each one
                for bucket_index in bucket_keys:


                    print('bucket number',bucket_index)
                    alpha_list_in_bucket=bucket_out_dict[bucket_index]
                    print('length alpha_list_in_bucket',len(alpha_list_in_bucket))
                    #print('bucket n',bucket_index,alpha_list_in_bucket)
    
                    # [STEP 2a] generate target for model
                    target_dict = generate_target(data,input_dict)
                    
                    # [STEP 2b] generate acube_reg
                    acube_reg = get_acube_reg(data,alpha_list_in_bucket,target_dict,input_dict)     
                    
                    # [STEP 2c] generate covariance matrix for alphas in bucket
                    # uncomment if your fit needs a covariance / correlation matrix of alphas
                    #alpha_cov=alpha_corr_pnl(data,acube_reg,input_dict)
                    alpha_cov=1
                    
                    # [STEP 3] generate super alphas
                    super_alpha_dict={}
                    super_alpha_dict['num_super_alphas']=num_sa ##
                    super_alpha_dict['fit_super_alphas']=1  ### 1 in fitting, 0 in constructpreA  
                    
                    super_alpha_dict['alpha_list_in_bucket']=alpha_list_in_bucket
                    
                    if super_alpha_dict['num_super_alphas']>0:
                        acube_reg, super_alpha_dict = create_super_alphas_tvr(acube_reg, super_alpha_dict, input_dict)
             
                    # [STEP 4] fit the model
                    fitted_model_dict = fit_model(data,acube_reg,target_dict,input_dict)
                    
                    fitted_model_dict['super_alpha_dict']=super_alpha_dict
                    fitted_model_dict['alpha_list_in_bucket']=alpha_list_in_bucket
                    ### fitted_model_dict should containt everything you will need in construct_preA
               
                    model[bucket_index]=fitted_model_dict
                    
                ### [STEP 5] fit second layer model to combine preA's of each bucket
                ret1_reg=target_dict['ret1_reg']
                acube_reg_final = np.empty((ret1_reg.size, input_dict['n_buckets']), dtype=np.float32, order='F')
                for i, bucket_index in enumerate(bucket_keys):
                    temp_fitted_model_dict=model[bucket_index]
                    
                    bucket_preA=temp_fitted_model_dict['y_pred']
                    acube_reg_final[:,i]=bucket_preA
                
                clf = Ridge(alpha=0.1)
                clf.fit(acube_reg_final, ret1_reg) 
                w = clf.coef_ # get the weights
                w = w.ravel()
                w = w*0+1.0 ### set preA weights to equalweight
                w = w/np.sum(abs(w)) # normalize the weights
                
                print('final bucket weights')
                print(w)
                
                for i, bucket_index in enumerate(bucket_keys):
                    
                    model[bucket_index]['weight_final']=w[i] ### 
                    print('bucket index',bucket_index,'weight_final',model[bucket_index]['weight_final'])
                    model[bucket_index]['y_pred']=0.0 ### clean out y_pred from model dict to save space in cache

                
                model_dict[date] = pickle.dumps(model)
                model_size_MB = sum([len(pickle.dumps(v)) for v in model_dict.items()]) / 1024 ** 2
                print('model_size_MB = ',model_size_MB)             
        


        ### Construct an alpha_weights matrix to be used to attribution.
        ### You can assign weights to alphas based on your model, or you can simply use 'filter_matrix'
        alpha_attribution_weights = np.copy(filter_matrix)
        
        alpha_attribution_weights=np.nan_to_num(alpha_attribution_weights*1.0)
        aNN=alpha_attribution_weights.shape[0]
        dNN=alpha_attribution_weights.shape[1]
        
        alpha_attribution_weights[:,0]=0
        
        for di in range(1,dNN):
            number_non_zero_weights=np.sum((abs(alpha_attribution_weights[:,di])>0))
            if number_non_zero_weights<20:
                #print('number_non_zero_weights=',number_non_zero_weights)
                alpha_attribution_weights[:,di]=0
        
        return model_dict, alpha_attribution_weights



    def construct_preA(self, data, model_dict, mode): ### preA matrix is stock positions raw signal would like to take
        ### current set up generates positions are taken proportional to predicted returns from neural network
        print('contruct_preA mode:')
        print(mode)
        
        ### Unpickle all the model objects
        for k in model_dict:
            model_dict[k] = pickle.loads(model_dict[k])

        alpha_list = data['alpha_list']
        numalphas = alpha_list.size
        numdates = data['numdates']
        numstocks = data['numstocks']
        delay = data['delay']
        region = data['region']

        ### Running in 'ffw', construct the last day preA of shape (numstocks) using the latest model object
        ### This array will be used to stitch preA as preA[:, -1 - delay] = preA_ld
        if mode == 'last':
            
            ## In Global region FFW mode, only the primary_region portion of the preA is stitched.
            if region == 'GLOBAL':
                s, e = data['si_map'][data['primary_region']]
                numstocks = e - s + 1            
            
            
            preA_ld = np.zeros(shape=numstocks, dtype=np.float32, order='F')

            #### In 'ffw' only the latest available model will be present in model_dict
            #### in order to avoid unnecessary i/o.
            ### There is a single model object, extract it. The model is simple linear weights.
            modelparams = (list(model_dict.values())[0])

            load_alpha = data['load_alpha']
            
            dict_of_bucket_models=modelparams

        
            #### loop through the buckets:
            for bucket_index,fitted_model_dict in dict_of_bucket_models.items():
                alpha_list_in_bucket=fitted_model_dict['alpha_list_in_bucket']
                super_alpha_dict=fitted_model_dict['super_alpha_dict']
                weight_final=fitted_model_dict['weight_final']

                NNweights=fitted_model_dict['NNweights']
                NNarchitecture=fitted_model_dict['NNarchitecture']
                input_dict_NN=fitted_model_dict['input_dict_NN']
                ### re-create NN model and set weights
                NN = ModelNN(input_dict_NN)
                NN.set_weights_architecture(NNweights,NNarchitecture)
                model_object = NN

                number_of_alphas_used=len(alpha_list_in_bucket)
                    
                acube_reg_1d = np.empty((numstocks, number_of_alphas_used), dtype=np.float32, order='F')
            
                for i, aid in enumerate(alpha_list_in_bucket):
                    alpha = load_alpha(aid, verbose=False)
                    alpha = alpha/1000
                    if region != 'GLOBAL':
                        alpha = op.cs_zscore(op.at_zero2nan(alpha))
                    
                    acube_reg_1d[:,i]=alpha[:,-1-delay]

                acube_reg_1d=op.at_nan2zero(acube_reg_1d)
                
                super_alpha_dict['fit_super_alphas']=0 ### do not re-fit super-alphas4
                ii={}
                if super_alpha_dict['num_super_alphas']>0:
                    acube_reg_1d, super_alpha_dict = create_super_alphas_tvr(acube_reg_1d, super_alpha_dict,ii)
                
#                 predicted_returns = np.mean(acube_reg_1d,axis=1) 
                predicted_returns = model_object.predict(acube_reg_1d)  
                predicted_returns = op.at_nan2zero(predicted_returns.ravel())
                
                ### normalize the predictions
                if region != 'GLOBAL':
                    predicted_returns = predicted_returns/(np.nansum(abs(predicted_returns))+0.001)
                
                preA_ld=preA_ld+weight_final*op.at_nan2zero(predicted_returns)

            if region != 'GLOBAL':
                preA_ld=preA_ld*data['valids'][:,-1-delay] 
            else:
                preA_ld=preA_ld*data['valids'][s:e+1,-1-delay] 
                
            return preA_ld

        ### When running in 'refit' or 'full', mode argument to construct_preA is 'full'.
        ### Thus construct the full preA of shape (numstocks, numdates) using the all historical model objects
        if mode == 'full':
            ## make empty preA matrix
            ## make dictionary of ever used alphas
            ## go through re-fit days, re-create the network on each day 
            ## fill out the preA from today to next re-fit day by loading the alphas 
            load_alpha = data['load_alpha']
            refit_indices = sorted(model_dict.keys())
            refit_indices = np.array(refit_indices)
            print('refit_indices  = ', refit_indices)
            print('refit_indices[:-1]  = ', refit_indices[:-1])
            
            preA = np.zeros(shape=(numstocks, numdates), dtype=np.float32, order='F')
            
            first_refit_di=np.where(data['dates']==refit_indices[0])[0][0]
            print('first_refit_di day = ', first_refit_di)
            
            dict_of_bucket_models={}
            dict_of_recreated_models={}
            
            print('testing new version')
            for di in range(first_refit_di,numdates):
                
                date=data['dates'][di]
                is_refit_day=np.sum(refit_indices==date)
                
                if is_refit_day>0:
                    print('refit day = ', di,date)
                    dict_of_bucket_models=model_dict[date]
                    
                    #### loop through the buckets:
                    for bucket_index,fitted_model_dict in dict_of_bucket_models.items():
                        NNweights=fitted_model_dict['NNweights']
                        NNarchitecture=fitted_model_dict['NNarchitecture']
                        input_dict_NN=fitted_model_dict['input_dict_NN']
                    
                        ### re-create NN model and set weights
                        NN = ModelNN(input_dict_NN)
                        NN.set_weights_architecture(NNweights,NNarchitecture)
                    
                        dict_of_recreated_models[bucket_index]=NN
                    
                    
                #### loop through the buckets:
                for bucket_index,fitted_model_dict in dict_of_bucket_models.items():
                    #print('fitted_model_dict',fitted_model_dict)
                        
                    alpha_list_in_bucket=fitted_model_dict['alpha_list_in_bucket']
                    super_alpha_dict=fitted_model_dict['super_alpha_dict']
                    weight_final=fitted_model_dict['weight_final']

                    model_object=dict_of_recreated_models[bucket_index]
                
                    number_of_alphas_used=len(alpha_list_in_bucket)
                    acube_reg_1d = np.empty((numstocks, number_of_alphas_used), dtype=np.float32, order='F')
                
                    for i, aid in enumerate(alpha_list_in_bucket):
                        alpha = load_alpha(aid, sind=di-2, eind=di, verbose=False)
                        alpha = alpha/1000
                        
                        if region != 'GLOBAL':
                            alpha = op.cs_zscore(op.at_zero2nan(alpha))
                        
                        acube_reg_1d[:,i]=alpha[:,-1-delay]
                    
                    acube_reg_1d=op.at_nan2zero(acube_reg_1d)
                    
                    super_alpha_dict['fit_super_alphas']=0 ### do not re-fit super-alphas
                    ii={}
                    if super_alpha_dict['num_super_alphas']>0:
                        acube_reg_1d, super_alpha_dict = create_super_alphas_tvr(acube_reg_1d, super_alpha_dict,ii)
    
#                     predicted_returns = np.mean(acube_reg_1d,axis=1)             
                    predicted_returns = model_object.predict(acube_reg_1d)  
                    predicted_returns = op.at_nan2zero(predicted_returns.ravel())
                    
                    ### normalize the predictions
                    if region != 'GLOBAL':
                        predicted_returns = predicted_returns/(np.nansum(abs(predicted_returns))+0.001)
                    
                    preA[:,di-delay]=preA[:,di-delay]+weight_final*op.at_nan2zero(predicted_returns)
                    
            if region != 'GLOBAL':
                preA = op.at_nan2zero(preA*data['valids'])
            else:
                preA = op.at_nan2zero(preA*data['valids'])
            return preA
    
        else:
            raise Exception('Unrecognized mode: %s' % mode)
            
            
def bucket_basic_tvr(data,alpha_list,input_dict, Bucket_key):
    '''
    function to split alphas into N buckets based on a given metric
    N = 1 will not split alphas, all in one bucket
    '''     
    print('running bucket function')
    numbuckets=input_dict['n_buckets']
    features_map=input_dict['features_map']
    simres_map=input_dict['simres_map']
    di, lookback = input_dict['di'], input_dict['lookback']
    idx_start, idx_end = (di - data['delay'] - lookback), (di - data['delay'] - 1)
    
    if Bucket_key == 'liq2': 
        if data['region'] != 'GLOBAL':
            mask_ = 'dailyliq2_top3000top1200'
            bucket_metric = op.at_nan2zero(np.array([np.nanmean(features_map[aid][mask_][idx_start:idx_end]) for aid in alpha_list]))
        else:
            bucket_metric = np.array([np.nanmean(simres_map[aid]['liqsize'][idx_start:idx_end]) for aid in alpha_list])
    else:
        # Bucket_key == 'tvr'
        if data['region'] != 'GLOBAL':
            mask_ = 'dailytvr_top3000top1200'
            bucket_metric = op.at_nan2zero(np.array([np.nanmean(features_map[aid][mask_][idx_start:idx_end]) for aid in alpha_list]))
        else:
            bucket_metric = np.array([np.nanmean(simres_map[aid]['dailytvr'][idx_start:idx_end]) for aid in alpha_list])
        
    buckets = get_bins(bucket_metric, numbins=numbuckets)
    print('buckets',np.unique(buckets))

    bucket_out_dict={}
    for i in np.unique(buckets):
        bucket_out_dict[i]=alpha_list[buckets==i]

    return bucket_out_dict

def get_bins(inp, numbins=5):
    
    print('bucketing into ',numbins,' equal sized buckets')
    
    bins = np.linspace(1+1/numbins,2,numbins)
    #bins = np.arange(1.+1/numbins, 2.0, 1/numbins)
    
    bins[-1]=2.01 # to fix corner case of 1.999999999 upper limit creating extra bucket
    print('bins',list(bins))
    inp_rank = op.cs_rank(inp)
    inp_rank=op.at_nan2zero(inp_rank)

    out = np.digitize(inp_rank, bins)

    return out


def alpha_corr_pnl(data,alpha_list,input_dict):
    '''
    function to generate correlation matrix based on alpha pnls
    '''
    
    print('running alpha_cov function')

    features_map=input_dict['features_map']
    lookback=input_dict['lookback']
    di=input_dict['di']
    idx_start = di - data['delay'] - lookback
    idx_end = di - data['delay'] - 1

    #1. make correlation matrix and alpha ir list
    pnl_arr = []
    for i, aid in enumerate(alpha_list):
        sim = features_map[aid]

        if data['universe'] == 'top500' or data['universe'] == 'top250' or data['universe'] == 'top250_nonordic':
            dpnl = sim['dailypnl_top500top250'][idx_start:idx_end]

        elif data['universe'] == 'top1000' or data['universe'] == 'top600' or data['universe'] == 'top600_nonordic':
            dpnl = sim['dailypnl_top1000top600'][idx_start:idx_end]

        else:
            dpnl = sim['dailypnl_top3000top1200'][idx_start:idx_end]


        pnl_arr.append(dpnl)

    corrmat = np.corrcoef(pnl_arr)
    print('shape corrmat',corrmat.shape)

    return corrmat

def generate_target(data,input_dict):
    '''
    function to generate target for fitting model
    usually forward biased ret1 or variants of the same
    '''
    idx_start=input_dict['idx_start']
    idx_end=input_dict['idx_end']
    target_var=input_dict['target_var']
    return_cap_for_target=input_dict['return_cap_for_target']
    target_ndays=input_dict['target_ndays']
    
    print('target_var',target_var)
    
    ### Load ret1 excess and delay ####
    target = data['load_simvar'](target_var, verbose=True)
    target = (target)*100
    # target = op.cs_zscore(target)
    target = op.ts_mean(op.at_zero2nan(target*data['valids']),target_ndays)
    target = op.ts_delay(target, -target_ndays - data['delay'])
    
    target_cut = np.copy(target[:, idx_start: idx_end])
    target_cut[:, -1 - data['delay']:] = np.nan
    target_di = (target_cut).reshape(-1, order='F')
    
    ret1_valids = np.isfinite(target_di)
    ret1_reg = target_di[ret1_valids]

    ret1_reg[ret1_reg>return_cap_for_target]=return_cap_for_target
    ret1_reg[ret1_reg<-return_cap_for_target]=-return_cap_for_target


    target_dict={}
    target_dict['ret1_reg']=op.at_nan2zero(ret1_reg)
    target_dict['ret1_valids']=ret1_valids
    
    
    return target_dict

def get_acube_reg(data,alpha_list_in_bucket,target_dict,input_dict):
    '''
    function to generate acube_reg matrix (n_data x n_features)
    will load alphas and apply any transformations necessary
    valids to be used are copied from the target valids 
    '''

    
    ret1_reg=target_dict['ret1_reg']
    ret1_valids=target_dict['ret1_valids']
    idx_start=input_dict['idx_start']
    idx_end=input_dict['idx_end']    
    
    num_alphas=len(alpha_list_in_bucket)
    
    print('num_alphas',num_alphas)
    
    acube_reg = np.empty((ret1_reg.size, num_alphas), dtype=np.float32, order='F')
    
    
    for i, aid in enumerate(alpha_list_in_bucket):
        alpha = data['load_alpha'](aid, sind=idx_start, eind=idx_end-1, verbose=False)
        alpha = alpha/1000
        if data['region'] != 'GLOBAL':
            alpha = op.cs_zscore(op.at_zero2nan(alpha))
        
        acube_reg[:, i] = op.at_nan2zero(alpha.reshape(-1, order='F')[ret1_valids])
        
    acube_reg=op.at_nan2zero(acube_reg)
    print('acube_reg shape',acube_reg.shape)
    
    return acube_reg


def fit_model(data,acube_reg,target_dict,input_dict):
    '''
    function to fit a model that will combine features (raw alphas or super-alphas) 
    output a dictionary with model weights and all parameters needed
    to recreate the model in constructPreA and make predictions
    For easiest use, use models or create model class with model.fit and model.peredict methods
    '''

    fitted_model_dict={}
    ret1_reg=target_dict['ret1_reg']
    
    #Set parameters for Neural network model
    input_dict_NN = {}
    input_dict_NN['epochs'] = 100
    input_dict_NN['batch_size'] = int(acube_reg.shape[0]/256)
    input_dict_NN['number_of_features'] = acube_reg.shape[1]
    input_dict_NN['reg_lam'] = 0.0
    input_dict_NN['neurons1'] = 32
    input_dict_NN['neurons2'] = 32
    input_dict_NN['neurons3'] = 32
    input_dict_NN['LeakyReLU_slope'] = 0.1
    input_dict_NN['dropout_rate1'] = 0.2
    input_dict_NN['dropout_rate2'] = 0.2
    input_dict_NN['dropout_rate3'] = 0.2
    input_dict_NN['learning_rate'] = 0.001
    
    # NN model
    clf=ModelNN(input_dict_NN)

    clf.fit(acube_reg, ret1_reg) 
    
    NNweights, NNarchitecture = clf.get_weights_architecture()
    
    y_pred=clf.predict(acube_reg)
    y_pred=op.at_nan2zero(y_pred).ravel()
    
    print(f"🔊🔊🔊🔊 Correlation between predictions and actuals = {np.corrcoef(y_pred, ret1_reg)}")

    ### fitted_model_dict should containt everything you will need in construct_preA
    fitted_model_dict['y_pred']=y_pred

    fitted_model_dict['NNweights']=NNweights
    fitted_model_dict['NNarchitecture']=NNarchitecture
    fitted_model_dict['input_dict_NN']=input_dict_NN
    
    return fitted_model_dict

def create_super_alphas_tvr(acube_reg, super_alpha_dict,input_dict):
    '''
    function to combine raw alphas into super-alphas, reducing the number of features
    each super alpha is a weighted sum of subset of raw alphas
    weights are saved during fitting and used in constructpreA
    '''
    
    num_super_alphas=super_alpha_dict['num_super_alphas']
    fit_super_alphas=super_alpha_dict['fit_super_alphas']

    acube_reg=op.at_nan2zero(acube_reg)
    num_raw_alphas=acube_reg.shape[1]
    
    if num_super_alphas>(num_raw_alphas/2):
        num_super_alphas=int(np.floor(0.5*num_raw_alphas))
        if fit_super_alphas==1:
            print('number of super-alphas greater than 50% of raw alphas')
            print('set num_super_alphas=0.5*num_raw_alphas')
            print('num_super_alphas',num_super_alphas)
            print('num_raw_alphas',num_raw_alphas)
    
    n_data=acube_reg.shape[0]
    n=num_super_alphas
    
    acube_reg_new = np.empty((n_data, n), dtype=np.float32, order='F')
    

    x=np.linspace(0,num_raw_alphas,n+1)
    x=np.floor(x)
    
    if fit_super_alphas==1: ### generate weights 
        list_of_sa_weights=[]
        features_map=input_dict['features_map']
        idx_start=input_dict['idx_start']
        idx_end=input_dict['idx_end']
        alpha_list_in_bucket=super_alpha_dict['alpha_list_in_bucket']
        
        sort_metric='dailytvr_top3000top1200'
    
        metric_array=acube_reg[0,:]
    
        for i, aid in enumerate(alpha_list_in_bucket):
            
            metric_temp=np.nanmean(features_map[aid][sort_metric][idx_start:idx_end])
            metric_array[i]=metric_temp
        
    
        # print('metric_array',metric_array)
    
        idx_sort=np.argsort(metric_array) 
        # print('metric_array_sorted',metric_array[idx_sort])
        super_alpha_dict['idx_sort']=idx_sort
        
               
        
    if fit_super_alphas==0:
        list_of_sa_weights=super_alpha_dict['list_of_sa_weights']
        idx_sort=super_alpha_dict['idx_sort']
    
    
    acube_reg=acube_reg[:,idx_sort] ### line to sort raw alphas by sort_metric above before combining into super-alphas
    
    for i in range(n):
        
        sind=int(x[i])
        eind=int(x[i+1])
        #print('i',i,'sind',sind,'eind',eind)
        
        if fit_super_alphas==1: ### generate weights 
            temp_w = np.zeros((num_raw_alphas,1), dtype=np.float32, order='F')
            
            temp_w[sind:eind]=1.0
            temp_w=temp_w/np.sum(temp_w)
            
            list_of_sa_weights.append(temp_w)
            
        if fit_super_alphas==0: ### use generated weights (in constructPreA)
            temp_w=list_of_sa_weights[i]
    
        superalpha_temp=np.matmul(acube_reg,temp_w)
        acube_reg_new[:,i]=superalpha_temp.ravel()
        
    
    acube_reg_new=op.at_nan2zero(acube_reg_new)
    super_alpha_dict['list_of_sa_weights']=list_of_sa_weights
    
    return acube_reg_new, super_alpha_dict


def create_super_alphas(acube_reg, super_alpha_dict):
    '''
    function to combine raw alphas into super-alphas, reducing the number of features
    each super alpha is a weighted sum of subset of raw alphas
    weights are saved during fitting and used in constructpreA
    '''
    
    num_super_alphas=super_alpha_dict['num_super_alphas']
    fit_super_alphas=super_alpha_dict['fit_super_alphas']
    
    acube_reg=op.at_nan2zero(acube_reg)
    num_raw_alphas=acube_reg.shape[1]
    n_data=acube_reg.shape[0]
    n=num_super_alphas
    
    acube_reg_new = np.empty((n_data, n), dtype=np.float32, order='F')
    

    x=np.linspace(0,num_raw_alphas,n+1)
    x=np.floor(x)
    
    if fit_super_alphas==1: ### generate weights 
        list_of_sa_weights=[]
    if fit_super_alphas==0:
        list_of_sa_weights=super_alpha_dict['list_of_sa_weights']
    
    
    for i in range(n):
        
        sind=int(x[i])
        eind=int(x[i+1])
        #print('i',i,'sind',sind,'eind',eind)
        
        if fit_super_alphas==1: ### generate weights 
            temp_w = np.zeros((num_raw_alphas,1), dtype=np.float32, order='F')
            
            temp_w[sind:eind]=1.0
            temp_w=temp_w/np.sum(temp_w)
            
            list_of_sa_weights.append(temp_w)
            
        if fit_super_alphas==0: ### use generated weights (in constructPreA)
            temp_w=list_of_sa_weights[i]
    
        superalpha_temp=np.matmul(acube_reg,temp_w)
        acube_reg_new[:,i]=superalpha_temp.ravel()
        
    
    acube_reg_new=op.at_nan2zero(acube_reg_new)
    super_alpha_dict['list_of_sa_weights']=list_of_sa_weights
    
    return acube_reg_new, super_alpha_dict