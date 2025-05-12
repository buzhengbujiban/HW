import os
import socket
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.svm import SVR
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.metrics import mean_squared_error
from pytbl import tbl_write
from qr_utils import str_to_epoch
import sys
from qrfitter3.evaluator import rcor, cor, pcor, fr, FR, PFR, evaluate
from qrfitter3.evaluator import feval_cor_fr, decode_fr_from_cor_fr, decode_cor_from_cor_fr, cor, _customized_metric_cor_weight
from qrfitter3.evaluator import DateGrouper, Evaluator
from qrfitter3.utils import *
try:
    from deepforest import CascadeForestRegressor
except ModuleNotFoundError:
    pass

def write_y_hat(y_hat, ds, outfile):
    def tbl_write_ex(df, fn):
        if os.path.exists(fn):
            stat = os.stat(fn)
            if stat.st_size == 0:
                # remove empty file
                print("removing empty file {}".format(fn))
                os.remove(fn)
            else:
                # mkold
                import tempfile
                full_fn = os.path.realpath(fn)
                fd, new_fn = tempfile.mkstemp(prefix="%s."%full_fn)
                os.close(fd)
                print("rename {} to {}".format(fn, new_fn))
                os.rename(fn, new_fn)
        tbl_write(df, fn)
    print ('Saving y_hat file to: %s' % outfile)
    y_hat_df = pd.DataFrame({
        'date': ds['D'],
        'sid': ds['S'],
        'time': ds['I'],
        # 'epoch':list(map(lambda x, y: str_to_epoch(f'{x} {y}'), ds['D'], ds['I'])),
        'yhat': y_hat.astype(np.float32),
        'y': ds['Y'][:, ds['y_names'].index(ds['fit_y'])],
        'weight': ds['w']
    })
    #print(y_hat_df)
    #NOTICE: there are times NAS fails to save the files
    # tbl_write_ex(y_hat_df, outfile+".tbl") #20240705
    save_df_as_sdiv_daily(y_hat_df, outfile, fn='yhat', S='sid', D='date', I='time')
    return y_hat_df

# Interface for Regressor
class IRegressor:
    def __init__(self, config):
        self.config = config
        self.outdir = self.config["outdir"]
        self.SAVE_FILE_NAME = 'abstract'

    def _init_fot_fit(self):
        self.update_config()
        os.makedirs(self.outdir, exist_ok=True)
        save_json(self.config, os.path.join(self.outdir, "updated_fitter_cfg.json"))  # 将该次实验的config存档
        os.chmod(os.path.join(self.outdir, "updated_fitter_cfg.json"), 0o700)
        log_fn = os.path.join(self.outdir, "train_log.log")
        self.logger = get_logger(log_fn, name=log_fn)
        os.chmod(log_fn, 0o700)
        self.logger.info(self.config)

    def _init_for_inference(self):
        log_fn = os.path.join(self.outdir, "inference_log.log")
        self.logger = get_logger(log_fn, name=log_fn)
        os.chmod(log_fn, 0o700)
        self.logger.info(self.config)

    def update_config(self):
        self.config['hostname'] = socket.gethostname()
        self.config['pid'] = os.getpid()
        self.config['SAVE_FILE_NAME'] = self.SAVE_FILE_NAME

    def fit(self, ds_train, ds_valid):
        self.y_hat_train, df_fit_result = self.train(ds_train['X'],
                                                     ds_train['Y'][:, ds_train['y_names'].index(ds_train['fit_y'])].copy(),
                                                     ds_train['w'],
                                                     ds_train['x_names'],
                                                     (ds_valid['X'], ds_valid['Y'][:, ds_valid['y_names'].index(ds_valid['fit_y'])].copy(), ds_valid['w']))
        
        if df_fit_result is not None:
            print(df_fit_result)
            df_fit_result['epoch'] = df_fit_result.index.values
            tbl_write(df_fit_result, os.path.join(self.outdir, '{}_rounds.tbl'.format(self.SAVE_FILE_NAME)), 'zstd')

    def predict(self, X):
        pass

    def save_feature_names(self, dataset):
        feature_names = dataset['train']['x_names']
        save_json(feature_names, os.path.join(self.outdir, "feature_names.json"))
        os.chmod(os.path.join(self.outdir, "feature_names.json"), 0o700)
        with open(os.path.join(self.outdir, '{}.fmap'.format(self.SAVE_FILE_NAME)), 'w') as outfile:
            i = 0
            for feat in feature_names:
                outfile.write('{0}\t{1}\tq\n'.format(i, feat))
                i = i + 1

    def evaluate(self, datasets) -> dict:
        print(datasets['train']['y_names'])
        #col_labels = y_colnames_to_col_labels(datasets['train']['y_names'], pattern="(\\-?\\d+\\D+)$")
        col_labels = datasets['train']['y_names']
        print(col_labels)

        do_evaluate = self.config.get("do_evaluate", True)
        if not self.config['useExistModel']: 
            # eval for train 
            if datasets.get("eval", {}):
                y_hat_train = self.predict(datasets['eval']['X'])  
                ds_train = datasets['eval']
            else:
                y_hat_train = self.y_hat_train
                ds_train = datasets['train']
            y_hat_train_file = os.path.join(self.outdir, 'y_hat_train')
            y_hat_train_df = write_y_hat(y_hat_train, ds_train, y_hat_train_file)
            # eval for validation
            y_hat_valid = self.predict(datasets['valid']['X'])  
            ds_valid = datasets['valid']
            y_hat_valid_file = os.path.join(self.outdir, 'y_hat_valid')
            y_hat_valid_df = write_y_hat(y_hat_valid, ds_valid, y_hat_valid_file)
            
            if do_evaluate:
                groups_train = DateGrouper(ds_train['D'].astype(int)).by_month()
                eval_train = Evaluator(y_hat_train, ds_train, groups_train, col_labels=col_labels, print_pcor=True)
                df_train = eval_train.calculate_metrics()
                df_train['date'] = df_train.index.values
                tbl_write(df_train, os.path.join(self.outdir, "train_by_month.tbl"), 'zstd')
                df_yname2horizon = pd.DataFrame([ds_train['y_names'], col_labels], index=['yname','hor.name']).T
                df_yname2horizon['term'] = df_yname2horizon.index.values
                tbl_write(df_yname2horizon, os.path.join(self.outdir, "yname2horizon.tbl"))

                groups_valid = DateGrouper(ds_valid['D'].astype(int)).by_month()
                eval_valid = Evaluator(y_hat_valid, ds_valid, groups_valid, col_labels=col_labels, print_pcor=True)
                df_valid = eval_valid.calculate_metrics()
                df_valid['date'] = df_valid.index.values
                tbl_write(df_valid, os.path.join(self.outdir, "valid_by_month.tbl"), 'zstd')

        # eval for test
        if 'test' in datasets:
            ds_test = datasets['test']
            y_hat_test = self.predict(ds_test['X'])
            # write y_hat_test out
            y_hat_test_file = os.path.join(self.outdir, 'y_hat_test')
            y_hat_test_df = write_y_hat(y_hat_test, ds_test, y_hat_test_file)
            # by months
            if do_evaluate:
                groups_test = DateGrouper(ds_test['D'].astype(int)).by_month()
                eval_test = Evaluator(y_hat_test, ds_test, groups_test, col_labels, print_pcor=True)
                df_test = eval_test.calculate_metrics()
                df_test['date'] = df_test.index.values
                tbl_write(df_test, os.path.join(self.outdir, "test_by_month.tbl"), 'zstd')
                # by date
                groups_test = DateGrouper(ds_test['D'].astype(int)).by_date()
                eval_test = Evaluator(y_hat_test, ds_test, groups_test, col_labels, print_pcor=True)
                df_test = eval_test.calculate_metrics()
                df_test['date'] = df_test.index.values
                tbl_write(df_test, os.path.join(self.outdir, "test_by_date.tbl"), 'zstd')
        if do_evaluate:
            set_results = {}
            for set_name in datasets.keys():
                eval_metrics = evaluate(yhat_df=eval(f"y_hat_{set_name}_df"))
                self.logger.info(f"{set_name} set: {str(eval_metrics)}")
                set_results[set_name] = eval_metrics
            # check_valid_test_monotonicity(set_results, self.outdir)  # TODO:determine baseline for tree

    #TODO
    def save_yhat(self, dataset, set_name: str):
        pass

    def run(self, datasets):
        self._init_fot_fit()
        self._init_model()
        self.save_feature_names(datasets)
        if not self.config['useExistModel']:
            self.fit(datasets['train'], datasets['valid'])
        self.save_model(os.path.join(self.outdir, '{}.model'.format(self.SAVE_FILE_NAME)))
        self.evaluate(datasets)
        self.logger.info(f"Saving model.")

    def run_inference(self, datasets):
        self._init_model()
        for set_name, dataset in datasets.items():
            y_hat = self.predict(dataset["X"])
            y_hat_test_file = os.path.join(self.outdir, f'y_hat_{set_name}')
            y_hat_test_df = write_y_hat(y_hat, dataset, y_hat_test_file)


class RidgeRegressor(IRegressor):
    def __init__(self, config):
        super().__init__(config)
        self.SAVE_FILE_NAME = 'ridge'
        
    def _init_model(self):
        self.para_dict = self.config['para_dict'].copy()
        assert self.para_dict, "please specify hyperparameters in item: regressor of config file"
        print ('fitter_class: %s' % self.config['class'])
        print ('fitter para:', self.para_dict)
        model_path = os.path.join(self.outdir , '{}.model'.format(self.SAVE_FILE_NAME))
        print('model_path:', model_path)
        if self.config['useExistModel'] and os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def train(self, X, y, w, x_vars, testset=None):
        self.x_vars = x_vars
        extra = None
        self.model = Ridge(**self.para_dict)
        if testset is None:
            y_pred = self.model.fit(X, y, sample_weight=w)
        else:
            evals_result = {}
            X1, y1, w1 = testset
            
            self.model.fit(X, y, sample_weight=w)

            y_pred = self.model.predict(X)
            y1_pred = self.model.predict(X1)
            
            evals_result['train'] = {}
            evals_result['test'] = {}
            evals_result['train']['rmse'] = [mean_squared_error(y_pred, y, sample_weight=w)**0.5]
            evals_result['test']['rmse'] = [mean_squared_error(y1_pred, y1, sample_weight=w1)**0.5]
            evals_result['train']['cor'] = [cor(y, y_pred, w)]
            evals_result['test']['cor'] = [cor(y1, y1_pred, w1)]
              
            print ('evals_result', evals_result)
            extra = pd.DataFrame({
                'train_rmse': evals_result['train']['rmse'],
                'test_rmse': evals_result['test']['rmse'],
                'train_cor': evals_result['train']['cor'],
                'test_cor': evals_result['test']['cor']
            })
        y_hat = y_pred
        return y_hat, extra

    def save_model(self, out_model_file):
        #self.model.save_model(out_model_file)
        joblib.dump(self.model, out_model_file)

    def predict(self, X):
        assert X.shape[1] == len(self.x_vars), "X cols must equal to len(x_vars)!"
        y_hat = self.model.predict(X)
        return y_hat

class ENRegressor(IRegressor):
    def __init__(self, config):
        super().__init__(config)
        self.SAVE_FILE_NAME = 'elastic_net'
        
    def _init_model(self):
        self.para_dict = self.config['para_dict'].copy()
        assert self.para_dict, "please specify hyperparameters in item: regressor of config file"
        print ('fitter_class: %s' % self.config['class'])
        print ('fitter para:', self.para_dict)
        model_path = os.path.join(self.outdir , '{}.model'.format(self.SAVE_FILE_NAME))
        print('model_path:', model_path)
        if self.config['useExistModel'] and os.path.exists(model_path):
            self.model = joblib.load(model_path)
    
    def train(self, X, y, w, x_vars, testset=None):
        self.x_vars = x_vars
        extra = None
        if "ridge_alpha" in self.para_dict:
            if "alpha" in self.para_dict:
                sys.exit("can't have both ridge_alpha and alpha")
            print("WARNING: using ridge-style alpha")
            self.para_dict['alpha'] = self.para_dict['ridge_alpha'] / X.shape[0]
            del self.para_dict['ridge_alpha']
        self.model = ElasticNet(**self.para_dict)
        if testset is None:
            y_pred = self.model.fit(X, y, sample_weight=w)
        else:
            evals_result = {}
            X1, y1, w1 = testset
            
            self.model.fit(X, y, sample_weight=w)

            y_pred = self.model.predict(X)
            y1_pred = self.model.predict(X1)
            
            evals_result['train'] = {}
            evals_result['test'] = {}
            evals_result['train']['rmse'] = [mean_squared_error(y_pred, y, sample_weight=w)**0.5]
            evals_result['test']['rmse'] = [mean_squared_error(y1_pred, y1, sample_weight=w1)**0.5]
            evals_result['train']['cor'] = [cor(y, y_pred, w)]
            evals_result['test']['cor'] = [cor(y1, y1_pred, w1)]
              
            print ('evals_result', evals_result)
            extra = pd.DataFrame({
                'train_rmse': evals_result['train']['rmse'],
                'test_rmse': evals_result['test']['rmse'],
                'train_cor': evals_result['train']['cor'],
                'test_cor': evals_result['test']['cor']
            })
        y_hat = y_pred
        return y_hat, extra

    def save_model(self, out_model_file):
        #self.model.save_model(out_model_file)
        joblib.dump(self.model, out_model_file)

    def predict(self, X):
        assert X.shape[1] == len(self.x_vars), "X cols must equal to len(x_vars)!"
        y_hat = self.model.predict(X)
        return y_hat
        
class lgbRegressor(IRegressor):
    def __init__(self, config):
        super().__init__(config)
        self.SAVE_FILE_NAME = 'lgb'
        
    def _init_model(self):
        model_path = os.path.join(self.outdir , '{}.model'.format(self.SAVE_FILE_NAME))
        print('model_path:', model_path)
        if self.config['useExistModel'] and os.path.exists(model_path):
            self.model = lgb.Booster(model_file=model_path)
        else:
            self.para_dict = self.config['para_dict'].copy()
            if "ac_rounds" in self.para_dict:
                self.ac_rounds = self.para_dict["ac_rounds"]
                self.ac_iter_per_round = self.para_dict["ac_iter_per_round"]
                self.ac_alpha = self.para_dict["ac_alpha"]
                self.ac_roll_len = self.para_dict["ac_roll_len"]
                self.ac_roll_y = self.para_dict["ac_roll_y"]
                del self.para_dict["ac_rounds"]
                del self.para_dict["ac_iter_per_round"] 
                del self.para_dict["ac_alpha"]
                del self.para_dict["ac_roll_len"]
                del self.para_dict["ac_roll_y"]
            assert self.para_dict, "please specify hyperparameters in item: regressor of config file"
            print ('fitter_class: %s' % self.config['class'])
            print ('fitter para:', self.para_dict)
               
    def setAutoCorrTraining(self, dates, sids, model_loc):
        import os
        assert hasattr(self, "ac_rounds"), "please set ac_rounds in config file"

        self.model_loc = model_loc
        self.roll_df = pd.DataFrame({"date": dates, "sid": sids, "y_hat": np.zeros(len(dates))})
        os.makedirs(self.model_loc, exist_ok=True)

    def _rollYHat(self, roll_y, fill_y, extra_roll_mult = 0):
        self.roll_df["y_hat"] = roll_y
        yser = pd.DataFrame({"y_hat": fill_y})['y_hat']
        return self.roll_df.groupby(["sid", "date"])["y_hat"].shift(self.ac_roll_len * (1+extra_roll_mult)).fillna(yser).values

    def train(self, X, y, w, x_vars, validset=None):
        self.x_vars = x_vars
        extra = None
        print("lgb version", lgb.__version__, flush=True)
        print(f"train size: {y.shape}")
        if "categorical_feature" in self.para_dict:
            cat_fea = self.para_dict.pop("categorical_feature").split(",")
        else:
            cat_fea = []
        if "max_mcs_ratio" in self.para_dict:
            tot_items = X.shape[0]
            max_mcs = int(tot_items * self.para_dict["max_mcs_ratio"])
            print(f"max_mcs_ratio={self.para_dict['max_mcs_ratio']}, tot_items={tot_items}, max_mcs={max_mcs}", flush=True)
            if max_mcs < self.para_dict['min_child_samples']:
                print(f"override min_child_samples={self.para_dict['min_child_samples']} to max_mcs={max_mcs}", flush=True)
                self.para_dict['min_child_samples'] = max_mcs
            del self.para_dict['max_mcs_ratio']
        if not hasattr(self, "ac_rounds"):
            self.ac_rounds = 0
        extras = []
        for it in range(self.ac_rounds+1):
            # if it == 0, normal traning
            # if it != 0, auto-correlation training
            lgbparam = self.para_dict.copy()
            if it != 0:
                lgbparam['n_estimators'] = self.ac_iter_per_round
                y_hat = self.model.predict(X)
                if not self.ac_roll_y:
                    y_hat_roll = self._rollYHat(y_hat, y)
                    y_hat_roll = y_hat_roll * np.std(y) / np.std(y_hat_roll)
                else:
                    y_hat_roll = self._rollYHat(y, y, it - 1)

                y_train = y + self.ac_alpha * y_hat_roll
                extra_train_param = {
                    "init_model": f"{self.model_loc}/lgb_{it-1}.model",
                }
                print(f"DEBUG: std yhat = {np.std(y_hat)}, std y = {np.std(y)}, std yhat_roll = {np.std(y_hat_roll)}, ac_alpha = {self.ac_alpha}")
                print(f"preparing iteration {it}. corr(y, y_hat) = {np.corrcoef(y, y_hat)[0,1]} corr(y, y_hat_roll) ={np.corrcoef(y, y_hat_roll)[0,1]} corr(y, y_train) = {np.corrcoef(y, y_train)[0,1]}", flush=True)
            else:
                y_train = y
                extra_train_param = {}
            if validset is None:

                self.model = lgb.LGBMRegressor(**lgbparam)
                self.model.fit(X=X, 
                            y=y_train, 
                            sample_weight=w,
                            feature_name=x_vars,
                            categorical_feature=cat_fea,
                            **extra_train_param)
                if hasattr(self, "model_loc"):
                    self.model.booster_.save_model(f"{self.model_loc}/lgb_{it}.model")
            else:
                evals_result = {}
                X1, y1, w1 = validset
                #test_dmat = xgb.DMatrix(X1, label=y1, weight=w1, feature_names=x_vars)
                
                print(f"valid size: {y1.shape}")
                self.model = lgb.LGBMRegressor(**lgbparam)
                if lgb.__version__ >= '4.0.0':
                    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(10)]
                else:
                    callbacks = [lgb.early_stopping(stopping_rounds=50)]
                self.model.fit(X=X, 
                            y=y_train, 
                            sample_weight=w,
                            eval_sample_weight=[w, w1],
                            eval_set=[(X, y_train), (X1, y1)],
                            eval_names=['train', 'valid'],  # train set will be ignored in early_stopping
                            callbacks=callbacks,
                            feature_name=x_vars,
                            eval_metric=_customized_metric_cor_weight,
                            categorical_feature=cat_fea, 
                            **extra_train_param)  
                if hasattr(self, "model_loc"):
                    self.model.booster_.save_model(f"{self.model_loc}/lgb_{it}.model")
                evals_result = self.model.evals_result_
                # print ('evals_result', evals_result)
                self.logger.info(f"best iteration: {self.model.best_iteration_}")
                num_iter = len(evals_result["train"]["cor"])
                extras.append(pd.DataFrame({
                    'train_rmse': [np.nan] * num_iter,
                    'valid_rmse': [np.nan] * num_iter,
                    'train_cor': evals_result['train']['cor'],
                    'valid_cor': evals_result['valid']['cor']
                }))

        y_hat = self.model.predict(X, num_iteration=self.model.best_iteration_)
        ret_extra = None
        if len(extras)> 0:
            ret_extra = pd.concat(extras).reset_index()
        return y_hat, ret_extra

    def save_model(self, out_model_file):
        #self.model.save_model(out_model_file)
        self.model.booster_.save_model(out_model_file, num_iteration=self.model.best_iteration_)

    def predict(self, X):
        # assert X.shape[1] == len(self.x_vars), "X cols must equal to len(x_vars)!"
        if hasattr(self.model, "best_iteration_"):
            print(f"predicting using best_iteration_")
            y_hat = self.model.predict(X, num_iteration=self.model.best_iteration_)
        else:
            y_hat = self.model.predict(X)  # 加载保存的模型是没有这个的
        return y_hat

class DeepForestRegressor(IRegressor):
    def __init__(self, config):
        super().__init__(config)
        self.SAVE_FILE_NAME = 'df'
        np.int = np.int_
        np.bool = np.bool_
    
    def _init_model(self):
        model_path = os.path.join(self.outdir , '{}.model'.format(self.SAVE_FILE_NAME))
        print('model_path:', model_path)
        if self.config['useExistModel'] and os.path.exists(model_path):
            self.model = CascadeForestRegressor()
            self.model.load(dirname=model_path)
        else:
            self.para_dict = self.config['para_dict'].copy()

    def train(self, X, y, w, x_vars, validset=None):
        print(f"train size: {y.shape}")
        X1, y1, w1 = validset
        print(f"valid size: {y1.shape}")

        self.model = CascadeForestRegressor(random_state=42, **self.para_dict)
        self.model.fit(X, y, w)
        y_hat = self.model.predict(X)
        return y_hat, None
    
    def save_model(self, out_model_file):
        self.model.save(dirname=out_model_file)

    def predict(self, X):
        y_hat = self.model.predict(X)  # 加载保存的模型是没有这个的
        return y_hat

class xgbRegressor(IRegressor):
    def __init__(self, config):
        super().__init__(config)
        self.SAVE_FILE_NAME = 'xgb'
        
    def _init_model(self, load_existing=False):
        # num_boost_round is set outside xgb.train(params=...)
        self.para_dict = self.config['para_dict'].copy()
        self.num_boost_round = self.para_dict['num_boost_round']
        self.para_dict.pop('num_boost_round')
        assert self.para_dict, "please specify hyperparameters in item: regressor of config file"
        print ('fitter_class: %s' % self.config['class'])
        print ('fitter para:', self.para_dict)
        model_path = os.path.join(self.outdir , '{}.model'.format(self.SAVE_FILE_NAME))
        print('model_path:', model_path)
        if self.config['useExistModel'] and os.path.exists(model_path):
            self.model = xgb.Booster()
            self.model.load_model(model_path)

    def train(self, X, y, w, x_vars, testset=None):
        train_dmat = xgb.DMatrix(X, label=y, weight=w, feature_names=x_vars)
        self.x_vars = x_vars
        extra = None

        if testset is None:
            self.model = xgb.train(params=self.para_dict,
                                   dtrain=train_dmat,
                                   num_boost_round=self.num_boost_round)
        else:
            evals_result = {}
            X1, y1, w1 = testset
            test_dmat = xgb.DMatrix(X1, label=y1, weight=w1, feature_names=x_vars)
            eval_set = [(train_dmat, 'train'), (test_dmat, 'test')]
            self.model = xgb.train(params=self.para_dict,
                                   dtrain=train_dmat,
                                   num_boost_round=self.num_boost_round,
                                   verbose_eval=2,
                                   feval=feval_cor_fr,
                                   evals=eval_set,
                                   evals_result=evals_result)
            print ('evals_result', evals_result)

            if 'rmse' in evals_result['train']:
                extra = pd.DataFrame({
                    'train_fitloss': evals_result['train']['rmse'],
                    'test_fitloss': evals_result['test']['rmse'],
                    'train_cor': [decode_cor_from_cor_fr(cor_fr) for cor_fr in evals_result['train']['cor_fr']],
                    'test_cor': [decode_cor_from_cor_fr(cor_fr) for cor_fr in evals_result['test']['cor_fr']],
                    'train_fr': [decode_fr_from_cor_fr(cor_fr) for cor_fr in evals_result['train']['cor_fr']],
                    'test_fr': [decode_fr_from_cor_fr(cor_fr) for cor_fr in evals_result['test']['cor_fr']]
                })
            elif 'mphe' in evals_result['train']:
                extra = pd.DataFrame({
                    'train_fitloss': evals_result['train']['mphe'],
                    'test_fitloss': evals_result['test']['mphe'],
                    'train_cor': [decode_cor_from_cor_fr(cor_fr) for cor_fr in evals_result['train']['cor_fr']],
                    'test_cor': [decode_cor_from_cor_fr(cor_fr) for cor_fr in evals_result['test']['cor_fr']],
                    'train_fr': [decode_fr_from_cor_fr(cor_fr) for cor_fr in evals_result['train']['cor_fr']],
                    'test_fr': [decode_fr_from_cor_fr(cor_fr) for cor_fr in evals_result['test']['cor_fr']]
                })
            else:
                extra = pd.DataFrame({
                    'train_fitloss': np.nan,
                    'test_fitloss': np.nan,
                    'train_cor': [decode_cor_from_cor_fr(cor_fr) for cor_fr in evals_result['train']['cor_fr']],
                    'test_cor': [decode_cor_from_cor_fr(cor_fr) for cor_fr in evals_result['test']['cor_fr']],
                    'train_fr': [decode_fr_from_cor_fr(cor_fr) for cor_fr in evals_result['train']['cor_fr']],
                    'test_fr': [decode_fr_from_cor_fr(cor_fr) for cor_fr in evals_result['test']['cor_fr']]
                })

        y_hat = self.model.predict(train_dmat)
        return y_hat, extra

    def save_model(self, out_model_file):
        self.model.save_model(out_model_file)

    def predict(self, X):
        assert X.shape[1] == len(self.x_vars), "X cols must equal to len(x_vars)!"
        test_dmat = xgb.DMatrix(X, feature_names=self.x_vars)
        y_hat = self.model.predict(test_dmat)
        return y_hat


class SVMRegressor(IRegressor):
    def __init__(self, config):
        super().__init__(config)
        self.SAVE_FILE_NAME = 'svm'

    def _init_model(self):
        self.para_dict = self.config['para_dict'].copy()
        assert self.para_dict, "please specify hyperparameters in item: regressor of config file"
        print ('fitter_class: %s' % self.config['class'])
        print ('fitter para:', self.para_dict)
        model_path = os.path.join(self.outdir , '{}.model'.format(self.SAVE_FILE_NAME))
        print('model_path:', model_path)
        if self.config['useExistModel'] and os.path.exists(model_path):
            self.model = joblib.load(model_path)
    
    def train(self, X, y, w, x_vars, testset=None):
        self.x_vars = x_vars
        extra = None
        self.model = SVR(**self.para_dict)
        if testset is None:
            self.model.fit(X, y, sample_weight=w)
        else:
            evals_result = {}
            X1, y1, w1 = testset
            
            self.model.fit(X, y, sample_weight=w)

            y_pred = self.model.predict(X)
            y1_pred = self.model.predict(X1)
            
            evals_result['train'] = {}
            evals_result['test'] = {}
            evals_result['train']['rmse'] = [mean_squared_error(y_pred, y, sample_weight=w)**0.5]
            evals_result['test']['rmse'] = [mean_squared_error(y1_pred, y1, sample_weight=w1)**0.5]
            evals_result['train']['cor'] = [cor(y, y_pred, w)]
            evals_result['test']['cor'] = [cor(y1, y1_pred, w1)]
              
            print ('evals_result', evals_result)
            extra = pd.DataFrame({
                'train_rmse': evals_result['train']['rmse'],
                'test_rmse': evals_result['test']['rmse'],
                'train_cor': evals_result['train']['cor'],
                'test_cor': evals_result['test']['cor']
            })
        y_hat = y_pred
        return y_hat, extra

    def save_model(self, out_model_file):
        #self.model.save_model(out_model_file)
        joblib.dump(self.model, out_model_file)

    def predict(self, X):
        assert X.shape[1] == len(self.x_vars), "X cols must equal to len(x_vars)!"
        y_hat = self.model.predict(X)
        return y_hat
