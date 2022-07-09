# LOAD LIBRARIES
import pandas as pd
import numpy as np
import cupy
import cudf
import matplotlib.pyplot as plt
import gc
import os
from sklearn.model_selection import KFold
import xgboost as xgb

print('XGB Version', xgb.__version__)
print('RAPIDS version', cudf.__version__)

# 5.17.15-76051715-generic

# version name for saved model files
VER = 1

# train random seed
SEED = 42

# fill nan value
NAN_VALUE = -127  # will fit in int8

# folds per model
FOLDS = 5

# data directory
data = 'data'


def read_file(path='', usecols=None):
    # load dataframe
    if usecols is not None:
        df = cudf.read_parquet(path, columns=usecols)
    else:
        df = cudf.read_parquet(path)
    # reduce dtype for customer and date
    df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    df.S_2 = cudf.to_datetime(df.S_2)
    # SORT BY CUSTOMER AND DATE (so agg('last') works correctly)
    # df = df.sort_values(['customer_ID','S_2'])
    # df = df.reset_index(drop=True)
    # FILL NAN
    df = df.fillna(NAN_VALUE)
    print('shape of data:', df.shape)
    return df


print('Reading train data...')
TRAIN_PATH = os.path.join(data, 'train.parquet')
train = read_file(path=TRAIN_PATH)

print(train.head())


def process_and_feature_engineer(df):
    # FEATURE ENGINEERING FROM
    # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID', 'S_2']]
    cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
    num_features = [col for col in all_cols if col not in cat_features]

    test_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

    test_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

    df = cudf.concat([test_num_agg, test_cat_agg], axis=1)
    del test_num_agg, test_cat_agg
    print('shape after engineering', df.shape)

    return df


train = process_and_feature_engineer(train)

# add targets
targets = cudf.read_csv(os.path.join(data, 'train_labels.csv'))
targets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
targets = targets.set_index('customer_ID')
train = train.merge(targets, left_index=True, right_index=True, how='left')
train.target = train.target.astype('int8')
del targets

# needed to make cv deterministic (cudf merge above randomly shuffles rows)
train = train.sort_index().reset_index()

# features
FEATURES = train.columns[1:-1]
print(f'There are {len(FEATURES)} features!')

# xgb model parameters
xgb_parms = {
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'random_state': SEED
}


# needed with DeviceQuantileDMatrix below
class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256 * 1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0  # set iterator to 0
        self.batch_size = batch_size
        self.batches = int(np.ceil(len(df) / self.batch_size))
        super().__init__()

    def reset(self):
        """Reset the iterator"""
        self.it = 0

    def next(self, input_data):
        """Yield next batch of data."""
        if self.it == self.batches:
            return 0  # Return 0 when there's no more batch.

        a = self.it * self.batch_size
        b = min((self.it + 1) * self.batch_size, len(self.df))
        dt = cudf.DataFrame(self.df.iloc[a:b])
        input_data(data=dt[self.features], label=dt[self.target])  # , weight=dt['weight'])
        self.it += 1
        return 1


# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_metric_mod(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

    gini = [0, 0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


importances = []
oof = []
train = train.to_pandas()  # free GPU memory
TRAIN_SUBSAMPLE = 1.0
gc.collect()

skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, valid_idx) in enumerate(skf.split(
        train, train.target)):

    # train with subsample of train fold data
    if TRAIN_SUBSAMPLE < 1.0:
        np.random.seed(SEED)
        train_idx = np.random.choice(train_idx,
                                     int(len(train_idx) * TRAIN_SUBSAMPLE), replace=False)
        np.random.seed(None)

    print('#' * 25)
    print('### Fold', fold + 1)
    print('### Train size', len(train_idx), 'Valid size', len(valid_idx))
    print(f'### Training with {int(TRAIN_SUBSAMPLE * 100)}% fold data...')
    print('#' * 25)

    # train, valid, test for fold k
    Xy_train = IterLoadForDMatrix(train.loc[train_idx], FEATURES, 'target')
    X_valid = train.loc[valid_idx, FEATURES]
    y_valid = train.loc[valid_idx, 'target']

    dtrain = xgb.DeviceQuantileDMatrix(Xy_train, max_bin=256)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid)

    # train model fold k
    model = xgb.train(xgb_parms,
                      dtrain=dtrain,
                      evals=[(dtrain, 'train'), (dvalid, 'valid')],
                      num_boost_round=9999,
                      early_stopping_rounds=100,
                      verbose_eval=100)
    model.save_model(f'XGB_v{VER}_fold{fold}.xgb')

    # get feature importance for fold k
    dd = model.get_score(importance_type='weight')
    df = pd.DataFrame({'feature': dd.keys(), f'importance_{fold}': dd.values()})
    importances.append(df)

    # infer oof fold k
    oof_preds = model.predict(dvalid)
    acc = amex_metric_mod(y_valid.values, oof_preds)
    print('Kaggle Metric =', acc, '\n')

    # save oof
    df = train.loc[valid_idx, ['customer_ID', 'target']].copy()
    df['oof_pred'] = oof_preds
    oof.append(df)

    del dtrain, Xy_train, dd, df
    del X_valid, y_valid, dvalid, model
    _ = gc.collect()

print('#' * 25)
oof = pd.concat(oof, axis=0, ignore_index=True).set_index('customer_ID')
acc = amex_metric_mod(oof.target.values, oof.oof_pred.values)
print('OVERALL CV Kaggle Metric =', acc)

# clean ram
del train
_ = gc.collect()

oof_xgb = pd.read_parquet(TRAIN_PATH, columns=['customer_ID']).drop_duplicates()
oof_xgb['customer_ID_hash'] = oof_xgb['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype('int64')
oof_xgb = oof_xgb.set_index('customer_ID_hash')
oof_xgb = oof_xgb.merge(oof, left_index=True, right_index=True)
oof_xgb = oof_xgb.sort_index().reset_index(drop=True)
oof_xgb.to_csv(f'oof_xgb_v{VER}.csv', index=False)
oof_xgb.head()

# plot oof predictions
plt.hist(oof_xgb.oof_pred.values, bins=100)
plt.title('OOF Predictions')
plt.show()

# clear vram, ram for inference below
del oof_xgb, oof
_ = gc.collect()

df = importances[0].copy()
for k in range(1, FOLDS):
    df = df.merge(importances[k], on='feature', how='left')
df['importance'] = df.iloc[:, 1:].mean(axis=1)
df = df.sort_values('importance', ascending=False)
df.to_csv(f'xgb_feature_importance_v{VER}.csv', index=False)

NUM_FEATURES = 20
plt.figure(figsize=(10, 5 * NUM_FEATURES // 10))
plt.barh(np.arange(NUM_FEATURES, 0, -1), df.importance.values[:NUM_FEATURES])
plt.yticks(np.arange(NUM_FEATURES, 0, -1), df.feature.values[:NUM_FEATURES])
plt.title(f'XGB Feature Importance - Top {NUM_FEATURES}')
plt.show()


# calculate size of each separate test part
def get_rows(customers, test, NUM_PARTS=4, verbose=''):
    chunk = len(customers)//NUM_PARTS
    if verbose != '':
        print(f'We will process {verbose} data as {NUM_PARTS} separate parts.')
        print(f'There will be {chunk} customers in each part (except the last part).')
        print('Below are number of rows in each part:')
    rows = []

    for k in range(NUM_PARTS):
        if k == NUM_PARTS - 1:
            cc = customers[k * chunk:]
        else:
            cc = customers[k*chunk:(k + 1) * chunk]
        s = test.loc[test.customer_ID.isin(cc)].shape[0]
        rows.append(s)
    if verbose != '':
        print(rows)
    return rows, chunk


# compute size of 4 parts for test data
NUM_PARTS = 4
TEST_PATH = os.path.join(data, 'test.parquet')

print(f'Reading test data...')
test = read_file(path=TEST_PATH, usecols=['customer_ID', 'S_2'])
customers = test[['customer_ID']].drop_duplicates().sort_index().values.flatten()
rows, num_cust = get_rows(customers, test[['customer_ID']], NUM_PARTS=NUM_PARTS, verbose='test')


# infer test data in parts
skip_rows = 0
skip_cust = 0
test_preds = []

for k in range(NUM_PARTS):
    # read part of test data
    print(f'\nReading test data...')
    test = read_file(path = TEST_PATH)
    test = test.iloc[skip_rows:skip_rows+rows[k]]
    skip_rows += rows[k]
    print(f'=> Test part {k+1} has shape', test.shape )

    # process and feature engineer part of test data
    test = process_and_feature_engineer(test)
    if k==NUM_PARTS-1: test = test.loc[customers[skip_cust:]]
    else: test = test.loc[customers[skip_cust:skip_cust+num_cust]]
    skip_cust += num_cust

    # test data for xgb
    X_test = test[FEATURES]
    dtest = xgb.DMatrix(data=X_test)
    test = test[['P_2_mean']] # reduce memory
    del X_test
    gc.collect()

    # infer xgb models on test data
    model = xgb.Booster()
    model.load_model(f'XGB_v{VER}_fold0.xgb')
    preds = model.predict(dtest)
    for f in range(1, FOLDS):
        model.load_model(f'XGB_v{VER}_fold{f}.xgb')
        preds += model.predict(dtest)
    preds /= FOLDS
    test_preds.append(preds)

    # clean memory
    del dtest, model
    _ = gc.collect()

# write submission file
test_preds = np.concatenate(test_preds)
test = cudf.DataFrame(index=customers,data={'prediction':test_preds})
sub = cudf.read_csv(os.path.join(data, 'sample_submission.csv')[['customer_ID']])
sub['customer_ID_hash'] = sub['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
sub = sub.set_index('customer_ID_hash')
sub = sub.merge(test[['prediction']], left_index=True, right_index=True, how='left')
sub = sub.reset_index(drop=True)

# display predictions
sub.to_csv(f'submission_xgb_v{VER}.csv',index=False)
print('Submission file shape is', sub.shape )
sub.head()

# plot predictions
plt.hist(sub.to_pandas().prediction, bins=100)
plt.title('Test Predictions')
plt.show()
