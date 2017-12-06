#!/usr/bin/env python
# coding=utf-8

import numpy as np
import pandas as pd
import lightgbm as lgb
import gc


"""
增加训练集和验证集
"""

print('Loading data...')
data_path = '../input/'
output_path = "../result/"

train = pd.read_csv(data_path + 'train.csv')
train['target'] = train['target'].astype(np.uint8)
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv')
members = pd.read_csv(data_path + 'members.csv',
                      parse_dates=['registration_init_time', 'expiration_date'])
members['bd'] = members['bd'].astype(np.uint8)
songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')

print('Data preprocessing...')

train = train.merge(songs, on='song_id', how='left')
test = test.merge(songs, on='song_id', how='left')

# 特征工程开始
members['membership_days'] = members['expiration_date'].subtract(
    members['registration_init_time']).dt.days.astype(int)


members['registration_year'] = members['registration_init_time'].dt.year
members['registration_month'] = members['registration_init_time'].dt.month
members['registration_date'] = members['registration_init_time'].dt.day

members['expiration_year'] = members['expiration_date'].dt.year
members['expiration_month'] = members['expiration_date'].dt.month
members['expiration_date'] = members['expiration_date'].dt.day

members = members.drop(['registration_init_time'], axis=1)


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan

def isrc_to_country(isrc):

    if type(isrc) == str:
        return isrc[:2]
    else:
        return np.nan

def is_english(name):
    if(type(name) != str):
        print("not str")
        return 0
    name = name.lower()
    if len(name) > 0 and name[0] >= 'a' and name[0] <= 'z':
        return 1
    return 0


songs_extra['is_english'] = songs_extra['name'].apply(is_english)
songs_extra.is_english = songs_extra.is_english.astype('category')
songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis=1, inplace=True)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')

train = train.merge(songs_extra, on='song_id', how='left')
train.song_length.fillna(200000, inplace=True)
train.song_length = train.song_length.astype(np.uint32)
train.song_id = train.song_id.astype('category')

test = test.merge(songs_extra, on='song_id', how='left')
test.song_length.fillna(200000, inplace=True)
test.song_length = test.song_length.astype(np.uint32)
test.song_id = test.song_id.astype('category')
del members, songs
gc.collect()

print("done merging!")

print("adding new feature")

def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') + 1


train['genre_ids'].fillna('no_genre_id', inplace=True)
test['genre_ids'].fillna('no_genre_id', inplace=True)
train['genre_ids_count'] = train['genre_ids'].apply(
    genre_id_count).astype(np.int8)
test['genre_ids_count'] = test['genre_ids'].apply(
    genre_id_count).astype(np.int8)


def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
    return sum(map(x.count, ['|', '/', '\\', ';']))


train['lyricist'].fillna('no_lyricist', inplace=True)
test['lyricist'].fillna('no_lyricist', inplace=True)
train['lyricists_count'] = train['lyricist'].apply(
    lyricist_count).astype(np.int8)
test['lyricists_count'] = test['lyricist'].apply(
    lyricist_count).astype(np.int8)


def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1


train['composer'].fillna('no_composer', inplace=True)
test['composer'].fillna('no_composer', inplace=True)
train['composer_count'] = train['composer'].apply(
    composer_count).astype(np.int8)
test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)


def is_featured(x):
    if 'feat' in str(x):
        return 1
    return 0


train['artist_name'].fillna('no_artist', inplace=True)
test['artist_name'].fillna('no_artist', inplace=True)
train['is_featured'] = train['artist_name'].apply(is_featured).astype(np.int8)
test['is_featured'] = test['artist_name'].apply(is_featured).astype(np.int8)


def artist_count(x):
    if x == 'no_artist':
        return 0
    else:
        return x.count('and') + x.count(',') + x.count('feat') + x.count('&')


train['artist_count'] = train['artist_name'].apply(
    artist_count).astype(np.int8)
test['artist_count'] = test['artist_name'].apply(artist_count).astype(np.int8)
train['artist_composer'] = (train['artist_name'] ==
                            train['composer']).astype(np.int8)
test['artist_composer'] = (test['artist_name'] ==
                           test['composer']).astype(np.int8)

# if artist, lyricist and composer are all three same
train['artist_composer_lyricist'] = ((train['artist_name'] == train['composer']) & (
    train['artist_name'] == train['lyricist']) & (train['composer'] == train['lyricist'])).astype(np.int8)
test['artist_composer_lyricist'] = ((test['artist_name'] == test['composer']) & (
    test['artist_name'] == test['lyricist']) & (test['composer'] == test['lyricist'])).astype(np.int8)



def song_lang_boolean(x):
    # is song language 17 or 45.
    if '17.0' in str(x) or '45.0' in str(x):
        return 1
    return 0


train['song_lang_boolean'] = train['language'].apply(
    song_lang_boolean).astype(np.int8)
test['song_lang_boolean'] = test['language'].apply(
    song_lang_boolean).astype(np.int8)

_mean_song_length = np.mean(train['song_length'])


def smaller_song(x):
    if x < _mean_song_length:
        return 1
    return 0


train['smaller_song'] = train['song_length'].apply(
    smaller_song).astype(np.int8)
test['smaller_song'] = test['song_length'].apply(smaller_song).astype(np.int8)

# number of times a song has been played before
_dict_count_song_played_train = {
    k: v for k, v in train['song_id'].value_counts().iteritems()}
_dict_count_song_played_test = {
    k: v for k, v in test['song_id'].value_counts().iteritems()}


def count_song_played(x):
    try:
        return _dict_count_song_played_train[x]
    except KeyError:
        try:
            return _dict_count_song_played_test[x]
        except KeyError:
            return 0


train['count_song_played'] = train['song_id'].apply(
    count_song_played).astype(np.int64)
test['count_song_played'] = test['song_id'].apply(
    count_song_played).astype(np.int64)

# number of times the artist has been played
_dict_count_artist_played_train = {
    k: v for k, v in train['artist_name'].value_counts().iteritems()}
_dict_count_artist_played_test = {
    k: v for k, v in test['artist_name'].value_counts().iteritems()}


def count_artist_played(x):
    try:
        return _dict_count_artist_played_train[x]
    except KeyError:
        try:
            return _dict_count_artist_played_test[x]
        except KeyError:
            return 0


train['count_artist_played'] = train['artist_name'].apply(
    count_artist_played).astype(np.int64)
test['count_artist_played'] = test['artist_name'].apply(
    count_artist_played).astype(np.int64)

train['count_artist_played'] = train['artist_name'].apply(
    count_artist_played).astype(np.int64)
test['count_artist_played'] = test['artist_name'].apply(
    count_artist_played).astype(np.int64)

print("Done adding features")

print("Train test and validation sets")

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

# train.to_csv("../feature_merge/train.csv",index=False)
# test.to_csv("../feature_merge/test.csv",index=False)

X_train = train.drop(['target'], axis=1)
y_train = train['target'].values

X_test = test.drop(['id'], axis=1)
ids = test['id'].values

del train, test
gc.collect()


print("Train test and validation sets")
X_tr=X_train[:5000000]
y_tr=y_train[:5000000]
X_val=X_train[5000000:]
y_val=y_train[5000000:]

lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_val = lgb.Dataset(X_val, y_val)
print('Processed data...')



print('Processed data...')

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'gbdt',
    'learning_rate': 0.2,
    'verbose': 0,
    'num_leaves': 108,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,
    'max_bin': 256,
    'max_depth': 20,
    'num_rounds': 400,
    'metric': 'auc'
}

print('Training LGBM model...')
#params['application'] = 'binary'
#params['verbosity'] = 0

# model = lgb.train(params, train_set=d_train, num_boost_round=200, valid_sets=watchlist,
#                   verbose_eval=5)
model_f1 = lgb.train(params, train_set=lgb_train,
                     valid_sets=lgb_val, verbose_eval=5)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'dart',
    'learning_rate': 0.2,
    'verbose': 0,
    'num_leaves': 108,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,
    'max_bin': 256,
    'max_depth': 20,
    'num_rounds': 400,
    'metric': 'auc'
}

model_f2 = lgb.train(params, train_set=lgb_train,
                     valid_sets=lgb_val, verbose_eval=5)

print('Making predictions')
p_test_1 = model_f1.predict(X_test)
p_test_2 = model_f2.predict(X_test)
p_test_avg = np.mean([p_test_1, p_test_2], axis=0)
print('Done making predictions')

print('Saving predictions Model model of gbdt')

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test_avg
subm.to_csv(output_path + 'submission_lgbm_avg_v2.csv',
            index=False, float_format='%.5f')

print('Done!')
