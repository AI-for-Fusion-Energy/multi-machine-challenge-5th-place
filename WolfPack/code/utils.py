import os
import glob
from jddb.file_repo import FileRepo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from tqdm.auto import tqdm

#load training and testing file repos

def open_repo(path):
    """
    Opens File a file repository when a path to dir is pass
    """

    file_repo = FileRepo(f"{path}")
    return file_repo

def get_labels(repo , shot_list):
    
    """
    Returns labels for shots in repo specifies in the shot_list
    """

    is_disrupt = []
    for shot in shot_list:
        dis_label = repo.read_labels(shot, ['IsDisrupt'])
        is_disrupt.append(dis_label['IsDisrupt'])

    return is_disrupt

def get_maximum_length(tag):

    """
    Returns a mini tag list if length of tags for the specific shot is == maximum length of tags in the list
    """

    all_lengths = [len(tag) for tag in tag ]

    max_tags = [tag for tag in tag if len(tag)==max(all_lengths)]
    return max_tags


def get_tags_and_mini_tags(repo , shots_dict_key):
    """returns all Tags and mini_tags based on the get_maximum_length function to a specific shot"""
    tags = [repo.get_tag_list(each_shot) for each_shot in shots_dict_key]
    mini_tags = get_maximum_length(tags)
    tags = np.unique(np.array([cv for te in tags for cv in te]))

    return tags , mini_tags

def new_test_matrix_build(file_shot_repo, shot_list, tags, all_tags):
    """Returns X arrays and tags to drop """
    file_shot_repo_dict = [dict(file_shot_repo.read_data(shot_list[i] , tag_list=tags).items()) for i in range(len(shot_list))]
    tag_to_drop = np.unique(np.array([key for data_dict in file_shot_repo_dict for key in all_tags if key not in data_dict.keys()]))
    file_x = np.asarray([np.nan_to_num(data_dict[f"{key}"]).T for data_dict in file_shot_repo_dict for key in all_tags if key not in tag_to_drop]  , dtype=object).reshape(-1 , len(all_tags) - len(tag_to_drop))
    return file_x , tag_to_drop

def new_train_matrix_build(shot_list, file_shot_repo , tags, all_tags):
    """Returns X & Y arrays and tags_to_drop """
    file_shot_repo_dict = [dict(file_shot_repo.read_data(shot_list[i], tag_list=tags).items()) for i in range(len(shot_list))]
    tag_to_drop = np.unique(np.array([key for data_dict in file_shot_repo_dict for key in all_tags if key not in data_dict.keys()]))
    file_x = np.asarray([np.nan_to_num(data_dict[f"{key}"]).T for data_dict in file_shot_repo_dict for key in all_tags if key not in tag_to_drop], dtype=object).reshape(-1 , len(all_tags) - len(tag_to_drop))

    file_y = np.asarray([np.array(list(file_shot_repo.read_labels(shot_list[i] , ['IsDisrupt']).values())).T.flatten().reshape(-1,) for i in range(len(shot_list))] , dtype=np.int32)
    return file_x , file_y , tag_to_drop



def reform_train_x_sum_of_squares_features(df , mini_tags , shot_list):
    """Returns a reformed dataframe"""
    train_df_dict = {i : [np.square(df[v][i]).sum() for v in range(df.shape[0])] for i in range(df.shape[1])}

    df_final = pd.DataFrame(train_df_dict)
    df_final = df_final.rename(columns  = dict(zip(train_df_dict.keys(), mini_tags )) )
    df_final['Shot_list'] = [f'ID_{shot}' for shot in shot_list]
    return df_final

def rebuild_train_files(train_file_repo , train_shot_list):
    df_final_x = [ ]
    df_final_y = [ ]

    for shot in tqdm(train_shot_list):

        train_tags , mini_tags  = get_tags_and_mini_tags(train_file_repo , [shot])
        train_x_df , train_y_df, _ =  new_train_matrix_build([shot] , train_file_repo , mini_tags[0] , train_tags)
        train_x_df = reform_train_x_sum_of_squares_features(train_x_df , mini_tags[0] , [shot])

        train_y_df = get_labels(train_file_repo , [shot])
        train_y_df = pd.DataFrame(train_y_df)

        df_final_x.append(train_x_df)
        df_final_y.append(train_y_df)

    df_final_x = pd.concat(df_final_x)
    df_final_y = pd.concat(df_final_y)

    df_final_x = df_final_x.reset_index().drop('index' , axis=1)
    df_final_y = df_final_y.reset_index().drop('index' , axis=1)

    return df_final_x , df_final_y

def rebuild_test_files(test_file_repo , test_shot_list):
    df_final_x = [ ]

    for shot in tqdm(test_shot_list):

        test_tags , mini_tags  = get_tags_and_mini_tags(test_file_repo , [shot])
        test_x_df , _ =  new_test_matrix_build(test_file_repo, [shot] , mini_tags[0] , test_tags)
        test_x_df = reform_train_x_sum_of_squares_features(test_x_df , mini_tags[0] , [shot])
        df_final_x.append(test_x_df)

    df_final_x = pd.concat(df_final_x)
    df_final_x = df_final_x.reset_index().drop('index' , axis=1)
    return df_final_x