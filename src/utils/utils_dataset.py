'''
Dataset configuration for the temporal evaluation tasks.
You should update this file to use your own dataset.
'''

import random
import pandas as pd
import os
import re

# update this dataset list before you define one
DATASET_LIST = ['olympic', 'dyknow', 'legal', 'environ', 'culture', 'movie', 'olympic_joined', 'medical']


# Helper functions ============================================
def set_random_seed(seed):
    random.seed(seed)


def get_dataset(task):
    '''
    Return task specific dataset, reading from data_dir.
    '''

    # set data_dir
    project_root = os.path.dirname(os.path.abspath(__file__))
    for _ in range(2):
        project_root = os.path.dirname(project_root)
    data_dir = os.path.join(project_root, "dataset", "crafted")


    # Provide dataset based on the task
    if task == 'legal':
        return SameSexLaw(os.path.join(data_dir, 'legal', 'legal.csv'))

    elif task == 'environ':
        return Carbon(os.path.join(data_dir, 'environ', 'environ.csv'))

    elif task == 'culture':
        return Heritage(os.path.join(data_dir, 'culture', 'culture.csv'))
    
    elif task == 'movie':
        return Movie(os.path.join(data_dir, 'movie', 'movie.csv'))
    
    elif task == 'olympic':
        return Olympic(os.path.join(data_dir, 'olympic', 'olympic.csv'))
    
    elif task == 'olympic_joined':
        return OlympicJoined(os.path.join(data_dir, 'olympic', 'olympic_dyknow_leaders.csv'))

    elif task == 'dyknow': # dyknow consists of three datasets
        df_name_list = ['dyknow_leaders.csv', 'dyknow_sports.csv', 'dyknow_organizations.csv']
        df_path_list = [os.path.join(data_dir, 'dyknow', df) for df in df_name_list]

        return Dyknow(df_path_list)
    
    elif task == 'medical':
        return Medical(os.path.join(data_dir, 'medical', 'medical.csv'))
    
    else:
        raise ValueError(f"Invalid task: {task}")
    

def df_to_str(df):
    '''
    Convert a DataFrame to a string.
    '''
    headers = "| " + " | ".join(df.columns) + " |"
    separators = "| " + " | ".join(['---'] * len(df.columns)) + " |"
    
    rows = [
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for _, row in df.iterrows()
    ]
    
    markdown = "\n".join([headers, separators] + rows)
    return markdown

# ============================================


'''
Base class for temporal datbases.
'''
class Mydataset:
    df = []
    df_curr = []
    fd = []
    granularity = []
    binary = None

    def get_length(self, curr = False):
        '''
        Get the length of the dataset.
        If curr = True, return the length of the current rows (i.e., self.df_curr).
        '''

        if len(self.df) == 0:
            return 0
        else: 

            target_df = self.df if not curr else self.df_curr
            length = 0
            for curr_df in target_df:
                length += len(curr_df)
            return length
        

    def get_current_rows(self):
        '''
        Get only current rows from the dataset. 
        This should be called before temporal alignment task.
        '''

        if self.get_length() == 0:
            raise ValueError("Dataset is empty!!")
            
        # get current rows
        print("Creating current rows...")
        new_df_list = []
        for curr_df in self.df: 
            curr_df = curr_df[curr_df['End'].isna()].copy()
            new_df_list.append(curr_df)
            print(f"Current df length: {len(curr_df)}")

        self.df_curr = new_df_list
        print(f"Created current df with total length: {self.get_length(curr=True)}\n")
        return new_df_list


    def set_binary(self, binary_ans_list):
        '''
        Set the binary answer list.
        '''
        self.binary = binary_ans_list


    def get_binary_sysprompt(self):
        '''
        Get the system prompt for binary classification.
        '''
        if self.binary is None:
            raise ValueError("Binary answer list is not set!!")
        
        return f"Answer the following question with {self.binary[0]} or {self.binary[1]}."


    def get_context(self, q_items, return_str=True, id=0):
        '''
        Get the context for the given rows.
        if return_str is True, return the context as a string.
        else, return the context as a DataFrame.
        '''

        df = self.df[id].copy() 
        correct_rows = q_items['correct_rows']
        incorrect_rows = q_items['incorrect_rows']

        # get answer_col
        if len(correct_rows) == 0:
            return '' if return_str else pd.DataFrame()

        else:
            answer_col_key = [k for k in correct_rows[0].keys() if k not in ['Start', 'End']][0]

            correct_names = set(row[answer_col_key] for row in correct_rows)
            incorrect_names = set(row[answer_col_key] for row in incorrect_rows)

            df_correct = df[df[answer_col_key].isin(correct_names)]
            df_incorrect = df[df[answer_col_key].isin(incorrect_names)]

            excluded_names = correct_names.union(incorrect_names)
            df_random_candidates = df[~df[answer_col_key].isin(excluded_names)]
            df_random = df_random_candidates.sample(n=3, random_state=42)  

            df_final = pd.concat([df_correct, df_incorrect, df_random], ignore_index=True)
            df_final_shuffled = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

            if return_str:
                return df_to_str(df_final_shuffled)
            else:
                return df_final_shuffled
    



class Dyknow(Mydataset):
    def __init__(self, df_path_list):
        print(f"Loading DyKnow dataset from...")
        for df_path in df_path_list:
            print(f"\t{df_path}")
        self.df_leaders = pd.read_csv(df_path_list[0])
        self.fd_leaders = [(['Country', 'Role'], ['Name'])]
        self.gran_leaders = 'month'

        self.df_sports = pd.read_csv(df_path_list[1])
        self.fd_sports = [(['Name', 'Sport'], ['Team'])]
        self.gran_sports = 'year'

        self.df_orgs = pd.read_csv(df_path_list[2])
        self.fd_orgs = [(['Organization_name', 'Organization_type', 'Role'], ['Name'])]
        self.gran_orgs = 'year'

        self.df = [self.df_leaders, self.df_sports, self.df_orgs]
        self.fd = [self.fd_leaders, self.fd_sports, self.fd_orgs]
        self.granularity = [self.gran_leaders, self.gran_sports, self.gran_orgs]

        print(f"Total length of the dataset: {self.get_length()}\n")


class SameSexLaw(Mydataset):
    def __init__(self, df_path):
        print(f"Loading Legal dataset from...")
        print(f"\t{df_path}")
        self.df = [pd.read_csv(df_path)] # Country,Law_type,Legality,Start
        self.fd = [[(['Country', 'Law_type'], ['Legality'])]]
        self.granularity = ['year']
        print(f"Total length of the dataset: {self.get_length()}\n")

        self.set_binary(['Yes/Legal', 'No/Illegal'])


class Carbon(Mydataset):
    def __init__(self, df_path):
        print(f"Loading Carbon dataset from...")
        print(f"\t{df_path}")
        self.df = [pd.read_csv(df_path)]
        self.fd = [[(['Jurisdiction', 'Type'], ['Status'])]]
        self.granularity = ['year']
        print(f"Total length of the dataset: {self.get_length()}\n")

        self.set_binary(['Yes/Implemented', 'No/Not Implemented'])


class Heritage(Mydataset):
    def __init__(self, df_path):
        print(f"Loading Heritage dataset from...")
        print(f"\t{df_path}")
        self.df = [pd.read_csv(df_path, dtype={"End": str})]
        self.fd = [[(['Heritage_element'], ['Status_Inscribed_or_Proclaimed'])]]
        self.granularity = ['year']
        print(f"Total length of the dataset: {self.get_length()}\n")

        self.set_binary(['Inscribed', 'Proclaimed'])


class Movie(Mydataset):
    def __init__(self, df_path):
        print(f"Loading Movie dataset from...")
        print(f"\t{df_path}")
        self.df = [pd.read_csv(df_path)]
        self.fd = [[(['Title'], ['Director'])]]
        self.granularity = ['year']
        print(f"Total length of the dataset: {self.get_length()}\n")


class Olympic(Mydataset):   
    def __init__(self, df_path):
        print(f"Loading Olympic dataset from...")
        print(f"\t{df_path}")
        self.df = [pd.read_csv(df_path)]
        self.fd = [[(['Game_edition', 'Season', 'Role'], ['Name'])]]
        self.granularity = ['year']
        print(f"Total length of the dataset: {self.get_length()}\n")



class OlympicJoined(Mydataset):
    def __init__(self, df_path):
        print(f"Loading Olympic_joined dataset from...")
        print(f"\t{df_path}")
        self.df = [pd.read_csv(df_path)]
        self.fd = [
            {
                '2hop': [(['Game_name', 'Role'], ['Name'])],
                '3hop': [(['Game_edition', 'Season', 'Role'], ['Name'])],
            }
        ]
        self.granularity = ['month']
        print(f"Total length of the dataset: {self.get_length()}\n")


class Medical(Mydataset):
    def __init__(self, df_path):
        print(f"Loading Medical dataset from...")
        print(f"\t{df_path}")
        self.df = [pd.read_csv(df_path)] # Country,Drug_name,Year
        self.fd = [(['Name', 'Gender', 'Blood_Type'], ['Doctor']), (['Name', 'Gender', 'Blood_Type'], ['Medication'])]
        self.granularity = ['month']
        print(f"Total length of the dataset: {self.get_length()}\n")


    def get_context(self, q_items, return_str=True, id=0):

        df = self.df[id].copy()
        correct_rows = q_items['correct_rows']
        incorrect_rows = q_items['incorrect_rows']
        sql = q_items['sql_for_db']

        # get answer_col
        if len(correct_rows) == 0:
            return '' if return_str else pd.DataFrame()

        else:
            answer_col_key = [k for k in correct_rows[0].keys() if k not in ['Start', 'End']][0]

            if answer_col_key == 'Medication':
                name = re.search(r'Name\s*=\s*"([^"]+)"', sql).group(1)
                gender = re.search(r'Gender\s*=\s*"([^"]+)"', sql).group(1)
                blood = re.search(r'Blood_Type\s*=\s*"([^"]+)"', sql).group(1)

                df_related = df[(df['Name'] == name) & (df['Gender'] == gender) & (df['Blood_Type'] == blood)]
                df_random_candidates = df[(df['Name'] != name) | (df['Gender'] != gender) | (df['Blood_Type'] != blood)]
                df_random = df_random_candidates.sample(n=7, random_state=42)

                df_final = pd.concat([df_related, df_random], ignore_index=True)
                df_final_shuffled = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

                if return_str:
                    return df_to_str(df_final_shuffled)
                else:
                    return df_final_shuffled

            
            else:
                correct_names = set(row[answer_col_key] for row in correct_rows)
                incorrect_names = set(row[answer_col_key] for row in incorrect_rows)

                df_correct = df[df[answer_col_key].isin(correct_names)]
                df_incorrect = df[df[answer_col_key].isin(incorrect_names)]

                excluded_names = correct_names.union(incorrect_names)
                df_random_candidates = df[~df[answer_col_key].isin(excluded_names)]
                df_random = df_random_candidates.sample(n=7, random_state=42)

                df_final = pd.concat([df_correct, df_incorrect, df_random], ignore_index=True)
                df_final_shuffled = df_final.sample(frac=1, random_state=42).reset_index(drop=True)


                if return_str:
                    return df_to_str(df_final_shuffled)
                else:
                    return df_final_shuffled
    