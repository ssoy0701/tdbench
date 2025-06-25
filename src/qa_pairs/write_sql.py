'''
Write SQL queries for TSQA tasks. Save a json file with id, sql_query, and answer_cnt.
'''


import pandas as pd
from sqlite3 import connect
import json
import argparse
import os
import random

import utils.utils_dataset as dconfig


# argparser
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices = dconfig.DATASET_LIST, default='dyknow', help='name of the dataset. Be sure you have updated utils_dataset.py')
parser.add_argument("--qtype", '-q', type=str, choices = ['now', 'basic', 'join'], default='basic', help='use now for temporal alignment task, basic for temporal reasoning task, join for the multi hop setting')
parser.add_argument("--random_seed", type=int, default='1116')
parser.add_argument("--df_idx", type=int, default=0, help="If the dataset has multiple dataframes (e.g., dyknow), choose idx of the target dataframe")
parser.add_argument("--sample", type=int, default=100, help="Number of samples to generate from each operation for temporal reasoning task.")
args = parser.parse_args()


RESULT_DIR = '../qa_pairs'
MAX_DATE_STR = '2262-04-11 00:00:00' 


# config
task = args.task
qtype = args.qtype
sample_num = args.sample
save_path = os.path.join(RESULT_DIR, task, f'{qtype}_sql.json')
if task == 'dyknow': # multiple dataframes
    save_path = save_path.replace("_sql", f"_{args.df_idx}_sql")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
random.seed(args.random_seed)



# read data and get fd, granularity of the target dataset
dataset = dconfig.get_dataset(task)
df_list = dataset.df
fd_list = dataset.fd


df = df_list[args.df_idx]
fds = fd_list[args.df_idx]
granularity = dataset.granularity[args.df_idx] 


# print data config
print("=====================================")
print("Task: ", task)
print("Query type: ", qtype)
print("Number of rows: ", dataset.get_length())
print("Functional dependencies: ", fds)
print("Granularity: ", granularity)
if task == 'basic':
    print("Sample number: ", sample_num)
print("=====================================")



def get_now_date():
    now = pd.Timestamp.now().date()
    return pd.Timestamp(now)



### temporal alignmnet task to ask about the current world's knowledge
if qtype == 'now': 

    generated_query_list = []
    total_query_list = []
    query_idx = 0
    
    # preprocess data columns to be datetime
    df['Start'] = pd.to_datetime(df['Start'].astype(str), errors='coerce')
    df = df.dropna(subset=['Start'])
    df['End'] = pd.to_datetime(df['End'].astype(str))

    # connect to sqlite
    conn = connect(':memory:')
    df.to_sql(task, conn, if_exists='replace', index=False)


    # iterate thorugh FDs
    for fd in fds:

        # generate base_query w.r.t. FDs
        fd_lhs_list, fd_rhs_list = fd

        conditions = " and ".join([f"{col} = [{col}]" for col in fd_lhs_list])
        base_query = f"select * from {task} where {conditions}"
        print("Base query: ", base_query)


    
        # get row info to generate queries
        for row_id, row in df.iterrows():

            # if not current row, skip
            if not pd.isna(row['End']):
                continue

            curr_query = base_query
            for fd_lhs in fd_lhs_list:
                value = row[fd_lhs]
                curr_query = curr_query.replace(f'[{fd_lhs}]', f'"{value}"') 
            

            # task-specific
            if task == 'movie':

                prev_query = generated_query_list[-1] if len(generated_query_list) > 0 else 'None'
                query = curr_query + f"[TIME] order by release_year desc limit 1"
                
                if prev_query == query:
                    continue
                generated_query_list.append(query)

                # save queries with metadata
                query_dict = {}
                query_dict['idx'] = query_idx

                fd_rhs = fd_rhs_list[0]


                gold_query = query.replace('[TIME]', '').replace('select *', f'select {fd_rhs}, start, end')
                gold_answer_rows = pd.read_sql(gold_query, conn)
                gold_answer_list = gold_answer_rows.to_dict(orient='records')

                incorrect_answer_list = []



            else:
                query = curr_query + f"[TIME] and End IS NULL"
                generated_query_list.append(query)

                # save queries with metadata
                query_dict = {}
                query_dict['idx'] = query_idx

                fd_rhs = fd_rhs_list[0]

                related_query = query.split('[TIME]')[0].replace('select *', f'select {fd_rhs}, start, end')
                related_rows = pd.read_sql(related_query, conn)

                gold_query = query.replace('[TIME]', '').replace('select *', f'select {fd_rhs}, start, end')
                gold_answer_rows = pd.read_sql(gold_query, conn)
                gold_answer_list = gold_answer_rows.to_dict(orient='records')

                incorrect_answer_rows = pd.concat([related_rows, gold_answer_rows]).drop_duplicates(keep=False)
                incorrect_answer_list = incorrect_answer_rows.to_dict(orient='records')
            
            query_dict['qtype'] = 'now'
            query_dict['sql_for_db'] = gold_query
            query_dict['correct_rows'] = gold_answer_list
            query_dict['incorrect_rows'] = incorrect_answer_list

            query_idx += 1
            total_query_list.append(query_dict)

        print("Sample query: ", gold_query)
        print("\tGold answer: ", gold_answer_list)
        print("\tIncorrect answer: ", incorrect_answer_list)

        print("Total number of queries: ", len(total_query_list))

    # save queries to json file
    with open(save_path, "a", encoding="utf-8") as f:
        json.dump(total_query_list, f, indent=4)
    print(f"Total {query_idx} queries saved to {save_path}.")

    conn.close()



### Basic questions


if qtype == 'basic':

    # preprocess data columns to be datetime
    df['Start'] = pd.to_datetime(df['Start'].astype(str), errors='coerce')
    df = df.dropna(subset=['Start'])

    df['End'] = pd.to_datetime(df['End'])
    df['End'] = df['End'].fillna(pd.Timestamp(pd.Timestamp.max.date()).strftime('%Y-%m-%d'))

    # connect to sqlite
    conn = connect(':memory:')
    df.to_sql(task, conn, if_exists='replace', index=False)


    # basic temporal operations
    basic_op_lists = ['after', 'before', 'equal', 'contain', 'finish', 'start', 'meet', 'met-by', 'overlap', 'during', 'finished-by', 'started-by', 'overlapped-by']
    generated_query_list = {op: [] for op in basic_op_lists}

    
    for fd in fds:

        # generate base_query w.r.t. FDs
        fd_lhs_list, fd_rhs_list = fd

        conditions = " and ".join([f"{col} = [{col}]" for col in fd_lhs_list])
        base_query = f"select * from {task} where {conditions}"
        print("Base query: ", base_query)


        for row_id, row in df.iterrows():

            # get row info to generate queries
            start = pd.to_datetime(row['Start'])
            end = pd.to_datetime(row['End'])

            # replace with values
            curr_query = base_query
            for fd_lhs in fd_lhs_list:
                value = row[fd_lhs]
                curr_query = curr_query.replace(f'[{fd_lhs}]', f'"{value}"') 

            # get related rows to limit the min/max time interval
            related_rows = pd.read_sql(curr_query, conn)
            if granularity == 'month':
                min_start = pd.to_datetime(related_rows['Start'].min()) - pd.DateOffset(years=2)

                if related_rows['End'].max() == MAX_DATE_STR:
                    max_end = get_now_date() + pd.DateOffset(years=2)
                else:
                    max_end = pd.to_datetime(related_rows['End'].max()) + pd.DateOffset(years=2)

                # sample start_year, end_year within the interval
                sampled_interval = pd.date_range(start=min_start, end=max_end, freq='M').to_list()

            else:
                min_start = pd.to_datetime(related_rows['Start'].min()) - pd.DateOffset(years=5)

                if related_rows['End'].max() == MAX_DATE_STR:
                    max_end = get_now_date() + pd.DateOffset(years=5)
                else:
                    max_end = pd.to_datetime(related_rows['End'].max()) + pd.DateOffset(years=5)

                # sample start_year, end_year within the interval
                sampled_interval = pd.date_range(start=min_start, end=max_end, freq='Y').to_list()

            curr_query = curr_query.replace('select *', f'select {fd_rhs_list[0]}, start, end')


            # add temporal context w.r.t. temporal operations
            for op in basic_op_lists:
                if op == 'after':
                    if granularity == 'month':
                        start_interval = pd.date_range(start=max(start - pd.DateOffset(years=3), min_start), \
                                                    end=start, freq='M').to_list()
                        sample = random.choice(start_interval) 
                        sample_maxday = sample.replace(day=sample.days_in_month) # "after July" means "after July 31st"

                    else:
                        start_interval = pd.date_range(start=max(start - pd.DateOffset(years=7), min_start), \
                                                    end=start, freq='Y').to_list()
                        sample = random.choice(start_interval) 
                        sample_maxday = sample.replace(month=12, day=31) # "after 2010" means "after Dec 31st"
                        
                    query = curr_query + f"[TIME] and Start > '{sample_maxday}'"


                elif op == 'before':
                    if end == get_now_date():
                        end_interval = pd.date_range(start=start, end=start + pd.DateOffset(years=3), freq='Y').to_list() # just for now
                        sample = random.choice(end_interval)
                        sample_minday = sample.replace(month=1, day=1)
                        query = curr_query + f"[TIME] and Start < '{sample}'"

                    else:
                        if granularity == 'month':
                            if str(end) == MAX_DATE_STR:
                                end = get_now_date()
                                
                            end_interval = pd.date_range(start=end, \
                                                        end=min(end + pd.DateOffset(years=3), max_end), \
                                                        freq='M').to_list()
                            sample = random.choice(end_interval)
                            sample_minday = sample.replace(day=1) # "before July" means "before July 1st"
                            query = curr_query + f"[TIME] and End < '{sample}'"

                        else:
                            if str(end) == MAX_DATE_STR:
                                end = get_now_date()
                                end_interval = pd.date_range(start=end, \
                                                            end=min(end + pd.DateOffset(years=7), max_end ), \
                                                            freq='Y').to_list()
                            else:
                                end_interval = pd.date_range(start=end, \
                                                            end=min(end + pd.DateOffset(years=5), max_end ), \
                                                            freq='Y').to_list()
                            sample = random.choice(end_interval)
                            sample_minday = sample.replace(month=1, day=1) # "before 2010" means "before Jan 1st"\
                            query = curr_query + f"[TIME] and End < '{sample}'"


                    

                elif op == 'equal':
                    if granularity == 'month':
                        start_query = f"Start between '{start.replace(day=1)}' and '{start.replace(day=start.days_in_month)}'"
                        end_query = f"End between '{end.replace(day=1)}' and '{end.replace(day=end.days_in_month)}'"
                    else:
                        start_query = f"Start between '{start.replace(month=1, day=1)}' and '{start.replace(month=12, day=31)}'"
                        end_query = f"End between '{end.replace(month=1, day=1)}' and '{end.replace(month=12, day=31)}'"
                    
                    if end == get_now_date():
                        query = curr_query + f"[TIME] and {start_query}"
                    else: 
                        query = curr_query + f"[TIME] and {start_query} and {end_query}"


                elif op == 'during':
                    if end == get_now_date():
                        start_first = start - pd.DateOffset(years=2)
                        start_second = start + pd.DateOffset(years=2)
                        start_query = f"Start between '{start_first}' and '{start_second}'"
                        query = curr_query + f"[TIME] and {start_query}"
                    else:
                        start_query = f"Start > '{start.replace(month=1, day=1)}'"
                        end_query = f"End < '{end.replace(month=12, day=end.days_in_month)}'" 
                        query = curr_query + f"[TIME] and {start_query} and {end_query}"



                elif op == 'contain':
                    if row_id % 2 == 0:
                        start_query = f"Start < '{start.replace(month=1, day=1)}'"
                        end_query = f"End > '{end.replace(month=12, day=end.days_in_month)}'" 
                        query = curr_query + f"[TIME] and {start_query} and {end_query}"

                    else:
                        offset = random.randint(1, 5)
                        start_offset = start + pd.DateOffset(months=offset)
                        end_offset = end - pd.DateOffset(months=offset)

                        start_query = f"Start < '{start_offset}'"
                        end_query = f"End > '{end_offset}'" 
                        query = curr_query + f"[TIME] and {start_query} and {end_query}"



                elif op == 'finished-by':
                    if end == get_now_date(): # become none query
                        one_year_before = get_now_date() - pd.DateOffset(years=1)
                        end_query = f"End between '{one_year_before.replace(month=1, day=1)}' and '{one_year_before.replace(month=12, day=31)}'"
                
                    else:
                        if granularity == 'month':
                            end_query = f"End between '{end.replace(day=1)}' and '{end.replace(day=end.days_in_month)}'"
                        else:
                            end_query = f"End between '{end.replace(month=1, day=1)}' and '{end.replace(month=12, day=31)}'"

                    query = curr_query + f"[TIME] and {end_query}"


                elif op == 'finish':
                    if end == get_now_date(): # become none query
                        one_year_after = get_now_date() + pd.DateOffset(years=1)
                        end_query = end_query = f"End between '{one_year_after.replace(month=1, day=1)}' and '{one_year_after.replace(month=12, day=31)}'"
                        query = curr_query + f"[TIME] and {end_query}"
                    
                    else:
                        if granularity == 'month':
                            offset = random.randint(1, 12)
                            sample = start - pd.DateOffset(months=offset)
                            start_query = f"Start > '{sample.replace(day=sample.days_in_month)}'" 
                            end_query = f"End between '{end.replace(day=1)}' and '{end.replace(day=end.days_in_month)}'"
                        else:
                            offset = random.randint(1, 5)
                            sample = start - pd.DateOffset(years=offset)
                            start_query = f"Start > '{sample.replace(month=12, day=31)}'" 
                            end_query = f"End between '{end.replace(month=1, day=1)}' and '{end.replace(month=12, day=31)}'"

                        query = curr_query + f"[TIME] and {start_query} and {end_query}"

                
                elif op == 'started-by':                
                    if granularity == 'month':                    
                        start_query = f"Start between '{start.replace(day=1)}' and '{start.replace(day=start.days_in_month)}'"
                    else:
                        start_query = f"Start between '{start.replace(month=1, day=1)}' and '{start.replace(month=12, day=31)}'"
                    query = curr_query + f"[TIME] and {start_query}"



                elif op == 'start':
                    if granularity == 'month':
                        start_query = f"Start between '{start.replace(day=1)}' and '{start.replace(day=start.days_in_month)}'"
                        offset = random.randint(1, 12)

                        if end == get_now_date():
                            query = curr_query + f"[TIME] and {start_query}"
                        else:
                            end_offset = end + pd.DateOffset(months=offset)
                            end_query = f"End < '{end_offset.replace(day=1)}'"
                            query = curr_query + f"[TIME] and {start_query} and {end_query}"
                    else:
                        start_query = f"Start between '{start.replace(month=1, day=1)}' and '{start.replace(month=12, day=31)}'"
                        offset = random.randint(1, 5)

                        if end == get_now_date():
                            query = curr_query + f"[TIME] and {start_query}"
                        else:
                            end_offset = end + pd.DateOffset(years=offset)
                            end_query = f"End < '{end_offset.replace(month=1, day=1)}'"
                            query = curr_query + f"[TIME] and {start_query} and {end_query}"


                elif op == 'meet':
                    if granularity == 'month':
                        offset = random.randint(1, 12)

                        if row_id % 2 == 0:
                            end_query = f"date(End) = date('{end}', '-{offset} month')" # none
                        else:
                            end_offset = end + pd.DateOffset(months=offset)
                            end_query = f"date(End) = date('{end_offset}', '-{offset} month')" # unique
                    
                    else:
                        offset = random.randint(1, 5)
                        end_query = f"date(End) = date('{end}', '-{offset} year')"
                        
                    query = curr_query + f"[TIME] and {end_query}"

                
                elif op == 'met-by':
                    if granularity == 'month':
                        offset = random.randint(1, 12)

                        if row_id % 2 == 0:
                            start_offset = start + pd.DateOffset(months=offset)
                            start_query = f"date(Start) = date('{start_offset}', '-{offset} month')" # unique
                            
                        else:
                            start_query = f"date(Start) = date('{start}', '-{offset} month')" # none

                    else:
                        offset = random.randint(1, 5)
                        start_query = f"date(Start) = date('{start}', '-{offset} year')"

                    query = curr_query + f"[TIME] and {start_query}"   


                elif op == 'overlap':

                    if end == get_now_date():
                        start_interval = pd.date_range(start=start, end=end,freq='Y').to_list()
                        if len(start_interval) > 0:
                            sample = random.choice(start_interval)
                            start_query = f"Start < '{sample}'"
                            end_offset = end - pd.DateOffset(years=1)
                            end_query = f"End between '{sample}' and '{end_offset.replace(month=12, day=31)}'" 
                            
                            query = curr_query + f"[TIME] and {start_query} and {end_query}"
                        else:
                            continue

                    
                    else:
                        if granularity == 'month':
                            start_interval = pd.date_range(start=start, end=end,freq='M').to_list()
                            if len(start_interval) > 0:
                                sample = random.choice(start_interval)
                                start_query = f"Start < '{sample}'"

                                offset = random.randint(1, 5)
                                end_offset = end + pd.DateOffset(years=offset)
                                end_query = f"End between '{sample}' and '{end_offset}'" 
                                
                                query = curr_query + f"[TIME] and {start_query} and {end_query}"
                            else:
                                continue

                        else: 
                            start_interval = pd.date_range(start=start, end=end,freq='Y').to_list()
                            if len(start_interval) > 0:
                                sample = random.choice(start_interval)
                                start_query = f"Start < '{sample}'"

                                offset = random.randint(1, 5)
                                end_offset = end + pd.DateOffset(years=offset)
                                end_query = f"End between '{sample}' and '{end_offset}'" 
                                
                                query = curr_query + f"[TIME] and {start_query} and {end_query}"
                            else:
                                continue


                elif op == 'overlapped-by':
                    if end == get_now_date():
                        start_first = start - pd.DateOffset(years=5)
                        start_second = start + pd.DateOffset(years=5)
                        start_query = f"Start between '{start_first.replace(month=1, day=1)}' and '{start_second.replace(month=12, day=31)}'"
                        query = curr_query + f"[TIME] and {start_query}"

                    else:
                        if granularity == 'month':
                            end_interval = pd.date_range(start=start, end=end,freq='M').to_list()
                            if len(end_interval) > 0:
                                sample = random.choice(end_interval)
                                end_query = f"End > '{sample}'"

                                offset = random.randint(1, 5)
                                start_offset = start - pd.DateOffset(years=offset)
                                start_query = f"Start between '{start_offset}' and '{sample}'" 
                                
                                query = curr_query + f"[TIME] and {start_query} and {end_query}"
                            else:
                                continue

                        else: 
                            end_interval = pd.date_range(start=start, end=end,freq='Y').to_list()
                            if len(end_interval) > 0:
                                sample = random.choice(start_interval)
                                end_query = f"End > '{sample}'"

                                offset = random.randint(1, 5)
                                start_offset = end - pd.DateOffset(years=offset)
                                end_query = f"Start between '{start_offset}' and '{sample}'" 
                                
                                query = curr_query + f"[TIME] and {start_query} and {end_query}"
                            else:
                                continue

                else:
                    raise NotImplementedError("Not implemented operation: ", op)


                generated_query_list[op].append(query)



    # After generating queries, they can be too much... you can sample queries
    total_query_list = []
    query_idx = 0
    for op, generated_query_list in generated_query_list.items():
        
        # shuffle generated queries
        random.shuffle(generated_query_list)

        for query in generated_query_list:

            # save queries with metadata
            query_dict = {}
            query_dict['idx'] = query_idx
            query_dict['op'] = op


            related_query = query.split('[TIME]')[0]
            related_rows = pd.read_sql(related_query, conn)
            related_rows = related_rows.replace("2262-04-11 00:00:00", None)


            gold_query = query.replace('[TIME]', '')
            gold_answer_rows = pd.read_sql(gold_query, conn)
            gold_answer_rows = gold_answer_rows.replace("2262-04-11 00:00:00", None)
            gold_answer_list = gold_answer_rows.to_dict(orient='records')

            incorrect_answer_rows = pd.concat([related_rows, gold_answer_rows]).drop_duplicates(keep=False)
            incorrect_answer_list = incorrect_answer_rows.to_dict(orient='records')
            
            
            if len(gold_answer_list) == 0: # there is no answer
                query_dict['qtype'] = qtype + '_none'

            elif len(gold_answer_list) == 1: # there is unique answer
                query_dict['qtype'] = qtype + '_unique'
                
            else: # there are multiple answers
                query_dict['qtype'] = qtype + '_multiple'

                # limit answers to be at most 5
                if len(gold_answer_list) > 5: 
                    continue


            query_dict['sql_for_db'] = gold_query
            query_dict['correct_rows'] = gold_answer_list
            query_dict['incorrect_rows'] = incorrect_answer_list

            query_idx += 1
            total_query_list.append(query_dict)

            if sample_num > 0:
                if query_idx % sample_num == 0:
                    break


        print(f"Included {op} operation, total {query_idx+1} queries generated.")
        print("Sample query: ", gold_query)
        print("\tGold answer: ", gold_answer_list)
        print("\n\n")


    # print query info
    print("Total number of queries: ", len(total_query_list))


    # count metadata. This can be useful for downstream evaluation 
    none, unique, multiple = 0, 0, 0
    for query in total_query_list:
        if query['qtype'] == 'basic_none':
            none += 1
        elif query['qtype'] == 'basic_unique':
            unique += 1
        else:
            multiple += 1

    print("Number of none queries: ", none)
    print("Number of unique queries: ", unique)
    print("Number of multiple queries: ", multiple)

    
    # save queries to json file
    with open(save_path, "a", encoding="utf-8") as f:
        json.dump(total_query_list, f, indent=4)
    print(f"Total {query_idx} queries saved to {save_path}.")

    conn.close()





### Multi-hop questions
if qtype == 'join':

    # connect to sqlite
    conn = connect(':memory:')
    df.to_sql(task, conn, if_exists='replace', index=False)


    # generated query list w.r.t. hops
    generated_query_list = {}
    for hops, fd_list in fds.items():
        generated_query_list[hops] = {f'{rhs[0]}': [] for _, rhs in fd_list}


    # generate base_query w.r.t. FDs
    for hops, fd_list in fds.items():
        for curr_fd in fd_list:
            fd_lhs_list, fd_rhs_list = curr_fd

            base_query = f'select * from {task} where '
            base_query += ' and '.join([f'{fd_lhs} = [{fd_lhs}]' for fd_lhs in fd_lhs_list])
            print("Base query: ", base_query)

            # get row info to generate queries
            for row_id, row in df.iterrows():
                curr_query = base_query
                for fd_lhs in fd_lhs_list:
                    value = row[fd_lhs]
                    curr_query = curr_query.replace(f'[{fd_lhs}]', f'"{value}"') 
                
                generated_query_list[hops][f'{fd_rhs_list[0]}'].append(curr_query)

    
    total_query_list = []
    query_idx = 0
    for hops, generated_query_list in generated_query_list.items():
        for fd_rhs, generated_query_list in generated_query_list.items():
            for query in generated_query_list:

                # save queries with metadata
                query_dict = {}
                query_dict['idx'] = query_idx
                query_dict['op'] = fd_rhs

                # task-specific for conditional hallucination analysis (i.e., middle answers)
                if task == 'olympic_joined':
                    if hops == '2hop':
                        gold_query = query.replace('select *', f'select {fd_rhs}, country, start, end')
                    elif hops == '3hop':
                        gold_query = query.replace('select *', f'select {fd_rhs}, country, city, start, end')
                else: 
                    gold_query = query.replace('select *', f'select {fd_rhs}, start, end')
                    
                gold_answer_rows = pd.read_sql(gold_query, conn)
                gold_answer_list = gold_answer_rows.to_dict(orient='records')
                incorrect_answer_list = []
                
            
                query_dict['qtype'] = qtype + '_' + hops
                query_dict['sql_for_db'] = gold_query
                query_dict['correct_rows'] = gold_answer_list
                query_dict['incorrect_rows'] = incorrect_answer_list

                query_idx += 1
                total_query_list.append(query_dict)


            print(f"Included {qtype} operation, total {query_idx+1} queries generated.")
            print("Sample query: ", gold_query)
            print("\tGold answer: ", gold_answer_list)
            print("\n\n")


    # save queries to json file
    with open(save_path, "a", encoding="utf-8") as f:
        json.dump(total_query_list, f, indent=4)
    print(f"Total {query_idx} queries saved to {save_path}.")

    conn.close()


