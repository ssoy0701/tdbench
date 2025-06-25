'''
Given a QA log file, this script will analyze the result and save it as a txt file & csv file. We verify both the final answer and the time reference in th LLM responses.
'''

import argparse
import random
import os
import pandas as pd
import json
import unidecode
import re
import signal

import utils.utils_llm as lutils
import utils.utils_dataset as dutils


TIME_VERIFIER = lutils.TIME_VERIFIER_MODEL # deepseek

NUM_TO_MONTH = {1: 'january', 2: 'february', 3: 'march', 4: 'april', 5: 'may', 6: 'june', 7: 'july', 8: 'august', 9: 'september', 10: 'october', 11: 'november', 12: 'december'}



def parse_args():
    '''
    Config
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices = dutils.DATASET_LIST, required=True, help='name of the dataset')
    parser.add_argument("--df_idx", type=int, default=0, help="If the dataset has multiple dataframes (e.g., dyknow), choose idx of the target dataframe") 
    parser.add_argument('--qtype', "-q", type=str, \
                        choices=['now', 'join', 'basic'], default= 'now', help='type of QA evaluation')
    parser.add_argument("--random_seed", "-s", type=int, default=1116)
    parser.add_argument("--file", "-f", type=str, default=None, required=True, help='file path of the QA result file (.jsonl)')
    
    args = parser.parse_args()

    return args



# heuristics for entity resolution errors
def get_answer_cand(answer):
    cand = []

    answer = answer.lower().strip()
    cand.append(answer) # ideal case
    cand.append(unidecode.unidecode(answer)) # for unicode errors

    if ',' in answer:
        split_answer = answer.split(',')
        for splitted in split_answer:
            cand.append(splitted.strip())

    if answer == 'kostas stephanopoulos':
        cand.append('konstantinos stephanopoulos')
    if answer == 'shōwa':
        cand.append('hirohito')

    split_answer = answer.split(' ')
    if len(split_answer) == 2:
        pass

    elif len(split_answer) == 3: # probably names with middle name
        cand.append(split_answer[0] + ' ' + split_answer[2])
        cand.append(split_answer[0] + ' ' + split_answer[1])
        cand.append(split_answer[0])

    return cand

        





##### Temporal Alignment task ##### ==========================



### evaluation function
def analyze_result_now(model_log_path, granularity):

    # read log file
    df = pd.read_json(model_log_path, lines=True)

    # set result categories
    # - correct: answer/time both correct
    # - partial_correct: answer correct, but time is wrong
    # - outdated: answer is outdated, time is correct
    # - partial_outdated: answer is outdated, time is also wrong
    # - incorrect: other hallucinations
    # - unsure: model exhibits uncertainty
    results = ['total', 'correct', 'partial_correct', 'outdated', 'partial_outdated', 'incorrect',  'unsure']

    # setup result dict
    result_dict = {r: 0 for r in results}
    num_questions = len(df)
    result_dict['total'] = num_questions
    

    # iterate through the qa results
    for row_id, row in df.iterrows():

        # parse model answer 
        model_answer = str(row['model_response']).strip().lower()

        # get gold answer
        answer_dict = row['correct_rows'][0]
        answer_col_key =  [k for k in answer_dict.keys() if k not in ['Start', 'End']][0]
        gold_answer, gold_date = answer_dict[answer_col_key], answer_dict['Start'] # we analyze start for now queries (e.g., since xx..)
        

        # preprocess gold answer and time
        gold_answer = get_answer_cand(gold_answer)
        gold_year = str(pd.to_datetime(gold_date).year)
        if granularity == 'month':
            gold_month = NUM_TO_MONTH[pd.to_datetime(gold_date).month] 


        # make outdated answer list
        outdated_list = []
        if len(row['incorrect_rows']) >= 0:
            for outdated in row['incorrect_rows']:
                if granularity == 'month':
                    outdated_list.append([get_answer_cand(outdated[answer_col_key]), [str(pd.to_datetime(outdated['Start']).year), NUM_TO_MONTH[pd.to_datetime(outdated['Start']).month]]])
                
                else:
                    outdated_list.append([get_answer_cand(outdated[answer_col_key]), [str(pd.to_datetime(outdated['Start']).year)]])


        # analyze the result
        result = 'incorrect'

        if 'unsure' in model_answer:
            result = 'unsure'

        elif any(gold in model_answer for gold in gold_answer):

            if gold_year in model_answer:
                result = 'correct'

                if granularity == 'month':
                    if any(mon in model_answer for mon in NUM_TO_MONTH.values()) and gold_month not in model_answer:
                        result = 'partial_correct'

            else:
                result = 'partial_correct'

        else:
            if len(outdated_list) >= 0:
                for outdated in outdated_list:
                    if any(out in model_answer for out in outdated[0]):

                        if granularity == 'month':
                            if any(mon in model_answer for mon in NUM_TO_MONTH.values()) and outdated[1][1] not in model_answer:
                                result = 'partial_outdated'
                            else:
                                result = 'outdated'
                            break

                        else:
                            if outdated[1][0] in model_answer:
                                result = 'outdated'
                            else:
                                result = 'partial_outdated'
                            break


        # update result counts and df
        result_dict[result] += 1
        df.loc[row_id, 'analysis'] = result
        
    

    # add metadata 
    result_dict['percentage'] = {r: round(result_dict[r]/num_questions * 100, 2) for r in result_dict.keys()}


    # save result and df
    result_file_path = model_log_path.replace('.jsonl', '_result.txt')
    with open(result_file_path, 'w') as f:
        for k, v in result_dict.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

    result_csv_path = model_log_path.replace('.jsonl', '_result.csv')
    df.to_csv(result_csv_path, index = False)

    print(f"TXT result saved at {result_file_path}.")
    print(f"CSV result saved at {result_csv_path}.\n")

    return





##### Temporal Reasoning task #####

def get_criteria(relation):
    """
    Get criteria based on the relation
    :param relation: relation string
    :return: criteria list
    """

    # criteria dict
    criteria_dict = {
        'both': ['overlap', 'overlapped-by', 'equal', 'start', 'finish',  'during', 'contain'],
        'start': ['after', 'met-by', 'started-by'],
        'end': ['before', 'meet', 'finished-by']
    }

    for key, values in criteria_dict.items():
        if relation in values:
            return key
    raise ValueError(f"Unknown relation: {relation}")
    

# set timeout handler
class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out!")
signal.signal(signal.SIGALRM, timeout_handler)


# time verifying function
def verify_time_with_LLM(correct_rows, model_response, granularity, eval_criteria='both'):

    answer_col_key = [k for k in correct_rows[0].keys() if k not in ['Start', 'End']][0]  
            
    if eval_criteria == 'both':
        # create entity date
        entity_date_list = []
        for correct_row in correct_rows:


            answer_entity = correct_row[answer_col_key]
            if granularity == 'month':
                start_time = pd.to_datetime(correct_row['Start']).strftime('%Y')
                
                if correct_row['End'] is None:
                    end_time = 'present'
                else:
                    end_time = pd.to_datetime(correct_row['End']).strftime('%Y')
            else:
                start_time = pd.to_datetime(correct_row['Start']).strftime('%Y-%m')
                if correct_row['End'] is None:
                    end_time = 'present'
                else:                
                    end_time = pd.to_datetime(correct_row['End']).strftime('%Y-%m')       
                

            entity_date_list.append(f"For the entity `{answer_entity}`:\n**Start date:** {start_time}\n**End date:** {end_time}\n")

        entity_date = "\n".join(entity_date_list)

                        # create prompt
        prompt = f"""You are given a reference **start date** and **end date**. Check whether the response correctly includes both dates, even if they are expressed in a different but equivalent format (e.g., `26 Jan 2025`, `January 26, 2025`, `2025/01/26`, etc.).

- If **both** of the two dates is are correctly mentioned with the intended meaning (i.e., the start date is described as the start date, and the end date as the end date), respond with **"Yes"**.  
- If **one** of the two dates is correctly mentioned with the intended meaning, respond with **"Half"**.
- If **neither** date is correctly mentioned with the correct meaning, respond with **"No"**.  
Your answer must be one of: `Yes`, `Half`, or `No`. Be concise.

{entity_date}

**Response:**  
{model_response}

**Answer:"""


    else:
        # create entity date
        entity_date_list = []
        
        for correct_row in correct_rows:

            answer_entity = correct_row[answer_col_key]
            if granularity == 'month':
                start_time = pd.to_datetime(correct_row['Start']).strftime('%Y')
                if correct_row['End'] is None:
                    end_time = 'present'
                else:
                    end_time = pd.to_datetime(correct_row['End']).strftime('%Y') 
            else:
                start_time = pd.to_datetime(correct_row['Start']).strftime('%Y-%m')
                if correct_row['End'] is None:
                    end_time = 'present'
                else:
                    end_time = pd.to_datetime(correct_row['End']).strftime('%Y-%m')  
            
            date = start_time if eval_criteria == 'start' else end_time

            entity_date_list.append(f"For the entity `{answer_entity}`:\n**{eval_criteria.capitalize()} date:** {date}\n*")

        entity_date = "\n".join(entity_date_list)

        start_time = pd.to_datetime(correct_rows[0]['Start']).strftime('%Y-%m-%d')
        end_time = pd.to_datetime(correct_rows[0]['End']).strftime('%Y-%m-%d')
        answer_entity = correct_rows[0][answer_col_key]

        date = start_time if eval_criteria == 'start' else end_time

        prompt = f"""You are given a reference date, which is either a start date or an end date. Check whether the response correctly includes this specific date, even if it is expressed in a different but equivalent format (e.g., `26 Jan 2025`, `January 26, 2025`, `2025/01/26`, etc.).

- If the correct {eval_criteria} date is mentioned with the correct meaning, respond with **"Yes"**.  
- If the date is incorrect, missing, or referred to incorrectly (e.g., an end date mentioned as a start date), respond with **"No"**.  
Your answer must be one of: `Yes` or `No`. Be concise.

{entity_date}

**Response:**  
"{model_response}"

**Answer:"""

    # call model APIs and save responses
    fail_cnt = 0
    while (fail_cnt < 2):
    # call model
        try:
            signal.alarm(15)
            result = lutils.run_llm(prompt=prompt, \
                                    tasktype='no_system_msg', \
                                    model=TIME_VERIFIER)
            print('result:', result)
            signal.alarm(0)
            break

        except Exception as error:
            fail_cnt +=1
            
            if fail_cnt == 2:
                print(f"Too many errors. Skipping.")
                result = "RESPONSE ERROR"

        except TimeoutException as error:
            print(f"Timeout!")
            result = "RESPONSE ERROR"
    

    # check result
    last_line = result.split('\n')[-1].lower()
    if 'yes' in last_line:
        detected_time = 'consistent'
    elif 'half' in last_line:
        detected_time = 'partial_consistent'
    elif 'no' in last_line:
        detected_time = 'inconsistent'
    else:
        detected_time = 'ambiguous'

    return detected_time



def analyze_result_basic(model_log_path, granularity):


    # read log file
    df = pd.read_json(model_log_path, lines=True)

    # set result categories
    # - partial_correct: for multiple answers
    results_answer = ['correct', 'partial_correct', 'incorrect', 'unsure']
    results_time = ['consistent', 'partial_consistent', 'inconsistent', 'ambiguous']

    # setup result dict
    result_dict_answer = {r: 0 for r in results_answer}
    result_dict_time = {r: 0 for r in results_time}
    result_dict_all = {}
    for qtype in ['unique', 'none', 'multiple']:
        result_dict_all[qtype] = {'answer': result_dict_answer.copy(), \
                                    'time': result_dict_time.copy()}
    

    # iterate through the qa results
    for row_id, row in df.iterrows():

        # parse model answer 
        print(f"=== Row id: {row_id} ====")
        model_answer = str(row['model_response']).strip().lower()


        #### 1. Answer checking
        print("Verifying answers...")
        if 'unique' in row['qtype']:

            # get gold answer
            result_dict_answer = result_dict_all['unique']['answer']
            answer_col_key = [k for k in row['correct_rows'][0].keys() if k not in ['Start', 'End']][0]

            gold_answer = row['correct_rows'][0][answer_col_key]
            gold_answer = get_answer_cand(gold_answer)

            # preprocess incorrect answer
            inc_name_list = []
            for inc in row['incorrect_rows']:
                inc_name_list.extend(get_answer_cand(inc[answer_col_key]))   


            # analyze the result
            if 'unsure' in model_answer:
                result_answer = 'unsure'

            elif any(no_resp in model_answer for no_resp in ['no answer', 'no one', 'none']):
                result_answer = 'incorrect' # because there is answer

            # if there are any incorrect names, then it is incorrect
            elif (row['op'] != 'during' or ('between' not in row['question'] and row['op'] == 'during')) and any(inc_name in model_answer for inc_name in inc_name_list):
                result_answer = 'incorrect'

            # if there is only a correct answer
            elif any(gold in model_answer for gold in gold_answer):
                result_answer = 'correct'

            else: # there is no gold answer nor incorrect answers
                result_answer = 'incorrect'


        elif 'none' in row['qtype']:
            # get the result dict
            result_dict_answer = result_dict_all['none']['answer']

            # preprocess gold answer
            gold_answer = ['no answer', 'no one', 'none']

            # analyze the result
            if 'unsure' in model_answer:
                result_answer = 'unsure'

            elif any(gold in model_answer for gold in gold_answer):
                result_answer = 'correct'
            
            else:
                result_answer = 'incorrect'


        elif 'multiple' in row['qtype']:
            # get the result dict
            result_dict_answer = result_dict_all['multiple']['answer']
            answer_col_key = [k for k in row['correct_rows'][0].keys() if k not in ['Start', 'End']][0]


            # analyze the result
            if 'unsure' in model_answer:
                result_answer = 'unsure'

            elif any(no_resp in model_answer for no_resp in ['no answer', 'no one', 'none']):
                result_answer = 'incorrect' # because there is answer

            # if there are any incorrect names or correct names
            else: 
                result_answer_list = []

                # preprocess incorrect answer
                inc_name_list = []
                for inc in row['incorrect_rows']:
                    inc_name_list.append(get_answer_cand(inc[answer_col_key])) 

                # check any incorrect names
                for inc_name in inc_name_list:
                    if (row['op'] != 'during' or ('between' not in row['question'] and row['op'] == 'during')) and any(inc_n in model_answer for inc_n in inc_name):
                        result_answer_list.append('incorrect')
                
                # preprocess correct answer
                gold_name_list = []
                for correct_row in row['correct_rows']:
                    gold_name_list.append(get_answer_cand(correct_row[answer_col_key]))

                # check any correct names
                for gold_name in gold_name_list:
                    if any(gold_n in model_answer for gold_n in gold_name):
                        result_answer_list.append('correct')
                
                # check results
                if row['op'] != 'during' and 'incorrect' in result_answer_list: 
                    result_answer = 'incorrect'

                elif 'correct' in result_answer_list:
                    if len(row['correct_rows']) >=3:
                        # check if count of correct is greater than 3
                        if len([r for r in result_answer_list if r == 'correct']) >= 3:
                            result_answer = 'correct'
                        else:
                            result_answer = 'partial_correct'

                    else: 
                        if len([r for r in result_answer_list if r == 'correct']) == len(row['correct_rows']):
                            result_answer = 'correct'
                        else:
                            result_answer = 'partial_correct'

                else: # none of cor/incor answers are included
                    result_answer = 'incorrect'

        else:
            raise ValueError(f"Qtype {row['qtype']} is not implemented yet.")
                                
        
        ###### 2. Time accuracy checking
        print('Verifying time references...')
        # we don't check time for none type questions
        if 'none' in row['qtype']: # if answer is 'no answer'
            if any(no_resp in model_answer for no_resp in ['no answer', 'no one', 'none']):
                result_time = 'consistent'

            else:
                result_time = 'inconsistent'


        else: # unique or multiple
            result_time = verify_time_with_LLM(
                correct_rows= row['correct_rows'], \
                model_response= model_answer, \
                granularity= granularity, \
                eval_criteria=get_criteria(row['op']))


        # update result counts and df
        result_dict_answer[result_answer] += 1
        result_dict_time[result_time] += 1

        df.loc[row_id, 'analysis_answer'] = result_answer
        df.loc[row_id, 'analysis_time'] = result_time


    df['analysis_answer_time'] = df.apply( lambda row: 'correct' \
            if row['analysis_answer'] == 'correct' and row['analysis_time'] == 'consistent' else 'incorrect', \
                axis=1)
    

    # lets setup final dict
    final_dict = {}
    
    # analysis w.r.t answer cardinality
    for qtype in ['unique', 'none', 'multiple']: 
        qtype_df = df[df['qtype'] == 'basic' + '_' + qtype]
        num_questions = len(qtype_df)
        if num_questions == 0:
            continue

        
        correct_answer = len(qtype_df[qtype_df['analysis_answer'] == 'correct'])
        incorrect_answer = len(qtype_df[qtype_df['analysis_answer'] == 'incorrect'])
        ambiguous_answer = len(qtype_df[qtype_df['analysis_answer'] == 'ambiguous'])
        unsure_answer = len(qtype_df[qtype_df['analysis_answer'] == 'unsure'])
        
        consistent_time = len(qtype_df[qtype_df['analysis_time'] == 'consistent'])
        partial_time = len(qtype_df[qtype_df['analysis_time'] == 'partial_consistent'])
        inconsistent_time = len(qtype_df[qtype_df['analysis_time'] == 'inconsistent'])
        
        both_correct = len(qtype_df[qtype_df['analysis_answer_time'] == 'correct'])


        final_dict[qtype] = {
            'answer': {
                'total': num_questions ,
                'correct': correct_answer,
                'incorrect': incorrect_answer,
                'ambiguous': ambiguous_answer,
                'unsure': unsure_answer,
                'percentage': {
                    'correct': round(correct_answer / num_questions * 100, 1) if num_questions  else 0,
                    'incorrect': round(incorrect_answer / num_questions * 100, 1) if num_questions  else 0,
                    'ambiguous': round(ambiguous_answer / num_questions * 100, 1) if num_questions  else 0,
                    'unsure': round(unsure_answer / num_questions   * 100, 1) if num_questions    else 0,
                }
            },
            'time': {
                'total': num_questions,
                'consistent': consistent_time,
                'partial_consistent': partial_time,
                'inconsistent': inconsistent_time,
                'loose': consistent_time + (partial_time // 2),
                'percentage': {
                    'consistent': round(consistent_time / num_questions * 100, 1) if num_questions else 0,
                    'partial_consistent': round(partial_time / num_questions * 100, 1) if num_questions else 0,
                    'inconsistent': round(inconsistent_time / num_questions * 100, 1) if num_questions else 0,
                    'loose': round((consistent_time + (partial_time // 2)) / num_questions * 100, 1) if num_questions else 0
                }
            },
            'both': {
                'consistent_and_correct': both_correct,
                'percentage': {
                    'consistent_and_correct': round(both_correct / num_questions * 100, 1) if num_questions else 0
                }
            }
        }



    # analysis w.r.t. operations
    basic_op_lists = ['after', 'before', 'equal', 'contain', 'finish', 'start', 'meet', 'met-by', 'overlap', 'during', 'finished-by', 'started-by', 'overlapped-by']
    
    op_dict = {op: 0 for op in basic_op_lists}
    for op in basic_op_lists:
        
        op_df = df[df['op'].str.contains(op)]
        total_op_num = len(op_df)
        if total_op_num == 0:
            continue

        correct_percentage = round(len(op_df[op_df['analysis_answer'] == 'correct']) / total_op_num * 100, 1)
        consistent_percentage = round(len(op_df[op_df['analysis_time'] == 'consistent']) / \
                                                    (len(op_df[op_df['analysis_time'] == 'consistent']) + len(op_df[op_df['analysis_time'] == 'inconsistent'])) * 100, 1)
        correct_and_consistent_percentage = round(len(op_df[(op_df['analysis_answer_time'] == 'correct')]) / total_op_num * 100, 1)

        op_result = {'total': total_op_num, \
                     'correct': correct_percentage,\
                    'consistent': consistent_percentage,\
                    'correct_and_consistent': correct_and_consistent_percentage
                    }
        op_dict[op] = op_result

    final_dict['op'] = op_dict
    final_dict['total'] = {'total': len(df)}


    # dump with json
    result_file_path = model_log_path.replace('.jsonl', '_result.json')
    with open(result_file_path, 'w') as f:
        json.dump(final_dict, f, indent=4)

    result_csv_path = model_log_path.replace('.jsonl', '_result.csv')
    df.to_csv(result_csv_path, index = False)

    print(f"TXT result saved at {result_file_path}.")
    print(f"CSV result saved at {result_csv_path}.\n")
    
    return




##### Multi-hop QA #####

def analyze_result_join(task, model_log_path, granularity):


    # read original df
    dataset_df = dutils.get_dataset(task).df[0]


    # read qa result
    qa_result = []
    with open(model_log_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            qa_result.append(json.loads(line))


    # iterate through the qa results
    df = pd.read_json(model_log_path, lines=True)
    for row_id, row in df.iterrows():

        # parse model answer 
        print(f"=== Row id: {row_id} ====")
        model_answer = str(row['model_response']).strip().lower()

        # preprocess gold answer 
        answer_col = row['op']
        gold_answer_single = row['correct_rows'][0][answer_col].lower()
        gold_answer = get_answer_cand(gold_answer_single)

   
        # initialize all results
        result_middle = 'init'
        result_time = 'init'

        # analyze the result
        if 'unsure' in model_answer:
            result_answer = 'unsure'
            result_middle = 'none'
            result_time = 'none'

        elif any(no_resp in model_answer for no_resp in ['no answer', 'no one', 'none']):
            result_answer = 'incorrect' # because there is answer
            result_middle = 'none'
            result_time = 'none'

        # if there is only a correct answer
        elif any(gold in model_answer for gold in gold_answer):
            result_answer = 'correct'

        else: # there is no gold answer nor incorrect answers
            result_answer = 'incorrect'
        
        # now lets check middle answer, which is task-specific
        # (e.g., country for both 2-hop and 3-hop in olympic_joined)
        if result_middle == 'init': # no unsure, no "No answer"
            if task == 'olympic_joined':
                gold_middle = row['correct_rows'][0]['Country'].lower().strip()
                inc_middle = dataset_df[dataset_df['Country'].str.lower() != gold_middle]['Country'].unique().tolist()
            else:
                raise NotImplementedError("middle answer is not defined yet")

            # analyze middle
            if gold_middle in model_answer:
                result_middle = 'correct'
            elif any(inc in model_answer for inc in inc_middle):
                result_middle = 'incorrect'
            else:            
                result_middle = 'ambiguous'
       

        # now check time
        if result_time == 'init': # no unsure, no "No answer"
            

            gold_date_start = row['correct_rows'][0]['Start']
            gold_date_end = row['correct_rows'][0]['End']
            
            # year_hit
            gold_year_start = str(pd.to_datetime(gold_date_start).year)
            gold_year_end = str(pd.to_datetime(gold_date_end).year)

            year_hit = gold_year_start in model_answer or gold_year_end in model_answer
            result_time = 'consistent' if year_hit else 'inconsistent'

            # month hit
            if granularity == 'month':
                gold_month_start = NUM_TO_MONTH[pd.to_datetime(gold_date_start).month]
                gold_month_end = NUM_TO_MONTH[pd.to_datetime(gold_date_end).month]
                year_month_hit = (gold_month_start in model_answer and gold_year_start in model_answer) or (gold_month_end in model_answer and gold_year_end in model_answer)
                result_time = 'consistent' if year_month_hit else 'inconsistent'
        
        else: 
            year_hit = 'none'


        if row['qtype'] == 'join_3hop': # we need additional city info
            
            gold_middle_city = row['correct_rows'][0]['City']
            inc_middle_city = dataset_df[dataset_df['City'].str.lower() != gold_middle_city]['City'].unique().tolist()

            if gold_middle_city.lower() in model_answer:
                result_middle_3hop = 'correct'
            elif any(inc.lower() in model_answer for inc in inc_middle_city):
                result_middle_3hop = 'incorrect'
            else:
                result_middle_3hop = 'ambiguous'

        else:
            result_middle_3hop = 'none'



        df.loc[row_id, 'analysis_answer'] = result_answer
        df.loc[row_id, 'analysis_middle'] = result_middle 
        df.loc[row_id, 'analysis_middle_3hop'] = result_middle_3hop
        
        df.loc[row_id, 'analysis_time'] = result_time




    final_dict = {}

    for hops in ['2hop', '3hop']:

        curr_df = df[df['qtype'] == f'join_{hops}']
        
        if len(curr_df) == 0:
            continue

        final_dict[hops] = {
            
            'answer':{
                'total': len(curr_df),
                'correct': len(curr_df[curr_df['analysis_answer'] == 'correct']),
                'incorrect': len(curr_df[curr_df['analysis_answer'] == 'incorrect']), 
                'ambiguous': len(curr_df[curr_df['analysis_answer'] == 'ambiguous']),
                'unsure': len(curr_df[curr_df['analysis_answer'] == 'unsure']),
            }, 

            'time': {
                'total': len(curr_df),
                'consistent': len(curr_df[curr_df['analysis_time'] == 'consistent']),
                'inconsistent': len(curr_df[curr_df['analysis_time'] == 'inconsistent']), 
                'ambiguous': len(curr_df[curr_df['analysis_time'] == 'ambiguous']),
                'none': len(curr_df[curr_df['analysis_time'] == 'none']),
            },

            'middle_ans': {
                'total': len(curr_df),
                'correct': len(curr_df[curr_df['analysis_middle'] == 'correct']),
                'incorrect': len(curr_df[curr_df['analysis_middle'] == 'incorrect']),
                'ambiguous': len(curr_df[curr_df['analysis_middle'] == 'ambiguous']),
                'none': len(curr_df[curr_df['analysis_middle'] == 'none']),
            },
        }

        final_dict[hops]['metrics'] = {

            'final_answer_correct': round(final_dict[hops]['answer']['correct'] / final_dict[hops]['answer']['total'] * 100, 1),
            
            'final_answer_correct_and_consistent': round(len(curr_df[(curr_df['analysis_answer'] == 'correct') & (curr_df['analysis_time']!= 'inconsistent')]) / final_dict[hops]['answer']['total'] * 100, 1),
            
            'wrong_name_given_correct_country': round(len(curr_df[(curr_df['analysis_answer'] == 'incorrect') & (curr_df['analysis_middle'] == 'correct')]) / final_dict[hops]['middle_ans']['correct'] * 100, 1),
        }


        if hops == '3hop': # add hallucination rate with city too

            final_dict[hops]['middle_ans_3hop'] = {  
                'total': len(curr_df),
                'correct': len(curr_df[curr_df['analysis_middle_3hop'] == 'correct']),
                'incorrect': len(curr_df[curr_df['analysis_middle_3hop'] == 'incorrect']),
                'ambiguous': len(curr_df[curr_df['analysis_middle_3hop'] == 'ambiguous']),
                'none': len(curr_df[curr_df['analysis_middle_3hop'] == 'none']),
            }

            final_dict[hops]['metrics']['wrong_city_given_edition'] = \
                round(len(curr_df[(curr_df['analysis_middle_3hop'] == 'incorrect')]) / len(curr_df) * 100, 1)
            
                
            if final_dict[hops]['middle_ans_3hop']['correct'] > 0:
                final_dict[hops]['metrics']['wrong_country_given_correct_city'] = \
                    round(len(curr_df[(curr_df['analysis_middle'] == 'incorrect') & (curr_df['analysis_middle_3hop'] == 'correct')]) / (final_dict[hops]['middle_ans_3hop']['correct']) * 100, 1)
            
            else: 
                final_dict[hops]['metrics']['wrong_country_given_correct_city'] = 0.0

    print(final_dict)

    # dump with json
    result_file_path = model_log_path.replace('.jsonl', '_result.json')
    with open(result_file_path, 'w') as f:
        json.dump(final_dict, f, indent=4)

    result_csv_path = model_log_path.replace('.jsonl', '_result.csv')
    
    
    df.to_csv(result_csv_path, index = False)

    print(f"TXT result saved at {result_file_path}.")
    print(f"CSV result saved at {result_csv_path}.\n")
    
    return
    




def main():
    '''
    Main function to analyze QA result
    '''

    # config
    print("================= Config ==========================")
    args = parse_args()
    task = args.task.strip()
    qtype = args.qtype
    df_idx = args.df_idx
    random_seed = args.random_seed
    model_log_path = args.file
    granularity = dutils.get_dataset(task).granularity[df_idx]

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print(f'!! Evaluation Time Granularity: {granularity}...')
    print("===================================================")
    

    # set random seed
    random.seed(random_seed)

 
    # prepare log file
    print(f"Reading output jsonl file...\n")
    if not os.path.exists(model_log_path):
        raise ValueError(f"QA Log path does not exist: {model_log_path}")

    
    # analyze result and save
    print("Analyzing result...\n")
    if qtype == 'now':
        analyze_result_now(model_log_path, granularity)

    elif qtype == 'basic':
        analyze_result_basic(model_log_path, granularity)
    
    elif qtype == 'join':
        analyze_result_join(task, model_log_path, granularity)
    
    else:
        raise NotImplementedError(f"Task type {qtype} is not implemented yet.")


if __name__  =="__main__":
    main()