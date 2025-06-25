'''
Run QA task for given model and task, and save LLM responses to log file.
You should have generated QA pairs with src/qa_pairs/write_sql.py and src/qa_pairs/convert_sql.py before running this script.
'''


import random
import argparse
import os
import time
import json
import signal

import src.utils.utils_llm as lutils
import src.utils.utils_dataset as dutils

TRANSLATOR = lutils.TRANSLATOR_MODEL # gpt-4o

def parse_args():
    '''
    Config
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices = dutils.DATASET_LIST, default='dyknow', help='name of the dataset')
    parser.add_argument("--df_idx", type=int, default=0, help="If the dataset has multiple dataframes (e.g., dyknow), choose idx of the target dataframe")  
    parser.add_argument('--qtype', "-q", type=str, \
                        choices=['now', 'join', 'basic'], default= 'now', help='type of QA evaluation')
    parser.add_argument("--start_idx", type=int, default = 0, help='number of dataset index to start the QA task. For example, if set to 10, starts QA with 10th entity.')
    parser.add_argument("--random_seed", "-s", type=int, default=1116)
    parser.add_argument("--no_context", action='store_true', default=False, help='control open-book QA and closed-book QA')
    parser.add_argument("--model", type=str, choices = lutils.MODEL_LIST, required=True, help='check utils/utils_llm.py for the supported models')
  
    args = parser.parse_args()
    
    return args



# timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out!")



# run qa for the temporal alignment task
def run_qa_now(log_file_path, qa_path, model, start_idx=0, binary_prompt=None):
    
    # set the task
    systype = 'temp_align'

    # Set the timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)

    # start QA
    with open(qa_path, 'r') as file:
        qa_data = [json.loads(line) for line in file] 
    
    for q_items in qa_data:

        # skip until given start_index
        if(start_idx > q_items['idx']):
            print(f"Skipping idx: {q_items['idx']}...")
            continue

        # get qa prompt for the row
        print(f"Running idx: {q_items['idx']}...")     

        # we have multiple questions
        for prompt_idx, question in enumerate(q_items['questions']):
            
            # set question
            id = str(q_items['idx']) + '-' + str(prompt_idx)    
            question = 'Q: ' + question + '\nA:'


            # call model APIs and save responses
            fail_cnt = 0
            while (fail_cnt < 5):
                try:
                    signal.alarm(15)

                    print(question)
                    response = lutils.run_llm(question, \
                                            tasktype=systype, \
                                            model=model, \
                                            new_prompt=binary_prompt)
                    print(response)

                    signal.alarm(0)
                    break

                except Exception as error:
                    fail_cnt +=1
                    print(f"{model} ERROR with idx: {id} (fail_cnt: {fail_cnt}) : {error}")
                    
                    if fail_cnt == 5:
                        print(f"Too many errors. Skipping idx: {id}")
                        response = "RESPONSE ERROR"

                except TimeoutException as e:
                    response = "RESPONSE ERROR"



            # this is to avoid API rate limit
            time.sleep(1)

            # save responses to log file
            qa_result_dict = {'idx': id, \
                            "qtype": q_items['qtype'], \
                            "sql_for_db": q_items['sql_for_db'], \
                            'question': question, \
                            'model_response': response, \
                            'correct_rows': q_items['correct_rows'],\
                            'incorrect_rows': q_items['incorrect_rows']}

            # save as jsonl file
            with open(log_file_path, 'a') as file:
                file.write(json.dumps(qa_result_dict) + '\n')



# run qa for the temporal reasoning task (single-hop, multi-hop)
# - single-hop (basic): additional context depends on no_context param
# - multi-hop (joined): default setting is w/o context
def run_qa_basic(log_file_path, qa_path, qtype, dataset, df_idx, no_context, model, start_idx = 0):
    # set the task
    systype = 'basic' if qtype == 'basic' else 'join' 


    if qtype == 'basic':
        if no_context:
            log_file_path = log_file_path.replace(f'{qtype}', f'{qtype}_no_context')
    else:
        no_context = True

    # Set the timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)   


    # start QA
    with open(qa_path, 'r') as file:
        qa_data = [json.loads(line) for line in file]
    
    for q_items in qa_data:

        # skip until given start_index
        if(start_idx > q_items['idx']):
            print(f"Skipping idx: {q_items['idx']}...")
            continue

        # get qa prompt for the row
        print(f"\n\nRunning idx: {q_items['idx']}...")     


        # we have multiple questions
        for prompt_idx, question in enumerate(q_items['questions']):

            # get context
            context_str = '' if no_context else dataset.get_context(q_items)
            
            # set question
            id = str(q_items['idx']) + '-' + str(prompt_idx)    
            question = context_str + '\n\n'+ 'Q: ' + question + '\nA:'


            # call model APIs and save responses
            fail_cnt = 0
            while (fail_cnt < 5):
                try:
                    signal.alarm(15)

                    print(question)
                    response = lutils.run_llm(question, tasktype=systype, model=model)
                    print(response)

                    signal.alarm(0)
                    break

                except Exception as error:
                    fail_cnt +=1
                    print(f"{model} ERROR with idx: {id} (fail_cnt: {fail_cnt}) : {error}")
                    
                    if fail_cnt == 5:
                        print(f"Too many errors. Skipping idx: {id}")
                        response = "RESPONSE ERROR"

                except TimeoutException as e:
                    response = "RESPONSE ERROR"



            # this is to avoid API rate limit
            time.sleep(1)

            # save responses to log file
            qa_result_dict = {'idx': id, \
                            "qtype": q_items['qtype'], \
                            "op": q_items['op'], \
                            "sql_for_db": q_items['sql_for_db'], \
                            'question': question, \
                            'model_response': response, \
                            'correct_rows': q_items['correct_rows'],\
                            'incorrect_rows': q_items['incorrect_rows']}

            # save as jsonl file
            with open(log_file_path, 'a') as file:
                file.write(json.dumps(qa_result_dict) + '\n')



def run_qa(save_dir, task, qtype, no_context, dataset, df_idx, model, start_idx = 0):
    '''
    Run QA task for given model and task, and save LLM responses to log file.

    Args:
    - mixed: float, the probability of mixed question (i.e. all true question)
    - demo: bool, whether to use pre-defined demos for few-shot setting
    - rag: bool, whether to run RAG setting
    - index: int, the entity index to start running QA task.
    '''

    # get log file path and dataset. The results will be saved in this file.
    log_file_path = lutils.get_savename(save_dir= save_dir, \
                                            task = task, \
                                            model= model, \
                                            tasktype = qtype, \
                                            endswith= '.jsonl')
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    if task == 'dyknow':
        log_file_path = log_file_path.replace('.jsonl', f'_{df_idx}.jsonl')
    print(f"Log file path: {log_file_path}\n")



    # get the QA pairs
    qa_path = f"./qa_pairs/{task}/{qtype}_sql_{TRANSLATOR}.jsonl"
    if task == 'dyknow': # contains multiple datasets
        qa_path = qa_path.replace("_sql", f"_{df_idx}_sql")


    # run qtype
    if qtype in ['basic', 'join']: 
        run_qa_basic(log_file_path, qa_path, qtype, dataset, df_idx, no_context, model, start_idx)

    elif qtype == 'now':
        # sysprompt for binary questions
        binary_prompt = dataset.get_binary_sysprompt() if dataset.binary else None
            
        run_qa_now(log_file_path, qa_path, model, start_idx, binary_prompt)
    
    else:
        raise NotImplementedError(f"qtype {qtype} is not implemented yet.")
    

    return



def main():
    '''
    Main function to run QA task
    '''

    # config
    print("================= Config ==========================")
    args = parse_args()
    task = args.task.strip()
    qtype = args.qtype
    start_idx = args.start_idx
    df_idx = args.df_idx
    random_seed = args.random_seed
    model = args.model
    no_context = args.no_context

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print("===================================================")
    save_dir = './results'

    # set random seed
    random.seed(random_seed)
    dutils.set_random_seed(random_seed)


    # get dataset
    dataset = dutils.get_dataset(task)


    # start QA
    print(f"Starting main QA...\n")
    run_qa(save_dir, task, qtype, no_context, dataset, df_idx, model, start_idx)

    return


if __name__ == "__main__":
    main()

