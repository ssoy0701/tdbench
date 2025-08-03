'''
Given SQL queries, this script will convert them to natural language questions & according answers to json file. You should have: 
- generated SQL queries via write_sql.py.
- updated src/utils/utils_llm.py to setup the translator LLM.
'''

import json
import argparse
import random
import os
import re
import time

import utils.utils_llm as lutils
import utils.utils_dataset as dutils


# argparser
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices = dutils.DATASET_LIST, default='dyknow')
parser.add_argument("--qtype", '-q', type=str, choices = ['now', 'basic', 'join'], default='basic')
parser.add_argument("--random_seed", type=int, default='1116')
parser.add_argument("--df_idx", type=int, default=0, help="If the dataset has multiple dataframes (e.g., dyknow), choose idx of the target dataframe")
parser.add_argument("--print", action="store_true", help="Print the generated questions.")
parser.add_argument("--do_sample", action="store_true", help="Sample questions from each category.")
parser.add_argument("--sample_num", type=int, default=150, help="Number of samples to generate from each category (i.e., None/Unique/Multiple).")
args = parser.parse_args()

RESULT_DIR = '../qa_pairs'
TRANSLATOR = lutils.TRANSLATOR_MODEL # modify this if you want to use other LLMs (e.g., gpt4, gemini, etc.). Make sure you have updated src/utils/utils_llm.py to setup the translator LLM.


# config
task = args.task
qtype = args.qtype
sample_num_config = args.sample_num
do_sample = args.do_sample
print_qa = args.print
dpath = os.path.join(RESULT_DIR, task, f'{qtype}_sql.json')
if task == 'dyknow': # multiple dataframes
    dpath = dpath.replace("_sql", f"_{args.df_idx}_sql")
spath = dpath.replace('.json', f'_{TRANSLATOR}.jsonl')
random.seed(args.random_seed)



# read data
with open(dpath, "r", encoding="utf-8") as f:
    target_json = json.load(f)
    print("Total number of queries: ", len(target_json))


# system message for database description, used for the translator LLM
if task == 'dyknow':
    if args.df_idx == 0: # Country
        system_msg = "The following SQL query is about a relational database Leader(country, role, name, start, end) where start, end are date information."

    elif args.df_idx == 1: # Sport
        system_msg = "The following SQL query is about a relational database Sport(name, team, start, end) where start, end are date information, the exact dates of the athlete's contract."
    
    elif args.df_idx == 2: # Organization
        system_msg = "The following SQL query is about a relational database Organization(organization_name, organization_type, role, person_name, start, end) where start, end are date information."
    else:
        raise ValueError("df_idx is strange")
    
elif task == 'legal':
    system_msg = "The following SQL query is about a relational database SameSexLaw(country, law_type, legality, start, end) where start, end are date information and legality is either 'legal' or 'illegal'."

elif task == 'environ':
    system_msg = "The following SQL query is about a relational database CarbonMechanism(jurisdiction, type, name, status, start, end) where start, end are date information and status is either 'yes (exist)' or 'no (does not exist)'."

elif task == 'culture':
    system_msg = "The following SQL query is about a relational database CulturalHeritage(member_state, heritage_element, status, start, end) where start, end are date information and status is either 'proclaimed' or 'inscribed'."

elif task == 'movie':
    system_msg = "The following SQL query is about a relational database Movie(title, director, cast, release_year, start, end) where start (release date), end are date information. Here, focus on ORDER BY clause."

elif task == 'olympic_joined':
    system_msg = "The following SQL query is about a relational database OlympicGames(Game_edition, Game_name, Country, City, Name, Role) where 'Name' and 'Role' are about leaders of the **host country**."

elif task == 'medical':
    system_msg = "The following SQL query is about a relational database SyntheticPatients(name, gender, blood_type, doctor, medication, start, end) where start, end are date of admission and discharge."

else: 
    raise NotImplementedError

system_msg += "\n\nTranslate the provided SQL query to a natural language question. Do not include any artificial phrases like 'according to the database', 'from the table', or 'based on the query'. Also, do not describe the FROM clause or mention the table name. Focus only on the selected fields and filtering conditions. Generate 3 different questions each starting with 'Q: '. Only return the generated questions."




if qtype == 'now' or qtype == 'join':
    all_q = {f'{qtype}': target_json}

    # print given SQL info
    print("=====================================")
    print("Task: ", task)
    print("qtype: ", qtype)
    print("Total questions: ", len(target_json))
    print("=====================================")

    if not do_sample:
        sample_num_config = len(target_json)
        print(f"We will use all {sample_num_config} questions.")
        print("=====================================")
    else:
        print(f"We will sample {sample_num_config} questions.")
        print("=====================================")


elif qtype == 'basic':
    # categorize w.r.t. the number of answers
    none_of_above = [v for v in target_json if v['qtype'] == 'basic_none']
    unique = [v for v in target_json if v['qtype'] == 'basic_unique']
    multiple = [v for v in target_json if v['qtype'] == 'basic_multiple']
    all_q = {'none_of_above': none_of_above, 'unique': unique, 'multiple': multiple}

        
    # print given SQL info
    print("=====================================")
    print("Task: ", task)
    print("qtype: ", qtype)
    print("Total questions: ", len(target_json))
    print("\tNone of above: ", len(none_of_above))
    print("\tUnique: ", len(unique))
    print("\tMultiple: ", len(multiple))
    print("=====================================")

    # we always sample for basic questions
    print(f"We will sample {sample_num_config} questions from each category.")
    print("=====================================")




## Run GPT-4o to translate SQL queries into natural language questions
new_idx = 0

for q_type, q_items in all_q.items(): 
    
    print(f"Processing Query with {q_type} answers")
    print("=====================================")
 

    # sample questions from each category
    sample_num = min(sample_num_config, len(q_items))
    sampled = random.sample(q_items, sample_num)


    for q in sampled:
        print("processing: ", q['idx'])

        # generate prompt for SQL conversion
        prompt = q['sql_for_db']
    

        if task == 'olympic_joined': # multi-hop specific
            prompt_nlq = prompt.replace('country, city, start, end', '').replace('country, start, end', '').lower()
        else: 
            prompt_nlq = prompt.replace(', start, end', '').replace('start', 'start_date').replace('end', 'end_date')


        if qtype == 'now':
            if task == 'movie': # dataset-specific
                prompt_nlq = prompt.replace('order by', 'ORDER BY').replace(', start, end', '')
            else:
                prompt_nlq = prompt.replace('IS NULL', 'IS now()').replace(', start, end', '')
        
        print(prompt_nlq)
        print(q['correct_rows'])
        
        # generate question
        nlq = lutils.run_llm(prompt_nlq, 'convert_sql', TRANSLATOR, temp=0.3, new_prompt=system_msg)
        questions = re.findall(r"Q: (.*?)(?=\nQ:|\n*$)", nlq, re.DOTALL)
        q['questions'] = questions
        print('questions: ', len(questions))

        if print_qa:
            print("SQL: ", prompt)
            for i, question in enumerate(questions):
                print(f"Q{i+1}: ", question)
                time.sleep(2)

        # give new id
        q['idx'] = new_idx  
  
        # write to file
        with open(spath, "a", encoding="utf-8") as f:
            json.dump(q, f, ensure_ascii=False)
            f.write('\n') 

        new_idx += 1
        

