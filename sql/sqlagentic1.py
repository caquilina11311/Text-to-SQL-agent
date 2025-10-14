import difflib
from multiprocessing import process
from sqlalchemy import create_engine, inspect
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import requests
import json
import psycopg2
import re


#cli = openai.OpenAI()
st = SentenceTransformer('all-MiniLM-L6-v2')
schema = {}
te = {}

def decode_response(response):
    output = []
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if "response" in data:
                output.append(data["response"].lstrip("<s>"))
            if data.get("done",False):
                break
    return "".join(output).lstrip("<s>")

def tab_to_str(table):
    info = schema[table]
    fkstr = ""
    for key in info['Foreign_keys']:
        fkstr= fkstr + (f"{key['constrained_columns']} -> {key['referred_table']}.{key['referred_columns']}, ")
    return f"Table: {table},  Columns: {', '.join(info['Columns'])} Primary Keys: {', '.join(info['Primary_keys'])} Foreign Keys: {fkstr}"

def find_relevant(query, num_similar):
    eq = st.encode(query)
    most_similar = sorted(
        te.keys(),
        key=lambda t: np.dot(eq,te[t]),
        reverse=True
    )
    return most_similar[:num_similar]


def find_sim(query):
    relevant_tables = find_relevant(query, 5)
    relevant_to_string = ''.join([f"Table: {table}, {tab_to_str(schema[table])}\n" for table in relevant_tables])
    prompt_text = f"""
    Given the following schema and query, do the following 
    1. generate a SQL response based off of this query
    
    Question: {query}

    Relevant schema:\n {relevant_to_string}
    
    """

    return cli.responses.create(
       model='gpt-4.1-mini',
       input=prompt_text
    )

def generate_SQLcoder_response(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "sqlcoder", "prompt": prompt, "num_predict": 256},
        stream=True
    )
    return decode_response(response)

def generate_Llama_response(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama2", "prompt": prompt, "num_predict": 256},
        stream=True
    )
    return decode_response(response)

def generate_response(query, relevant_tables):
    relevant_to_string = ''.join([f"Table: {table}, {tab_to_str(table)}\n" for table in relevant_tables])
    prompt_text = f"""
    INSTRUCTION: generate a PostgreSQL query based on the following query using only the tables,columns,foreign and primary keys
    provided
    Schema: {relevant_to_string}\n
    Rules: 
    -user PostgreSQL functions only
    -only use existing tables,columns,foreign keys, and primary keys in the schema
    -return only a valid PostgreSQL query, no explanations
    -do not invent any columns or tables
    -use table aliases consistently 
    -always prefix every column with its respective alias, if it has one
    -make sure each column requested in the query appears

    Query: {query}
    """
    return generate_SQLcoder_response(prompt_text)

def fix_spelling_prompt(incorrect_col,columns):
    match = difflib.get_close_matches(incorrect_col, columns, n=1, cutoff=0.6)
    if match:
        return match[0]
    return incorrect_col
    

def check_existing(query, SQLentry, relevant_schema):
    incorrect_columns = []
    columntable= re.findall(r'(\b\w+\b)\.(\b\w+\b)',SQLentry)
    ali = re.findall(r'\b(?:FROM|JOIN)\s+([a-zA-Z_][\w]*)\s*(?:([a-zA-Z_][\w]*))?', SQLentry, re.IGNORECASE)
    alias_table = {}
    keywords = {"ON", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "GROUP", "ORDER", "BY", "HAVING", "LIMIT"}
    for c, a in ali:
        if(c in keywords):
            alias_table[a] = None
        else:
            alias_table[a] = c


    for a,c in columntable:
            table = alias_table.get(a,a)
            if table in schema and not c in schema[table]['Columns']:
                match = difflib.get_close_matches(c, schema[table]['Columns'], n=1, cutoff=0.6)
                if match:
                   SQLentry = re.sub(rf'\b{a}\.{c}\b', f'{a}.{(match[0])}', SQLentry)     
                else:
                    for tab in relevant_schema:
                        match = difflib.get_close_matches(c, schema[tab]['Columns'], n=1, cutoff=0.6)
                        if match:
                            SQLentry = re.sub(rf'\b{a}\.{c}\b', f'{tab}.{(match[0])}', SQLentry)   
                
    return SQLentry
def invalid_SQL_prompt(query,SQLentry,relevant_schema,corrections):
    pass
def validation(query, SQLentry, relevant_schema):
    SQLentry= check_existing(query, SQLentry, relevant_schema)
    print("FIXED SPELLING")
    print(SQLentry)

    prompt = f"""
    Given the relevant Schema, the SQL response generated and the original query 
    task:
    -Check for missing, inconsistent, or incorrect use table aliases
    -Check for general structual issues 
    -Check to see if the SQL response is referencing existing columns based on the Schema provided
    -Check to see if the SQL response answers the original query 

    rules:
    -only output one one, CORRECT or INCORRECT followed by a sentence of what is wrong
    -do not explain your reasoning or the checks
   
    relevant Schema:\n{''.join([f"Table: {table}, {tab_to_str(table)}\n" for table in relevant_schema])}
    SQL response:\n {SQLentry}
    Original Query:\n {query}
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json = {"model": "llama2", "prompt": prompt },
        stream = True
    )
    dcres = decode_response(response)
    print(dcres)
    if("INCORRECT" in dcres):
        pass
    else:
        return SQLentry
    

def setup_database():
    # choices = {"chinook","northwind"}
    # print("The following databases are available to use:")
    # print("chinook")
    # print("northwind")
    # option = input("Enter the name of the database you want to use:" )
    # print(f"You have chosen {option}")
    e =create_engine(f"postgresql+psycopg2://christopher:Clone1131!@localhost:5432/northwind")
    i = inspect(e)
    for name in i.get_table_names():
        schema[name] = {
        "Columns":[col["name"] for col in i.get_columns(name)],
        "Foreign_keys": i.get_foreign_keys(name),
        "Primary_keys": i.get_pk_constraint(name)
    }
    for table in schema.keys():
        te[table] = st.encode(tab_to_str(table), normalize_embeddings=True)
    
def test_sql(sql_query):
    con = psycopg2.connect(
        host = "localhost",
        port = 5432,
        database = "northwind",
        user = "christopher",
        password = "Clone1131!"
    )
    cursor = con.cursor()
    cursor.execute(sql_query)
    # try:
    #     cursor.execute(sql_query)
    # except Exception as e:
    #     print("wrong")
    #     con.close()
    #     return False
    con.close()
    return True


setup_database()
query = "List all orders made by the customer 'Alfreds Futterkiste', including the OrderID, OrderDate, ShipCountry, and the total amount for each order."
relevant_schema = find_relevant(query, 5)
relevant_to_string = ''.join([f"Table: {table}, {tab_to_str(table)}\n" for table in relevant_schema])
SQL_response = generate_response(query, relevant_schema)
print(SQL_response)
test_sql(validation(query, SQL_response, relevant_schema))



#find_sim("List all tracks by AC/DC")
