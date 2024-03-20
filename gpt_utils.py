from prompt_list import *
import json
import time
import openai
import re
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

import configparser
import os
import sys

import openai

def extract_and_extend_entities(question, openai_api_key):
    """
    Uses GPT to extract up to 5 entities directly related to the question and extends
    to 8 entities, assigning relevance scores to each.
    
    Args:
        question (str): The input question.
        openai_api_key (str): API key for OpenAI.
    
    Returns:
        dict: A dictionary of entities and their relevance scores.
    """
    openai.api_key = openai_api_key
    prompt = f"""
    For the question: '{question}', first, identify 1 to 5 single-word keywords directly from to the question. 
    These should be core concepts or entities central to the question's topic. 
    Then, extend this list to a total of 8 single-word keywords by including related concepts. 
    Assign a relevance score to each keyword so that the sum of all scores is 1. 
    The keywords that are extracted directly from the question should have higher relevance scores.
    Return the keywords and their relevance scores in the Standard JSON format that can be converted to Python dictionary.
    The JSON format should look like this:
    {{
        "Keyword1": 0.2,
        "Keyword2": 0.3,
        "Keyword3": 0.5
    }}
    """

    messages = [
            {"role": "system", "content": "Your task is to use GPT to extract up to 5 entities directly related to the question and extends to 8 entities, assigning relevance scores to each."},
            {"role": "user", "content": prompt},
        ]
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            max_tokens=100
        )
        # Extracting entities and scores from the response
        entities_with_scores_dict = response.choices[0].message.content
        return entities_with_scores_dict

    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return {}


def extract_entities_from_question(question, openai_api_key):
    """
    Extracts a minimum of one and a maximum of 3 keywords that are most relevant to the question.

    Args:
        question (str): The input question from which to extract keywords.
        openai_api_key (str): The API key for OpenAI GPT.

    Returns:
        List[str]: A list of extracted keywords/entities.
    """
    # Set up the OpenAI API key
    openai.api_key = openai_api_key

    # Constructing the prompt with a CoT approach
    system_message = "You are an AI assistant that helps people find the most relevant keywords or entities related to their questions."
    prompt = f"""You are an AI that analyzes and understands questions deeply. Given the question below, think step-by-step to identify the most relevant keywords (up to 3) that capture the essence of the question. These keywords should help in further understanding or researching the topic of the question.

    Question: "{question}"

    Think carefully and list the keywords, separated by semicolons:
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )
        
        # Assuming the AI's response will be in the format "Keyword1; Keyword2; Keyword3"
        result_text = response.choices[0].message.content.strip()
        entity_list = [entity.strip() for entity in result_text.split(';') if entity.strip()]

        return entity_list

    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        # Return an empty list in case of an error to ensure the function returns a meaningful result
        return []

def path_pruning_with_gpt(question_raw, paths, openai_api_key):
    '''
    Score and prune the list of relation chains based on their relevance to the input question.
    Args:
        question_raw (str): The input question.
        all_relations (List[EntityRelationPair]): List of all entity-relation-role pairs.
        openai_api_key (str): API key for OpenAI GPT.
    
    Returns:
        List[EntityRelationPair]: The top-3 most relevant relations.   
    '''
    # Construct the prompt to send to GPT
    prompt = f"Given the question: '{question_raw}', evaluate the following reasoning paths and select the most relevant ones. Only keep paths that are highly relevant to answering the question.\n\n"
    for idx, path in enumerate(paths, start=1):
        # Construct a readable representation of each path
        path_description = ' -> '.join([f"({s}, {p}, {o})" for s, p, o in path.path])
        prompt += f"Path {idx}: {path_description}\n"
    
    prompt += "\nReturn the IDs of the most relevant paths, separated by commas. Do not include any explanations or additional text."

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that helps people find information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=256
        )
        result = response.choices[0].message.content
        print(result)
    except Exception as e:
        print(f"openai error: {e}")
    

    return result

import openai

def prune_entities_with_gpt(question, entity_list, openai_api_key):
    """
    Uses GPT to reason about the most relevant entities to a question from a given entity list
    and returns the indices of the top 3 entities.
    
    Args:
        question (str): The input question.
        entity_list (List[str]): A list of entities.
        openai_api_key (str): API key for OpenAI.
    
    Returns:
        List[int]: Indices of the top 3 entities in the original list.
    """
    openai.api_key = openai_api_key

    prompt = f"Given the question: '{question}', I will list several entities. Based on their relevance to answering the question, respond only with the indices of the top 3 most relevant entities, separated by commas, in a single line. Do not include any explanations or additional text.\n\n"

    for index, entity in enumerate(entity_list, start=1):
        prompt += f"{index}. {entity}\n"

    prompt += "\nIndices of the top 3 most relevant entities:"

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that retrieve the indices of entities that are most relevant to answering the question."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=256
        )
        result = response.choices[0].message.content
        print(result)
        
        # Filter out any unexpected whitespace or empty strings
        top_entities_indices = [int(index.strip()) - 1 for index in result if index.strip().isdigit()]
        
        return top_entities_indices  # Ensure only the top 3 entities are returned

    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return []


    

def import_config(config_file='config.ini'):
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['openai_key'] = content['openai']['key']
    return config

def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations


def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    if "llama" in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"  # your local llama server port
        engine = openai.Model.list()["data"][0]["id"]
    else:
        openai.api_key = opeani_api_keys

    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    try:
        response = openai.chat.completions.create(
                model=engine,
                messages = messages,
                temperature=temperature,
                max_tokens=max_tokens)
        result = response.choices[0].message.content
    except Exception as e:
        print("openai error: {e}")
    return result

    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    

def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name):
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    with open("ToG_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")

    
def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    

def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False


def generate_without_explored_paths(question, args):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return response

def generate_without_explored_paths_neo(question, key):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, 0.0, 256, key, 'gpt-3-turbo')
    return response

# def extract_entities_from_question(question, key):
#     prompt = extract_key_words_prompt + question + "\nA:"
#     response = run_llm(prompt, 0.0, 256, key, 'gpt-3-turbo')
#     entity_list = response.split("; ")
#     return entity_list

