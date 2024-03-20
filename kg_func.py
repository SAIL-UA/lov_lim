from prompt_list import *
import json
import openai
import re
import time
from gpt_utils import *
from rdflib import Literal

from collections import namedtuple

import csv
from collections import defaultdict

EntityRelationPair = namedtuple('EntityRelationPair', ['subject', 'predicate', 'object', 'role'])

def all_entities(kg):
    query = """
    SELECT DISTINCT ?entity WHERE {
    { ?entity ?p ?o . }
    UNION
    { ?s ?p ?entity . }
    }
    """
    results = kg.query(query)
    # Process the results to extract entities
    entities = set()  # Use a set to avoid duplicates
    for result in results:
        entities.add(str(result[0]))

    # Convert the set to a list if you need list-specific operations
    entity_list = list(entities)
    return entity_list

def entity_as_subject(search_entity, kg):
    sparql_s_p = """
    SELECT DISTINCT ?subject ?predicate ?object
    WHERE {
    ?subject ?predicate ?object .
    FILTER (STR(?subject) = ?searchTerm)
    }
    """
    results = kg.query(sparql_s_p, initBindings={'searchTerm': Literal(search_entity)})
    return results

def entity_as_object(search_entity, kg):
    sparql_o_p = """
    SELECT DISTINCT ?subject ?predicate ?object
    WHERE {
    ?subject ?predicate ?object .
    FILTER (STR(?object) = ?searchTerm)
    }
    """
    results = kg.query(sparql_o_p, initBindings={'searchTerm': Literal(search_entity)})
    return results

def query_relations(entity, kg):
    '''
    Query the KG for all relations where the given entity is either a subject or an object.
    Returns a list of tuples, where each tuple contains a relation and the role of the entity ('subject' or 'object').
    
    Args:
        entity (str): The URI of the entity as a string.
        kg: The Knowledge Graph (an rdflib.Graph object).
        
    Returns:
        List[Tuple[str, str]]: A list of (relation, role) tuples.
    
    '''
    relations = []
    # relation_subject = entity_as_subject(entity, kg)
    relation_object = entity_as_object(entity, kg)
    # for subj, pred, obj in relation_subject:
    #     relations.append(EntityRelationPair(str(subj), str(pred), str(obj), 'subject'))
    for subj, pred, obj in relation_object:
        relations.append(EntityRelationPair(str(subj), str(pred), str(obj), 'object'))
    return relations

def load_entities_from_csv(csv_file_path):
    """
    Load entities from a CSV file.
    
    Args:
        csv_file_path (str): The path to the CSV file containing the entities.
    
    Returns:
        List[str]: A list of entities.
    """
    entities = []
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            entities.append(row[0])  # Assuming each row has an entity in the first column
    return entities

def select_top_entities_with_scores(entity_list, entity_scores_dict, top_n=3):
    """
    Selects the top 3 entities based on their relevance scores and returns all matched entities' scores.
    
    Args:
        entity_list (List[str]): A list of entities loaded from the CSV file.
        entity_scores_dict (dict): A dictionary of entities and their relevance scores.
    
    Returns:
        Tuple[Dict[str, float], List[str]]: A tuple containing a dictionary of all matched entities and their scores,
                                            and a list of the top 3 entities based on their scores.
    """
    # Identifying keywords and searching in the entity list
    keywords = [key for key, score in entity_scores_dict.items() if score > 0.1]  # Assuming scores are positive
    matched_entities = {}
    
    for entity in entity_list:
        entity_score = sum([score for keyword, score in entity_scores_dict.items() if keyword.lower() in entity.lower()])
        if entity_score > 0:  # If the entity matches any keyword
            matched_entities[entity] = entity_score

    # Sorting entities based on scores to find the top 3
    sorted_entities = sorted(matched_entities.items(), key=lambda item: item[1], reverse=True)
    top_entities = [entity for entity, score in sorted_entities[:top_n]]

    # Returning both the dictionary of matched entities with scores and the top 3 entities
    return (matched_entities, top_entities)


def score_and_select_entities(entities, keywords_scores):
    """
    Score entities based on the presence of keywords and select the top 5.
    
    Args:
        entities (List[str]): A list of entities to be scored.
        keywords_scores (dict): A dictionary of keywords and their relevance scores.
    
    Returns:
        List[str]: A list of the top 5 entities based on their scores.
    """
    entity_scores = defaultdict(float)
    
    for entity in entities:
        for keyword, score in keywords_scores.items():
            if keyword.lower() in entity.lower():
                entity_scores[entity] += score

    # Sort entities based on scores and select the top 5
    top_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return [entity for entity, _ in top_entities]

    
# Example usage
if __name__ == "__main__":
    entity_list = ["diabetes research", "insulin levels", "glucose metabolism", "cause of diabetes", "treatment options"]
    entity_scores_dict = {"diabetes": 0.4, "insulin": 0.3, "glucose": 0.2, "research": 0.1}  # Example scores
    
    matched_entities_scores, top_entities = select_top_entities_with_scores(entity_list, entity_scores_dict)
    print("All matched entities and their scores:", matched_entities_scores)
    print("Top 3 entities based on scores:", top_entities)