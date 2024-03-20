from gpt_utils import path_pruning_with_gpt, import_config, extract_entities_from_question, \
    extract_and_extend_entities, prune_entities_with_gpt
from kg_func import query_relations, score_and_select_entities, load_entities_from_csv, \
    select_top_entities_with_scores, all_entities
import rdflib
from collections import namedtuple
import json

# Load entities for entity extraction
# csv_file = "entities.csv"
# lov_entities = load_entities_from_csv(csv_file)

# Load the Knowledge Graph from the .nq file
kg = rdflib.ConjunctiveGraph()
try:
    kg.parse("lov_valid.nq", format="nquads")
except Exception as e:
    print(f"Error loading .nq file: {e}")

config = import_config("config.ini")
openai_api_key = config['openai_key']

# Get all entities from the KG
lov_entities = all_entities(kg)
print(f"Total entities: {len(lov_entities)}")

while True:
    try:
        question_raw = None
        try:
            # Prompt the user to enter a number
            # What is the main cause of diabetes?
            question_raw = input("Please input the question: ")
            # question_raw = "What is the main cause of diabetes?"
            if question_raw == "exit":
                break
            print(f"You entered the question {question_raw}.")
        except ValueError:
            # If there's an error during conversion, inform the user
            print("Invalid symbol in your input question!")

        # Extract topic entities and extend the related entities to 8
        topic_entity_dict_json = extract_and_extend_entities(question_raw, openai_api_key)
        # Convert JSON to Python dictionary
        topic_entity_dict = json.loads(topic_entity_dict_json)
        print(f"Entities extracted from question: {topic_entity_dict_json}")

        # Filter the most relevant entities, which has the score > 1. Use these entity to filter the entities for 
        # future score calculation
        # NO GPT involved in this step
        matched_entities, top_entities = select_top_entities_with_scores(lov_entities, topic_entity_dict, 20)
        # print(f"Pre-selected entities: {matched_entities}")

        # Entity pruning in GPT
        # Prune the entities based on the question
        pruned_top_entities_indices = prune_entities_with_gpt(question_raw, top_entities, openai_api_key)
        # print(f"Pruned entities: {pruned_top_entities_indices}")

        # Get the entities using the returned indices
        entity_for_relation_query = [top_entities[i] for i in pruned_top_entities_indices]

        # namedtupe for reasoning paths, ([], e, 0): [] is list for paths, e is the last entity for next query, 0 is the depth
        EntityRelationPath = namedtuple('EntityRelationPath', ['path', 'last_entity', 'last_entity_role','depth'])
        # Initialize reasoning paths with the pruned entities
        reasoning_paths = [EntityRelationPath(path=[], last_entity=e, last_entity_role=-1, depth=0) for e in entity_for_relation_query]

        # Phase 1 results output
        print(f"-------The entities most relevant to the question are:-----------")
        for i, e in enumerate(entity_for_relation_query):
            print(f"Entity {i}: {e}\n")
        
    except ValueError as e:
        print(f"Error: {e}")
        continue





