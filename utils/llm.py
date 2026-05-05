#from vllm import LLM
from pydantic import BaseModel, Field, root_validator
from typing import List, Optional, Literal
import json
import tiktoken
import time
import datetime
from enum import Enum
from tqdm import tqdm
import os
from google.genai import types
from google.genai.types import CreateBatchJobConfig
import fsspec
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from anthropic import Anthropic, transform_schema
from google.cloud import storage
import re
from vertexai.preview import tokenization

# 1. Innermost: findings within a specific sequence (report)
class SequenceFinding(BaseModel):
    sequence_id: str = Field(..., description="The identifier of the report, e.g., 'Sequence 1'")
    text: str = Field(..., description="The extracted text describing the disease in this report")

# 2. Middle: a single disease bundled with its findings across sequences
class DiseaseEntry(BaseModel):
    disease_name: str = Field(..., description="The name of the disease or finding")
    findings: List[SequenceFinding]

# 3. Outermost: full result (list of diseases)
class ExtractionResult(BaseModel):
    diseases: List[DiseaseEntry]

# Model for attribute extraction
class AttributeExtractionResult(BaseModel):
    presence: Optional[Literal["present", "absent"]] = Field(None, description="Strictly output either 'present' or 'absent'. Use 'absent' if the sentence explicitly states the entity is not there.")
    certainty: Optional[Literal["certain", "uncertain"]] = Field(None, description="Strictly output either 'certain' or 'uncertain'. Use 'uncertain' if the sentence is not clear about the presence of the entity.")
    location: Optional[str] = Field(None, description="The anatomical location of the entity")
    severity: Optional[str] = Field(None, description="The severity, grade, or size/dimensions of the entity")
    morphology: Optional[str] = Field(None, description="The shape, margin, or texture of the entity")
    tracking_information: Optional[str] = Field(None, description="Changes compared to prior exams or temporal status")
    error: bool = Field(..., description="Whether the sentence is not related to the target entity")

class RelationEnum(str, Enum):
    Cat = "Cat"
    Dx_Status = "Dx_Status"
    Dx_Certainty = "Dx_Certainty"
    Location = "Location"
    Associate = "Associate"
    Evidence = "Evidence"
    Morphology = "Morphology"
    Distribution = "Distribution"
    Measurement = "Measurement"
    Severity = "Severity"
    Comparison = "Comparison"
    Onset = "Onset"
    NoChange = "No Change"
    Improved = "Improved"
    Worsened = "Worsened"
    Placement = "Placement"
    PastHx = "Past Hx"
    OtherSource = "Other Source"
    AssessmentLimitations = "Assessment Limitations"

class EntityRelation(BaseModel):
    """Represents a single relation of an entity with its value"""
    relation: RelationEnum = Field(..., description="The relation name among the predefined types")
    value: str = Field(..., description="The value corresponding to the relation")
    obj_ent_idx: Optional[int] = Field(
        None,
        description="For Associate/Evidence relations, the ent_idx of the object entity"
    )

    @root_validator(skip_on_failure=True)
    def require_obj_ent_idx_for_certain_relations(cls, values):
        rel = values.get('relation')
        idx = values.get('obj_ent_idx')
        if rel in {RelationEnum.Associate, RelationEnum.Evidence} and idx is None:
            raise ValueError("'obj_ent_idx' must be provided for Associate and Evidence relations")
        return values

class Entity(BaseModel):
    """Represents a single entity with all its extracted relations"""
    name: str = Field(..., description="The name of the entity, exactly as provided")
    sent_idx: int = Field(..., description="Index of the sentence from which this entity was extracted")
    ent_idx: int = Field(..., description="Unique identifier for this entity within the report section")
    relations: List[EntityRelation] = Field(..., description="List of all relations for this entity")

    @root_validator(skip_on_failure=True)
    def validate_relations(cls, values):
        relations = values.get('relations', [])
        types = [rel.relation for rel in relations]
        # Must include exactly one Cat and one Status
        if types.count(RelationEnum.Cat) != 1 or types.count(RelationEnum.Dx_Status) != 1 or types.count(RelationEnum.Dx_Certainty) != 1:
            raise ValueError("Each entity must include exactly one 'Cat' relation and exactly one 'Status' relation and exactly one 'Dx_Certainty' relation")
        return values

class StructuredOutput(BaseModel):
    """Schema for structured extraction matching the provided output format"""
    entities: List[Entity] = Field(..., description="All extracted entities with their relations")

    @root_validator(skip_on_failure=True)
    def validate_entities(cls, values):
        entities = values.get('entities', [])
        if not entities:
            raise ValueError("Output must include at least one entity")
        # Ensure unique ent_idx and consistent
        idxs = [e.ent_idx for e in entities]
        if len(idxs) != len(set(idxs)):
            raise ValueError("Each entity must have a unique 'ent_idx'")
        return values

class Finding(BaseModel):
    IDX: int = Field(description="The index number of the finding from the input")
    DAY: int = Field(description="The day number when the finding was observed")
    finding: str = Field(description="The description of the finding")

class Episode(BaseModel):
    episode: int = Field(description="Sequential episode number")
    days: List[int] = Field(description="Array of day numbers that belong to this episode")

class FindingGroup(BaseModel):
    group_name: str = Field(description="Name of the finding group")
    findings: List[Finding] = Field(description="List of all findings in this group")
    episodes: List[Episode] = Field(description="Temporal groupings of these findings")
    rationale: str = Field(description="Explanation for the grouping decisions")

class RadiologyOutput(BaseModel):
    """Output schema for normalized and grouped radiological findings"""
    results: List[FindingGroup] = Field(description="List of finding groups including episodes and findings")

'''
def get_local_llm(model):

    if model == "medgemma":
        model_name = "google/medgemma-27b-text-it"
        llm = LLM(model=model_name, # mistralai/Mistral-Small-24B-Instruct-2501
              tensor_parallel_size=4,  # number of attention heads must be divisible by the number of GPUs; currently 32 attention heads
              gpu_memory_utilization=0.85,
              )
    elif model == "mistral":
        model_name = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        llm = LLM(model=model_name, # mistralai/Mistral-Small-24B-Instruct-2501
              tensor_parallel_size=4,  # number of attention heads must be divisible by the number of GPUs; currently 32 attention heads
              )
    else:
        raise ValueError(f"Invalid model: {model}")

    return llm
'''


def save_llm_outputs(outputs, messages_json, save_path='results.jsonl'):
    """
    Save vLLM outputs to a JSONL file.
    Each result is stored together with its key from messages_json and the last user query.
    When guided decoding is used, JSON parsing is also performed.
    """
    # messages_json: dict, keys are e.g. "33_indication", values are list of messages
    # outputs: list, same order as list(messages_json.values())
    message_keys = list(messages_json.keys())
    results = []
    for i, output in enumerate(outputs):
        try:
            # 1. Extract the base text
            response_text = output.outputs[0].text

            # 2. Extract the key and the last user query from messages_json
            if i < len(message_keys):
                key = message_keys[i]
                messages = messages_json[key]
                # Extract the content of the last user message
                user_query = None
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_query = msg.get("content")
                        break
            else:
                key = None
                user_query = None

            # 3. Try to parse as JSON (used when guided decoding is enabled)
            try:
                parsed_json = json.loads(response_text)
                result = {
                    "index": i,
                    "key": key,
                    "user_query": user_query,
                    "raw_response": response_text,
                    "parsed_response": parsed_json,
                    "parsing_success": True
                }
            except json.JSONDecodeError as e:
                result = {
                    "index": i,
                    "key": key,
                    "user_query": user_query,
                    "raw_response": response_text,
                    "parsed_response": None,
                    "parsing_success": False,
                    "parsing_error": str(e)
                }

            # 4. Include extra metadata
            if hasattr(output, 'request_id'):
                result["request_id"] = output.request_id

            results.append(result)

        except Exception as e:
            # Record errors as well
            error_result = {
                "index": i,
                "error": str(e),
                "parsing_success": False
            }
            results.append(error_result)

    # Save as a single JSON file (pretty print)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {save_path}")


def read_batch_file(config, batch_file_path, fail_case=None):

    
    if fail_case != None:
        fail_custom_id = list(fail_case.keys())

    batch_file = []
    with open(batch_file_path, 'r') as f:
        for line in f:
            json_line = json.loads(line.strip())

            if fail_case != None and json_line['custom_id'] not in fail_custom_id:
                continue
            else:
                batch_file.append(json_line)

    if config['mode']['sample'] == True:
        batch_file = batch_file[:1000]

    return batch_file

def estimate_token_usage(deployment_name, conversation, gt_output=None, expected_completion_length=10000):
    # Use GPT-4's default token encoder

    enc = tiktoken.encoding_for_model("gpt-5")
    # Compute the number of prompt tokens
    prompt_text = " ".join([msg["content"] for msg in conversation])
    prompt_tokens = len(enc.encode(prompt_text))


    if gt_output:
        completion_tokens = len(enc.encode(gt_output))
    else:
        completion_tokens = expected_completion_length

    return prompt_tokens, completion_tokens


# ms azure east us 2, global
def estimate_llm_cost(deployment_name, prompt_tokens, completion_tokens, cached_input=False):

    cost_dict = {
        "gpt-5-nano": {
            "input_cost": 0.05,
            "cached_input_cost": 0.01,
            "output_cost": 0.4,
        },
        "gpt-5-mini": {
            "input_cost": 0.25,
            "cached_input_cost": 0.03,
            "output_cost": 2,
        },
        "gpt-5": {
            "input_cost": 1.25,
            "cached_input_cost": 0.13,
            "output_cost": 10,
        },
        "gpt-5-batch": {
            "input_cost": 0.63,
            "cached_input_cost": 0.07,
            "output_cost": 5,
        },
        "gpt-5.2": {
            "input_cost": 1.75,
            "cached_input_cost": 0.175,
            "output_cost": 14,
        },
        "gpt-4.1": {
            "input_cost": 2,
            "cached_input_cost": 0.5,
            "output_cost": 8,
        },
        "claude-sonnet-4-5": {
            "input_cost": 3,
            "cached_input_cost": 0.3,
            "output_cost": 15,
        },
        "claude-haiku-4-5": {
            "input_cost": 1,
            "cached_input_cost": 0.1,
            "output_cost": 5,
        },
        'gemini-3-flash-preview': {
            "input_cost": 0.5,
            "cached_input_cost": 0.05,
            "output_cost": 3,
        },
        'gemini-3-pro-preview': {
            "input_cost": 2,
            "cached_input_cost": 0.2,
            "output_cost": 12,
        },
        'gemini-2.5-flash': {
            "input_cost": 0.3,
            "cached_input_cost": 0.03,
            "output_cost": 2.5,
        }
    }
    # per 1M token cost
    if deployment_name not in cost_dict:
        raise ValueError(f"Invalid deployment name: {deployment_name}")
    
    million = 1000000
    total_cost = {}
    for deployment in cost_dict.keys():
        input_cost = cost_dict[deployment]['input_cost']
        cached_input_cost = cost_dict[deployment]['cached_input_cost']
        output_cost = cost_dict[deployment]['output_cost']
        
        if cached_input:
            cost = (cached_input_cost*prompt_tokens/million + output_cost*completion_tokens/million)
        else:
            cost = (input_cost*prompt_tokens/million + output_cost*completion_tokens/million)
            
        total_cost[deployment] = cost
        
    return total_cost

model_output_example = """{
  "entities": [
    {
      "name": "lung volumes",
      "sent_idx": 1,
      "ent_idx": 1,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "POSITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Measurement",
          "value": "low"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "crowding",
      "sent_idx": 2,
      "ent_idx": 2,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "POSITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Morphology",
          "value": "crowding"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Associate",
          "value": "bronchovascular structures",
          "obj_ent_idx": 3
        }
      ]
    },
    {
      "name": "bronchovascular structures",
      "sent_idx": 2,
      "ent_idx": 3,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "POSITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "pulmonary vascular congestion",
      "sent_idx": 3,
      "ent_idx": 4,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "POSITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "TENTATIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Severity",
          "value": "mild"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "heart size",
      "sent_idx": 4,
      "ent_idx": 5,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "POSITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Measurement",
          "value": "enlarged"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Severity",
          "value": "borderline"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "mediastinal contours",
      "sent_idx": 5,
      "ent_idx": 6,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "NEGATIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Location",
          "value": "mediastinal"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "hilar contours",
      "sent_idx": 5,
      "ent_idx": 7,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "NEGATIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Morphology",
          "value": "unremarkable"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "nodules",
      "sent_idx": 6,
      "ent_idx": 8,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "POSITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Location",
          "value": "lung fields"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Measurement",
          "value": "innumerable"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Associate",
          "value": "metastatic disease",
          "obj_ent_idx": 9
        }
      ]
    },
    {
      "name": "metastatic disease",
      "sent_idx": 6,
      "ent_idx": 9,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "POSITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Location",
          "value": "left upper and left lower lung fields"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "consolidation",
      "sent_idx": 7,
      "ent_idx": 10,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "NEGATIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Distribution",
          "value": "focal"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "pleural effusion",
      "sent_idx": 7,
      "ent_idx": 11,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "NEGATIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "pneumothorax",
      "sent_idx": 7,
      "ent_idx": 12,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "NEGATIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "right hemidiaphragm",
      "sent_idx": 7,
      "ent_idx": 13,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "POSITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Location",
          "value": "right hemidiaphragm"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "elevation",
      "sent_idx": 7,
      "ent_idx": 14,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "POSITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Location",
          "value": "right hemidiaphragm"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "lobectomy",
      "sent_idx": 8,
      "ent_idx": 15,
      "relations": [
        {
          "relation": "Cat",
          "value": "OTH"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "POSITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Location",
          "value": "right lower"
        ,
          "obj_ent_idx": null
        }
      ]
    },
    {
      "name": "rib deformities",
      "sent_idx": 9,
      "ent_idx": 16,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "POSITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Location",
          "value": "right hemithorax"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Associate",
          "value": "postsurgical changes",
          "obj_ent_idx": 18
        }
      ]
    },
    {
      "name": "postsurgical changes",
      "sent_idx": 9,
      "ent_idx": 18,
      "relations": [
        {
          "relation": "Cat",
          "value": "PF"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Status",
          "value": "POSITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Dx_Certainty",
          "value": "DEFINITIVE"
        ,
          "obj_ent_idx": null
        },
        {
          "relation": "Past Hx",
          "value": "prior"
        ,
          "obj_ent_idx": null
        }
      ]
    }
  ]
}"""

def json_parse(model_output):

    if isinstance(model_output, str):
        # If body is a string, attempt to parse it
        try:
            model_output = json.loads(model_output)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse body as JSON")
            model_output = None

    return model_output

def transform_messages_to_contents_for_gemini(messages):
    contents = []
    system_prompt = messages[0]['content']
    
    for message in messages[1:]:
        contents.append(
          {
            "parts": [
              {
                "text": message['content']
              }
            ],
            "role": message['role']
          }
        )
        
    return contents, system_prompt

def run_llm_chat(config, batch_file_path, client, deployment_name, step='structuring', fail_case=None):

    batch_file = read_batch_file(config, batch_file_path, fail_case)

    # Set up the folder structure
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(config[f'llm_{step}']['output_path'], 'lunguage_score' if config['mode']['lunguage_score'] == True else 'lesionbench')
    processed_response_dir = os.path.join(base_dir, 'processed_response')
    os.makedirs(processed_response_dir, exist_ok=True)

    # processed file path
    local_save_path = os.path.join(processed_response_dir, f"{config[f'llm_{step}']['deployment_name']}_outputs_{timestamp}.json")
    print(f"Processed results will be saved to: {local_save_path}")

    answer = esimate_inference_cost(config, batch_file_path, deployment_name)
    
    if answer != 'y':
        return None
    
    results = {}
    
    for file in tqdm(batch_file, desc="Processing File", total=len(batch_file)):
    
        messages = file['body']['messages']
        custom_id = file['custom_id']

        try:
            if deployment_name.startswith('gemini'):
                contents, system_prompt = transform_messages_to_contents_for_gemini(messages)
                response = client.models.generate_content(
                    model=deployment_name,
                    contents=contents,
                    config= types.GenerateContentConfig(
                        system_instruction = system_prompt,  # system prompt goes here
                        response_mime_type = 'application/json',
                        response_schema = StructuredOutput if step == 'structuring' else RadiologyOutput,
                    )
                )
                parsed_response = json_parse(response.text)
                
                if parsed_response is not None:
                    result_data = {
                        'custom_id': custom_id,
                        'success': True,
                        "raw_response": response.text,
                        "parsed_query": json_parse(messages[-1]['content']),
                        'parsed_response': parsed_response,
                        'usage': {
                            'input_tokens': response.usage_metadata.prompt_token_count if hasattr(response.usage_metadata, 'prompt_token_count') else None,
                            'output_tokens': (response.usage_metadata.candidates_token_count if hasattr(response.usage_metadata, 'candidates_token_count') else 0) + (response.usage_metadata.thoughts_token_count if hasattr(response.usage_metadata, 'thoughts_token_count') else 0),
                        } if hasattr(response, 'usage_metadata') else None
                    }
                else:
                    result_data = {
                        'custom_id': custom_id,
                        'success': False,
                        'error': 'Failed to parse JSON response'
                    }
            else:
                response = client.chat.completions.parse( # produce the structured format here
                    messages=messages,
                    max_completion_tokens=32768, # 32768
                    model=deployment_name,
                    response_format = StructuredOutput if step == 'structuring' else RadiologyOutput,
                    #reasoning_effort="low", # low, medium, high
                )
                parsed_response = json_parse(response.choices[0].message.content)
                
                if parsed_response is not None:
                    result_data = {
                        'custom_id': custom_id,
                        'success': True,
                        "raw_response": response.choices[0].message.content,
                        "parsed_query": json_parse(messages[-1]['content']),
                        'parsed_response': parsed_response,
                        'usage': {
                            'input_tokens': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else None,
                            'output_tokens': response.usage.completion_tokens + response.usage.completion_tokens_details.reasoning_tokens if hasattr(response.usage, 'completion_tokens') and hasattr(response.usage.completion_tokens_details, 'reasoning_tokens') else None,
                        } if hasattr(response, 'usage') else None
                    }
                else:
                    result_data = {
                        'custom_id': custom_id,
                        'success': False,
                        'error': 'Failed to parse JSON response'
                    }
        except Exception as e:
            result_data = {
                'custom_id': custom_id,
                'success': False,
                'error': f"Error processing result: {str(e)}"
            }
            
        results[custom_id] = result_data
    
    # Save in JSONL format (one result per line); same structure as run_llm_batch
    with open(local_save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {local_save_path}")
    print(f"Total results: {len(results)}")
    print(f"Successful: {sum(1 for r in results.values() if r.get('success', False))}")
    print(f"Failed: {sum(1 for r in results.values() if not r.get('success', False))}")


def resolve_refs(schema, defs=None):
    """
    Recursively dereference $ref entries in the JSON Schema produced by Pydantic.
    The Gemini Batch API may not support $ref/$defs structures.
    """
    if defs is None:
        defs = schema.get('$defs', {}) or schema.get('definitions', {})

    if isinstance(schema, dict):
        # If a $ref key is present, fetch its definition and resolve recursively
        if '$ref' in schema:
            ref_key = schema['$ref'].split('/')[-1]
            if ref_key in defs:
                return resolve_refs(defs[ref_key], defs)

        # For regular dicts, recurse on every value
        return {k: resolve_refs(v, defs) for k, v in schema.items() if k != '$defs' and k != 'definitions'}

    elif isinstance(schema, list):
        # For lists, recurse on each item
        return [resolve_refs(item, defs) for item in schema]

    return schema

def transform_batch_file_to_contents_for_gemini(config, batch_file_path, deployment_name, step='structuring', fail_case=None):

    batch_file = read_batch_file(config, batch_file_path, fail_case)

    # Save under the request folder
    base_dir = os.path.join(config[f'llm_{step}']['output_path'], 'lunguage_score' if config['mode']['lunguage_score'] == True else 'lesionbench')
    request_dir = os.path.join(base_dir, 'request')
    os.makedirs(request_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file_path = os.path.join(request_dir, f"{config[f'llm_{step}']['deployment_name']}_batch_requests_{timestamp}.jsonl")
    
    target_schema = StructuredOutput if step == 'structuring' else RadiologyOutput
    raw_schema = target_schema.model_json_schema()
    resolved_schema = resolve_refs(raw_schema)

    with open(output_file_path, 'w') as f:
        for file in batch_file:

            messages = file['body']['messages'] if 'body' in file else file['messages']
            contents, system_prompt = transform_messages_to_contents_for_gemini(messages)

            request_dict = {
                "model": deployment_name,
                "contents": contents,
                "systemInstruction": {
                    "parts": [
                        {
                            "text": system_prompt
                        }
                    ]
                },
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": resolved_schema
                }
            }

            batch_request = {"key": f"request_{file['custom_id']}", "request": request_dict}
            f.write(json.dumps(batch_request) + "\n")
    
    return output_file_path

def process_batch_file_for_azure(batch_file_path, client):

    try:
        batch_file = client.files.create(
                        file=open(batch_file_path, "rb"),
                        purpose="batch"
                        )
        print(f"Batch file uploaded: {batch_file}")
        print(client.files.list())
        print(batch_file.model_dump_json(indent=2))
        file_id = batch_file.id
        
        status = "pending"
        while status != "processed":
            time.sleep(15)
            file_response = client.files.retrieve(file_id)
            status = file_response.status
            print(f"{datetime.datetime.now()} File Id: {file_id}, Status: {status}")
    except KeyboardInterrupt:
        print("\nReceived an interrupt request from the user. Deleting the Batch File on the server...")
        client.files.delete(batch_file.id)
        print(f"Batch File deleted: {batch_file.id}")
        return None
    except Exception as e:
        
        client.files.delete(batch_file.id)
        print(f"Batch File deleted: {batch_file.id}")
        
        return None
    
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        # endpoint="https://YOUR_AZURE_ENDPOINT/",
        endpoint="/v1/chat/completions",                        
        completion_window="24h"
    )
    
    batch_id = batch_job.id
    print(batch_job.model_dump_json(indent=2))
    
    try:
        
        status = "validating"
        while status not in ("completed", "failed", "canceled"):
            time.sleep(60)
            batch_response = client.batches.retrieve(batch_id)
            status = batch_response.status
            print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

        print(batch_response.model_dump_json(indent=2))

        response = client.files.content(batch_response.output_file_id)
        raw_responses = response.text.strip().split('\n')  

        for raw_response in raw_responses:  
            json_response = json.loads(raw_response)  
            formatted_json = json.dumps(json_response, indent=2)  
            print(formatted_json)
            
    except KeyboardInterrupt:
        print("\nReceived an interrupt request from the user. Cancelling the Batch job on the server...")
        client.batches.cancel(batch_id) # send the cancel command to the server
        print(f"Cancellation requested for batch {batch_id}.")
        print(client.batches.list())
    except Exception as e:
        client.batches.cancel(batch_id) # send the cancel command to the server
        print(f"Cancellation requested for batch {batch_id}.")
        print(client.batches.list())
        
    return batch_response



def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    """Upload a local file to a GCS bucket and return its gs:// path."""
    storage_client = storage.Client() # authentication uses the GOOGLE_APPLICATION_CREDENTIALS env var
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_file_path)

    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    print(f"File upload complete: {gcs_uri}")
    return gcs_uri


def download_from_gcs(gcs_uri, local_file_name):
    """Parse a gs:// URI for bucket and filename, then download the object locally."""
    # Parse the gs://bucket_name/path/to/file form
    match = re.match(r'gs://([^/]+)/(.+)', gcs_uri)
    if not match:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    bucket_name = match.group(1)
    blob_name = match.group(2)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    print(f"Downloading from GCS: {gcs_uri} -> {local_file_name}")
    blob.download_to_filename(local_file_name)

def check_batch_job_status(client):
    # Fetch the full list of batch jobs
    # page_size: number of items per page (recommend setting higher than the default)
    batch_jobs = client.batches.list(config={"page_size": 100})

    print(f"{'Job ID (Name)':<60} | {'Status':<15} | {'Created Time'}")
    print("-" * 100)

    for job in batch_jobs:
        # job.name appears as a long path (projects/.../jobs/...).
        # job.state: JOB_STATE_RUNNING, SUCCEEDED, FAILED, etc.
        print(f"{job.name:<60} | {job.state:<15} | {job.create_time}")


def retrieve_and_save_from_gcs(gcs_uri, local_save_path):
    """Retrieve results from Cloud Storage and save as a local .jsonl file."""
    try:
        # Connect to the GCS filesystem (requires the gcsfs library)
        fs = fsspec.filesystem("gcs")

        # 1. Find the predictions.jsonl file via glob.
        # Batch API results usually follow '<base path>/job-id_folder/predictions.jsonl'.
        search_pattern = f"{gcs_uri.rstrip('/')}/*/predictions.jsonl"
        file_paths = fs.glob(search_pattern)

        if not file_paths:
            raise FileNotFoundError(f"No prediction .jsonl files found in directory: {gcs_uri}")

        # If multiple matches are returned, use the first
        target_file = file_paths[0]
        print(f"📄 Found result file on GCS: {target_file}")

        # 2. Download the file locally (using fs.get)
        print(f"⬇️ Downloading to {local_save_path}...")
        os.makedirs(os.path.dirname(local_save_path), exist_ok=True)
        fs.get(target_file, local_save_path)
        
        print(f"✅ Download completed successfully.")

    except Exception as e:
        print(f"❌ An error occurred while retrieving from GCS: {e}")


def process_batch_file_for_gemini(config, batch_file_path, client, deployment_name, step='structuring', fail_case=None):

    batch_file_path = transform_batch_file_to_contents_for_gemini(config, batch_file_path, deployment_name, step, fail_case)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_gcs_path = f"gs://lesionbench/results/{timestamp}"

    # Set up the folder structure
    base_dir = os.path.join(config[f'llm_{step}']['output_path'], 'lunguage_score' if config['mode']['lunguage_score'] == True else 'lesionbench')
    raw_response_dir = os.path.join(base_dir, 'raw_response')
    processed_response_dir = os.path.join(base_dir, 'processed_response')
    os.makedirs(raw_response_dir, exist_ok=True)
    os.makedirs(processed_response_dir, exist_ok=True)

    # raw file path
    raw_save_path = os.path.join(raw_response_dir, f"{config[f'llm_{step}']['deployment_name']}_batch_outputs_{timestamp}.jsonl")
    # processed file path
    local_save_path = os.path.join(processed_response_dir, f"{config[f'llm_{step}']['deployment_name']}_batch_outputs_{timestamp}.json")
    print(f"Raw results will be saved to: {raw_save_path}")
    print(f"Processed results will be saved to: {local_save_path}")
    
    input_gcs_uri = upload_to_gcs(batch_file_path, "lesionbench", f"{step}_batch_file_{'lunguage_score' if config['mode']['lunguage_score'] == True else 'lesionbench'}.jsonl")
    print(f"Uploaded file: {input_gcs_uri}")
    
    check_batch_job_status(client)

    # 3. Create the batch job (specify dest)
    file_batch_job = client.batches.create(
        model=deployment_name,
        src=input_gcs_uri,
        config=CreateBatchJobConfig(dest=output_gcs_path)
    )

    print(f"Created batch job: {file_batch_job.name}")
    
    job_name = file_batch_job.name
    batch_job = client.batches.get(name=job_name)
    status = "JOB_STATE_PENDING"

    try:
        while status != "JOB_STATE_SUCCEEDED":
            time.sleep(10)
            batch_job = client.batches.get(name=job_name)
            print(f"Polling status for job: {job_name}")
            print(f"Current state: {batch_job.state.name}")
            status = batch_job.state.name

            if status == "JOB_STATE_FAILED":
                raise Exception(f"Batch job {job_name} failed")

        print(f"Batch job {job_name} completed successfully")

        # --- [updated section] result download logic ---
        # 1st priority: check the GCS URI (likely path since the input came from GCS)
        if batch_job.dest and hasattr(batch_job.dest, 'gcs_uri') and batch_job.dest.gcs_uri:
            print(f"Results located in GCS: {batch_job.dest.gcs_uri}")
            # Save the raw results under the raw_response folder
            retrieve_and_save_from_gcs(batch_job.dest.gcs_uri, raw_save_path)
            print(f"Raw results saved to: {raw_save_path}")

    except KeyboardInterrupt:
        print("\nReceived an interrupt request from the user. Cancelling the Batch job on the server...")
        client.batches.cancel(name=job_name)
        print(f"Cancellation requested for batch {job_name}.")
        time.sleep(10)
        check_batch_job_status(client)
    except Exception as e:
        client.batches.cancel(name=job_name)
        print(f"Cancellation requested for batch {job_name}.")
        time.sleep(10)
        check_batch_job_status(client)

    # Collect results by reading the JSONL file line by line (raw_save_path was defined above)
    results = {}
    with open(raw_save_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                batch_response = json.loads(line)
                
                # Extract custom_id (strip the request_ prefix from key)
                custom_id = None
                if 'key' in batch_response:
                    key = batch_response['key']
                    if key.startswith('request_'):
                        custom_id = key.replace('request_', '')

                # Extract the result from response
                if 'response' in batch_response and 'candidates' in batch_response['response']:
                    if batch_response['response']['candidates']:
                        candidate = batch_response['response']['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            if candidate['content']['parts']:
                                text = candidate['content']['parts'][0].get('text', '')
                                try:
                                    json_obj = json.loads(text)
                                    
                                    result_data = {
                                        'custom_id': custom_id,
                                        'success': True,
                                        'raw_response': text,
                                        'parsed_query': json_parse(batch_response['request']['contents'][-1]['parts'][0]['text']),
                                        'parsed_response': json_obj,
                                        'usage': {
                                            'input_tokens': batch_response['response'].get('usageMetadata', {}).get('promptTokenCount'),
                                            'output_tokens': batch_response['response'].get('usageMetadata', {}).get('candidatesTokenCount', 0) + batch_response['response'].get('usageMetadata', {}).get('thoughtsTokenCount', 0)
                                        } if 'usageMetadata' in batch_response['response'] else None
                                    }
                                    results[custom_id] = result_data
                                except json.JSONDecodeError as e:
                                    result_data = {
                                        'custom_id': custom_id,
                                        'success': False,
                                        'error': f"Error parsing JSON from text: {str(e)}"
                                    }
                                    results[custom_id] = result_data
                            else:
                                result_data = {
                                    'custom_id': custom_id,
                                    'success': False,
                                    'error': 'No parts in content'
                                }
                                results[custom_id] = result_data
                        else:
                            result_data = {
                                'custom_id': custom_id,
                                'success': False,
                                'error': 'No content or parts in candidate'
                            }
                            results[custom_id] = result_data
                    else:
                        result_data = {
                            'custom_id': custom_id,
                            'success': False,
                            'error': 'No candidates in response'
                        }
                        results[custom_id] = result_data
                else:
                    result_data = {
                        'custom_id': custom_id,
                        'success': False,
                        'error': 'No response or candidates in batch_response'
                    }
                    results[custom_id] = result_data
            except json.JSONDecodeError as e:
                result_data = {
                    'custom_id': None,
                    'success': False,
                    'error': f"Error parsing line as JSON: {str(e)}"
                }
                results[custom_id] = result_data
            except Exception as e:
                result_data = {
                    'custom_id': custom_id,
                    'success': False,
                    'error': f"Error processing result: {str(e)}"
                }
                results[custom_id] = result_data
    
    # Save in JSONL format (one result per line); same format as the Claude code path
    with open(local_save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {local_save_path}")
    print(f"Total results: {len(results)}")
    print(f"Successful: {sum(1 for r in results.values() if r.get('success', False))}")
    print(f"Failed: {sum(1 for r in results.values() if not r.get('success', False))}")

    return results

def process_batch_file_for_claude(config, batch_file_path, client, deployment_name, step='structuring'):

    batch_file = read_batch_file(config, batch_file_path)
    custom_id_to_idx = {file['custom_id']: idx for idx, file in enumerate(batch_file)}

    requests = transform_batch_file_to_contents_for_claude(batch_file, deployment_name, step)

    # Pre-define the output path (uses a timestamp for distinction)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(config[f'llm_{step}']['output_path'], 'lunguage_score' if config['mode']['lunguage_score'] == True else 'lesionbench')

    # Set up the folder structure
    processed_response_dir = os.path.join(base_dir, 'processed_response')
    os.makedirs(processed_response_dir, exist_ok=True)

    # processed response path
    local_save_path = os.path.join(processed_response_dir, f"{config[f'llm_{step}']['deployment_name']}_batch_outputs_{timestamp}.json")
    print(f"Processed results will be saved to: {local_save_path}")

    message_batch = client.beta.messages.batches.create(
        requests=requests,
    )
    
    print(message_batch)
    batch_id = message_batch.id
    
    try:
        while True:
            message_batch = client.messages.batches.retrieve(
                batch_id
            )
            if message_batch.processing_status == "ended":
                break

            print(f"Batch {batch_id} is still processing...")
            time.sleep(10)

        
        # Collect and save results
        results = {}

        for result in client.messages.batches.results(batch_id):

            try:
                # Process only successful results
                if hasattr(result, 'result') and hasattr(result.result, 'message'):
                    message = result.result.message

                    # Extract the result from ToolUseBlock
                    tool_result = None
                    if hasattr(message, 'content') and message.content:
                        for content_block in message.content:
                            if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                                if hasattr(content_block, 'input'):
                                    input_data = content_block.input
                                    # Extract results from input (dict or dict-like)
                                    tool_result = input_data
                                    '''
                                    try:
                                        if isinstance(input_data, dict):
                                            tool_result = input_data.get('results')
                                        elif hasattr(input_data, 'get'):
                                            tool_result = input_data.get('results')
                                        if tool_result is not None:
                                            break
                                    except Exception:
                                        pass
                                    
                                    
                                    '''
                    # Save the result
                    result_data = {
                        'custom_id': result.custom_id,
                        'success': True,
                        'raw_response': json.dumps(tool_result),
                        'parsed_query': json_parse(batch_file[custom_id_to_idx[result.custom_id]]['body']['messages'][-1]['content']),
                        'parsed_response': tool_result,
                        'message_id': message.id if hasattr(message, 'id') else None,
                        'usage': {
                            'input_tokens': message.usage.input_tokens if hasattr(message.usage, 'input_tokens') else None,
                            'output_tokens': message.usage.output_tokens if hasattr(message.usage, 'output_tokens') else None,
                        } if hasattr(message, 'usage') else None
                    }
                    results[result.custom_id] = result_data
                else:
                    # Handle failed results
                    result_data = {
                        'custom_id': result.custom_id if hasattr(result, 'custom_id') else None,
                        'success': False,
                        'error': str(result) if hasattr(result, 'result') else 'Unknown error'
                    }
                    results[result.custom_id] = result_data
            except Exception as e:
                # Handle parsing errors
                result_data = {
                    'custom_id': result.custom_id if hasattr(result, 'custom_id') else None,
                    'success': False,
                    'error': f"Error parsing result: {str(e)}"
                }
                results[result.custom_id] = result_data

        # Save in JSONL format (one result per line) for the processed response
        with open(local_save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"Results saved to {local_save_path}")
        print(f"Total results: {len(results)}")
        print(f"Successful: {sum(1 for r in results.values() if r.get('success', False))}")
        print(f"Failed: {sum(1 for r in results.values() if not r.get('success', False))}")
            
    except KeyboardInterrupt:
        client.messages.batches.cancel(batch_id)
        print(f"Batch {batch_id} cancelled")
        return None
    except Exception as e:
        client.messages.batches.cancel(batch_id)
        print(f"Batch {batch_id} cancelled")
        print(f"Error: {str(e)}")
        return None

    return results

def transform_batch_file_to_contents_for_claude(batch_file, deployment_name, step='structuring'):

    requests = []

    all_tools = [
        {
            "name": "structuring",
            "description": "extract the structured information from the report",
            "input_schema": StructuredOutput.model_json_schema()
        },
        {
            "name": "connecting",
            "description": "build the sequential matching object",
            "input_schema": RadiologyOutput.model_json_schema()
        }
    ]
    
    target_tool = next((t for t in all_tools if t["name"] == step), None)
    
    if not target_tool:
        raise ValueError(f"Invalid step name: {step}. Must be 'structuring' or 'connecting'.")

    for file in batch_file:
        request = Request(
            custom_id=file['custom_id'],
            params=MessageCreateParamsNonStreaming(
                model=deployment_name,
                max_tokens=10000,
                system=[
                    {
                        "type": "text",
                        "text": file['body']['messages'][0]['content'],
                        "cache_control": {
                            "type": "ephemeral"
                        }
                    }
                ],   
                messages=file['body']['messages'][1:],
                tools=[target_tool],
                tool_choice={"type": "tool", "name": step}
            )
        )
        requests.append(request)
        
    return requests



def esimate_inference_cost(config, batch_file_path, deployment_name, fail_case=None):

    batch_file = read_batch_file(config, batch_file_path, fail_case)

    # Dict accumulating the total cost per model
    total_costs = {}
    for file in batch_file:
        messages = file['body']['messages'] if 'body' in file else file['messages']
        prompt_tokens, completion_tokens = estimate_token_usage(deployment_name, messages)
        cost = estimate_llm_cost(deployment_name, prompt_tokens, completion_tokens)

        # Accumulate the cost of this request per model
        for model_name, model_cost in cost.items():
            total_costs[model_name] = total_costs.get(model_name, 0) + model_cost

    # Print the total cost per model
    print(f"Current model: {deployment_name}")
    print("Total cost per model:")

    for model_name, model_cost in total_costs.items():
        print(f"{model_name}: {model_cost}")

    print("Are you sure to proceed? (y/n)")
    answer = input()

    return answer

def run_llm_batch(config, batch_file_path, client, deployment_name, step='structuring', fail_case=None):
    
    answer = esimate_inference_cost(config, batch_file_path, deployment_name, fail_case)
    
    if answer != 'y':
        return None
    
    if deployment_name.startswith('gemini'):
        batch_response = process_batch_file_for_gemini(config, batch_file_path, client, deployment_name, step, fail_case)
    elif deployment_name.startswith('claude'):
        batch_response = process_batch_file_for_claude(config, batch_file_path, client, deployment_name, step)
    else:
        batch_response = process_batch_file_for_azure(batch_file_path, client)

    return batch_response

