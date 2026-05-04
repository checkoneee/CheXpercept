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

# 1. 가장 안쪽: 특정 시퀀스(리포트)에서의 발견 내용
class SequenceFinding(BaseModel):
    sequence_id: str = Field(..., description="The identifier of the report, e.g., 'Sequence 1'")
    text: str = Field(..., description="The extracted text describing the disease in this report")

# 2. 중간: 하나의 질병과 그에 대한 시퀀스별 발견 내용들의 묶음
class DiseaseEntry(BaseModel):
    disease_name: str = Field(..., description="The name of the disease or finding")
    findings: List[SequenceFinding]

# 3. 가장 바깥쪽: 전체 결과 (질병들의 리스트)
class ExtractionResult(BaseModel):
    diseases: List[DiseaseEntry]

# Attribute extraction을 위한 모델
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
              tensor_parallel_size=4,  # attention head 수가 GPU 개수로 나누어져야 함, attention head 수 현재 32
              gpu_memory_utilization=0.85,
              )
    elif model == "mistral":
        model_name = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        llm = LLM(model=model_name, # mistralai/Mistral-Small-24B-Instruct-2501
              tensor_parallel_size=4,  # attention head 수가 GPU 개수로 나누어져야 함, attention head 수 현재 32
              )
    else:
        raise ValueError(f"Invalid model: {model}")
    
    return llm
'''


def save_llm_outputs(outputs, messages_json, save_path='results.jsonl'):
    """
    vLLM outputs을 JSONL 파일로 저장
    각 결과에 messages_json의 key와 마지막 user query도 함께 저장
    Guided Decoding 사용 시 JSON 파싱도 수행
    """
    # messages_json: dict, keys are e.g. "33_indication", values are list of messages
    # outputs: list, same order as list(messages_json.values())
    message_keys = list(messages_json.keys())
    results = []
    for i, output in enumerate(outputs):
        try:
            # 1. 기본 텍스트 추출
            response_text = output.outputs[0].text

            # 2. messages_json에서 key와 마지막 user query 추출
            if i < len(message_keys):
                key = message_keys[i]
                messages = messages_json[key]
                # 마지막 user 메시지 content 추출
                user_query = None
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_query = msg.get("content")
                        break
            else:
                key = None
                user_query = None

            # 3. JSON으로 파싱 시도 (Guided Decoding 사용 시)
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

            # 4. 추가 메타데이터 포함
            if hasattr(output, 'request_id'):
                result["request_id"] = output.request_id

            results.append(result)

        except Exception as e:
            # 에러 발생 시에도 기록
            error_result = {
                "index": i,
                "error": str(e),
                "parsing_success": False
            }
            results.append(error_result)

    # 하나의 JSON 파일로 저장 (pretty print)
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
    # GPT-4의 기본 토큰 인코더 사용

    enc = tiktoken.encoding_for_model("gpt-5")
    # 입력(prompt) 토큰 개수 계산
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
    
    # 폴더 구조 설정
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(config[f'llm_{step}']['output_path'], 'lunguage_score' if config['mode']['lunguage_score'] == True else 'lesionbench')
    processed_response_dir = os.path.join(base_dir, 'processed_response')
    os.makedirs(processed_response_dir, exist_ok=True)
    
    # processed 파일 경로
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
                        system_instruction = system_prompt,  # system prompt는 여기에
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
                response = client.chat.completions.parse( # 여기서 structured format으로 만들어야 한다.
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
    
    # JSONL 형식으로 저장 (한 줄에 하나의 결과) - run_llm_batch와 동일한 구조
    with open(local_save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {local_save_path}")
    print(f"Total results: {len(results)}")
    print(f"Successful: {sum(1 for r in results.values() if r.get('success', False))}")
    print(f"Failed: {sum(1 for r in results.values() if not r.get('success', False))}")


def resolve_refs(schema, defs=None):
    """
    Pydantic이 생성한 JSON Schema의 $ref를 재귀적으로 실제 정의로 치환(Dereferencing)합니다.
    Gemini Batch API는 $ref/$defs 구조를 지원하지 않을 수 있습니다.
    """
    if defs is None:
        defs = schema.get('$defs', {}) or schema.get('definitions', {})

    if isinstance(schema, dict):
        # $ref 키가 있으면 해당 정의 내용을 가져와서 재귀적으로 해결
        if '$ref' in schema:
            ref_key = schema['$ref'].split('/')[-1]
            if ref_key in defs:
                return resolve_refs(defs[ref_key], defs)
        
        # 일반 dict인 경우 모든 value에 대해 재귀 호출
        return {k: resolve_refs(v, defs) for k, v in schema.items() if k != '$defs' and k != 'definitions'}
    
    elif isinstance(schema, list):
        # list인 경우 각 항목에 대해 재귀 호출
        return [resolve_refs(item, defs) for item in schema]
    
    return schema

def transform_batch_file_to_contents_for_gemini(config, batch_file_path, deployment_name, step='structuring', fail_case=None):

    batch_file = read_batch_file(config, batch_file_path, fail_case)

    # request 폴더에 저장
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
        print("\n사용자가 중단 요청을 보냈습니다. 서버의 Batch File을 삭제합니다...")
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
        print("\n사용자가 중단 요청을 보냈습니다. 서버의 Batch 작업을 취소합니다...")
        client.batches.cancel(batch_id) # 서버에 취소 명령 전송
        print(f"Batch {batch_id} 취소 요청 완료.")
        print(client.batches.list())
    except Exception as e:
        client.batches.cancel(batch_id) # 서버에 취소 명령 전송
        print(f"Batch {batch_id} 취소 요청 완료.")
        print(client.batches.list())
        
    return batch_response



def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    """로컬 파일을 GCS 버킷에 업로드하고 gs:// 경로를 반환합니다."""
    storage_client = storage.Client() # 인증은 환경변수(GOOGLE_APPLICATION_CREDENTIALS) 따름
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(local_file_path)
    
    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    print(f"파일 업로드 완료: {gcs_uri}")
    return gcs_uri


def download_from_gcs(gcs_uri, local_file_name):
    """gs:// URI에서 버킷과 파일명을 파싱하여 로컬로 다운로드합니다."""
    # gs://bucket_name/path/to/file 형식을 파싱
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
    # 전체 배치 작업 목록 가져오기
    # page_size: 한 번에 가져올 개수 (기본값보다 넉넉하게 설정 추천)
    batch_jobs = client.batches.list(config={"page_size": 100})

    print(f"{'Job ID (Name)':<60} | {'Status':<15} | {'Created Time'}")
    print("-" * 100)

    for job in batch_jobs:
        # job.name은 긴 경로(projects/.../jobs/...)로 나옵니다.
        # job.state: JOB_STATE_RUNNING, SUCCEEDED, FAILED 등
        print(f"{job.name:<60} | {job.state:<15} | {job.create_time}")


def retrieve_and_save_from_gcs(gcs_uri, local_save_path):
    """Retrieve results from Cloud Storage and save as a local .jsonl file."""
    try:
        # GCS 파일시스템 연결 (gcsfs 라이브러리 필요)
        fs = fsspec.filesystem("gcs")
        
        # 1. glob을 사용하여 predictions.jsonl 파일 찾기
        # Batch API 결과는 보통 '지정경로/job-id_폴더/predictions.jsonl' 형태입니다.
        search_pattern = f"{gcs_uri.rstrip('/')}/*/predictions.jsonl"
        file_paths = fs.glob(search_pattern)
        
        if not file_paths:
            raise FileNotFoundError(f"No prediction .jsonl files found in directory: {gcs_uri}")

        # 여러 개가 잡힐 경우 첫 번째 파일을 대상으로 함
        target_file = file_paths[0]
        print(f"📄 Found result file on GCS: {target_file}")

        # 2. 로컬로 파일 다운로드 (fs.get 사용)
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
    
    # 폴더 구조 설정
    base_dir = os.path.join(config[f'llm_{step}']['output_path'], 'lunguage_score' if config['mode']['lunguage_score'] == True else 'lesionbench')
    raw_response_dir = os.path.join(base_dir, 'raw_response')
    processed_response_dir = os.path.join(base_dir, 'processed_response')
    os.makedirs(raw_response_dir, exist_ok=True)
    os.makedirs(processed_response_dir, exist_ok=True)
    
    # raw 파일 경로
    raw_save_path = os.path.join(raw_response_dir, f"{config[f'llm_{step}']['deployment_name']}_batch_outputs_{timestamp}.jsonl")
    # processed 파일 경로
    local_save_path = os.path.join(processed_response_dir, f"{config[f'llm_{step}']['deployment_name']}_batch_outputs_{timestamp}.json")
    print(f"Raw results will be saved to: {raw_save_path}")
    print(f"Processed results will be saved to: {local_save_path}")
    
    input_gcs_uri = upload_to_gcs(batch_file_path, "lesionbench", f"{step}_batch_file_{'lunguage_score' if config['mode']['lunguage_score'] == True else 'lesionbench'}.jsonl")
    print(f"Uploaded file: {input_gcs_uri}")
    
    check_batch_job_status(client)

    # 3. Batch Job 생성 (dest 지정)
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

        # --- [수정된 부분] 결과 다운로드 로직 ---
        # 1순위: GCS URI 확인 (입력이 GCS였으므로 여기가 실행될 확률 높음)
        if batch_job.dest and hasattr(batch_job.dest, 'gcs_uri') and batch_job.dest.gcs_uri:
            print(f"Results located in GCS: {batch_job.dest.gcs_uri}")
            # raw 결과는 raw_response 폴더에 저장
            retrieve_and_save_from_gcs(batch_job.dest.gcs_uri, raw_save_path)
            print(f"Raw results saved to: {raw_save_path}")

    except KeyboardInterrupt:
        print("\n사용자가 중단 요청을 보냈습니다. 서버의 Batch 작업을 취소합니다...")
        client.batches.cancel(name=job_name)
        print(f"Batch {job_name} 취소 요청 완료.")
        time.sleep(10)
        check_batch_job_status(client)
    except Exception as e:
        client.batches.cancel(name=job_name)
        print(f"Batch {job_name} 취소 요청 완료.")
        time.sleep(10)
        check_batch_job_status(client)

    # JSONL 파일을 줄 단위로 읽어서 결과 수집 (raw_save_path는 이미 위에서 정의됨)
    results = {}
    with open(raw_save_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                batch_response = json.loads(line)
                
                # custom_id 추출 (key에서 request_ 접두사 제거)
                custom_id = None
                if 'key' in batch_response:
                    key = batch_response['key']
                    if key.startswith('request_'):
                        custom_id = key.replace('request_', '')
                
                # response에서 결과 추출
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
    
    # JSONL 형식으로 저장 (한 줄에 하나의 결과) - Claude 코드와 동일한 형식
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

    # 출력 경로 미리 지정 (타임스탬프 등을 붙여 구분)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(config[f'llm_{step}']['output_path'], 'lunguage_score' if config['mode']['lunguage_score'] == True else 'lesionbench')
    
    # 폴더 구조 설정
    processed_response_dir = os.path.join(base_dir, 'processed_response')
    os.makedirs(processed_response_dir, exist_ok=True)
    
    # processed response 경로
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

        
        # 결과 수집 및 저장
        results = {}
        
        for result in client.messages.batches.results(batch_id):

            try:
                # 성공한 결과만 처리
                if hasattr(result, 'result') and hasattr(result.result, 'message'):
                    message = result.result.message
                    
                    # ToolUseBlock에서 결과 추출
                    tool_result = None
                    if hasattr(message, 'content') and message.content:
                        for content_block in message.content:
                            if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                                if hasattr(content_block, 'input'):
                                    input_data = content_block.input
                                    # input에서 results 추출 (dict 또는 dict-like 객체)
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
                    # 결과 저장
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
                    # 실패한 결과 처리
                    result_data = {
                        'custom_id': result.custom_id if hasattr(result, 'custom_id') else None,
                        'success': False,
                        'error': str(result) if hasattr(result, 'result') else 'Unknown error'
                    }
                    results[result.custom_id] = result_data
            except Exception as e:
                # 파싱 에러 처리
                result_data = {
                    'custom_id': result.custom_id if hasattr(result, 'custom_id') else None,
                    'success': False,
                    'error': f"Error parsing result: {str(e)}"
                }
                results[result.custom_id] = result_data
        
        # JSONL 형식으로 저장 (한 줄에 하나의 결과) - processed response
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

    # 각 모델별 총 비용을 누적해서 저장할 딕셔너리
    total_costs = {}
    for file in batch_file:
        messages = file['body']['messages'] if 'body' in file else file['messages']
        prompt_tokens, completion_tokens = estimate_token_usage(deployment_name, messages)
        cost = estimate_llm_cost(deployment_name, prompt_tokens, completion_tokens)
        
        # 이번 요청의 비용을 모델별로 누적
        for model_name, model_cost in cost.items():
            total_costs[model_name] = total_costs.get(model_name, 0) + model_cost

    # 모델별 총 비용 출력
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

