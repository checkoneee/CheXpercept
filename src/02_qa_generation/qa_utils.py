import random
import itertools
import os

def preprocess_anatomy_name(anatomy_name):
    """
    Preprocess anatomy name by replacing '&' with 'intersection of'.
    Example: 'right medial lung & right upper zone lung' -> 'intersection of right medial lung and right upper zone lung'
    """
    if ' & ' in anatomy_name:
        parts = anatomy_name.split(' & ')
        if 'lung base' in parts[1]:
            # lung base는 lower zone lung으로 바꿔
            parts[1] = parts[1].replace('lung base', 'lower zone lung')
        return f'the intersection of the {parts[0]} and the {parts[1]}', parts[1]
    elif 'costophrenic angle' in anatomy_name:
        if 'right' in anatomy_name:
            return f'the {anatomy_name}', 'right lower zone lung'
        elif 'left' in anatomy_name:
            return f'the {anatomy_name}', 'left lower zone lung'
        else:
            raise ValueError(f"Invalid anatomy name: {anatomy_name}")
    else:
        raise ValueError(f"Invalid anatomy name: {anatomy_name}")

def generate_distribution_qa(pos_zones, lesion_name, num_option=4):
    
    all_zones = [
        "right upper zone lung",
        "right mid zone lung", 
        "right lower zone lung",
        "left upper zone lung",
        "left mid zone lung",
        "left lower zone lung"
    ]
    zones_text = ", ".join(all_zones)

    answer_options = {
        1: '1 zone',
        2: '2 zones',
        3: '3 zones',
        4: '4 zones or more',
    }
    
    # Build question with options
    options_text = "\n".join([f"({i}) {opt}" for i, opt in answer_options.items()])
    question = f"In how many zones is the {lesion_name} present in the image? Consider the following zones: {zones_text}.\n\nOptions:\n{options_text}"
    
    if len(pos_zones) == 1:
        answer_index = 1
    elif len(pos_zones) == 2:
        answer_index = 2
    elif len(pos_zones) == 3:
        answer_index = 3
    else:
        answer_index = 4
    
    distribution_qa = {
        'question': question,
        'answer_options': list(answer_options.values()),
        'answer': answer_options[answer_index],
        'answer_index': answer_index,
        'answer_zones': pos_zones,
    }
    
    return distribution_qa

def generate_location_qa(pos_zones, neg_zones, lesion_name, num_option=4):
    
    # pos_zones에서 최대 (num_option-1)개 (0개부터 가능) 랜덤 선택
    max_select = min(num_option - 1, len(pos_zones))
    num_select = random.randint(0, max_select) if max_select > 0 else 0
    selected_pos_zones = random.sample(pos_zones, num_select) if num_select > 0 else []
    
    # selected_pos_zones는 무조건 옵션에 포함
    # 나머지 옵션은 neg_zones에서 선택하여 총 (num_option-1)개가 되도록 함
    num_remaining_options = num_option - 1 - len(selected_pos_zones)
    if num_remaining_options > 0 and len(neg_zones) > 0:
        num_neg_to_select = min(num_remaining_options, len(neg_zones))
        selected_neg_zones = random.sample(neg_zones, num_neg_to_select)
    else:
        selected_neg_zones = []
    
    # selected_pos_zones와 selected_neg_zones를 합쳐서 섞기
    selected_zones = selected_pos_zones + selected_neg_zones
    
    # 만약 selected_zones가 (num_option-1)개보다 적으면, 
    # neg_zones가 부족한 것이므로 남은 pos_zones에서 추가로 선택하여 채움
    if len(selected_zones) < num_option - 1:
        remaining_pos_zones = [z for z in pos_zones if z not in selected_pos_zones]
        num_additional_needed = (num_option - 1) - len(selected_zones)
        if len(remaining_pos_zones) > 0:
            num_additional = min(num_additional_needed, len(remaining_pos_zones))
            additional_pos_zones = random.sample(remaining_pos_zones, num_additional)
            selected_zones.extend(additional_pos_zones)
            selected_pos_zones.extend(additional_pos_zones)  # 정답에도 포함
    
    random.shuffle(selected_zones)
    
    # answer_options 생성 (1부터 num_option-1까지, 항상 (num_option-1)개 생성)
    answer_options = {}
    for i in range(1, num_option):
        answer_options[i] = selected_zones[i-1]
    
    # 마지막 옵션(num_option번)은 항상 "None of the above"
    option_range = f"1-{num_option-1}" if num_option > 2 else "1"
    answer_options[num_option] = f'None of the above (options {option_range})'
    
    # Build question with options
    options_text = "\n".join([f"({i}) {opt}" for i in range(1, num_option + 1) for opt in [answer_options[i]]])
    question = f"Where is the {lesion_name} located in the image? Select all locations where the lesion is present from the options.\n\nOptions:\n{options_text}"
    
    # answer_options를 순서대로 리스트로 변환 (1부터 num_option까지)
    answer_options_list = []
    for i in range(1, num_option + 1):
        if i in answer_options:
            answer_options_list.append(answer_options[i])
    
    # 정답은 selected_pos_zones에 있는 zone들의 인덱스 (1부터 num_option-1번 중에서)
    answer_indices = []
    answer_texts = []
    for idx in range(1, num_option):
        if idx in answer_options and answer_options[idx] in selected_pos_zones:
            answer_indices.append(idx)
            answer_texts.append(answer_options[idx])
    
    # 정답이 없는 경우 (selected_pos_zones가 비어있거나 선택된 옵션에 없는 경우) -> num_option번이 정답
    if not answer_indices:
        answer_indices = [num_option]
        answer_texts = [answer_options[num_option]]
    
    location_qa = {
        'question': question,
        'answer_options': answer_options_list,
        'answer': ', '.join(answer_texts) if answer_texts else 'None',
        'answer_index': answer_indices,
        'answer_zones': selected_pos_zones,
    }
    
    return location_qa

def generate_severity_measurement_qa(overlaps, lesion_name, location_answer_zones=None, num_option=4):
    
    # left lung과 right lung 중 has_overlap=True인 것들 찾기
    lungs_with_overlap = {
        'left lung': {
            'has_overlap': False,
            'overlap_ratio': 0.0,
        },
        'right lung': {
            'has_overlap': False,
            'overlap_ratio': 0.0,
        },
    }
    
    for suboptimal_component_id, overlap in overlaps.items():
        if 'left lung' in overlap and overlap['left lung']['has_overlap']:
            lungs_with_overlap['left lung']['has_overlap'] = True
            lungs_with_overlap['left lung']['overlap_ratio'] += overlap['left lung']['overlap_ratio']
        if 'right lung' in overlap and overlap['right lung']['has_overlap']:
            lungs_with_overlap['right lung']['has_overlap'] = True
            lungs_with_overlap['right lung']['overlap_ratio'] += overlap['right lung']['overlap_ratio']
    
    # location_qa의 정답 zone들을 기반으로 lung 선택
    selected_lung_name = None
    overlap_ratio = 0.0
    
    if location_answer_zones:
        # location_qa의 정답 zone들에서 left/right 확인
        has_left_zones = any('left' in zone for zone in location_answer_zones)
        has_right_zones = any('right' in zone for zone in location_answer_zones)
        
        # 정답 zone을 포함하는 lung 우선 선택
        if has_left_zones and has_right_zones:
            # 양쪽 모두 있으면 랜덤 선택
            if lungs_with_overlap['left lung']['has_overlap'] and lungs_with_overlap['right lung']['has_overlap']:
                selected_lung = random.choice(list(lungs_with_overlap.keys()))
                selected_lung_name = selected_lung
                overlap_ratio = lungs_with_overlap[selected_lung_name]['overlap_ratio']
            elif lungs_with_overlap['left lung']['has_overlap']:
                selected_lung_name = 'left lung'
                overlap_ratio = lungs_with_overlap['left lung']['overlap_ratio']
            elif lungs_with_overlap['right lung']['has_overlap']:
                selected_lung_name = 'right lung'
                overlap_ratio = lungs_with_overlap['right lung']['overlap_ratio']
            else:
                selected_lung_name = 'left lung'
                overlap_ratio = 0.0
        elif has_left_zones:
            # left zone만 있으면 left lung 선택
            if lungs_with_overlap['left lung']['has_overlap']:
                selected_lung_name = 'left lung'
                overlap_ratio = lungs_with_overlap['left lung']['overlap_ratio']
        elif has_right_zones:
            # right zone만 있으면 right lung 선택
            if lungs_with_overlap['right lung']['has_overlap']:
                selected_lung_name = 'right lung'
                overlap_ratio = lungs_with_overlap['right lung']['overlap_ratio']
    
    # location_answer_zones가 없거나 매칭되는 lung이 없으면 기존 로직 사용
    if selected_lung_name is None:
        if lungs_with_overlap['left lung']['has_overlap'] and lungs_with_overlap['right lung']['has_overlap']:
            selected_lung_name = random.choice(list(lungs_with_overlap.keys()))
            overlap_ratio = lungs_with_overlap[selected_lung_name]['overlap_ratio']
        elif lungs_with_overlap['left lung']['has_overlap']:
            selected_lung_name = 'left lung'
            overlap_ratio = lungs_with_overlap['left lung']['overlap_ratio']
        elif lungs_with_overlap['right lung']['has_overlap']:
            selected_lung_name = 'right lung'
            overlap_ratio = lungs_with_overlap['right lung']['overlap_ratio']
        else:
            # 병변이 없는 경우 (이론적으로는 발생하지 않아야 하지만 안전장치)
            overlap_ratio = 0.0
            selected_lung_name = 'left lung'  # 기본값
    
    # answer_options에 선택된 lung 명시
    answer_options = {
        1: f'Less than 1/3 of the {selected_lung_name} area',
        2: f'1/3 or more but less than 2/3 of the {selected_lung_name} area',
        3: f'2/3 or more of the {selected_lung_name} area',
    }
    
    # Build question with options
    options_text = "\n".join([f"({i}) {opt}" for i, opt in answer_options.items()])
    if selected_lung_name:
        question = f"What proportion of the {selected_lung_name} area does the {lesion_name} occupy? Note that the lung area excludes the heart area.\n\nOptions:\n{options_text}"
    else:
        question = f"What proportion of the lung area does the {lesion_name} occupy? Note that the lung area excludes the heart area.\n\nOptions:\n{options_text}"
    
    # overlap_ratio에 따라 answer_index 결정
    if overlap_ratio < 1/3:
        answer_index = 1
    elif overlap_ratio < 2/3:
        answer_index = 2
    else:
        answer_index = 3
    
    severity_measurement_qa = {
        'question': question,
        'answer_options': list(answer_options.values()),
        'answer': answer_options[answer_index],
        'answer_index': answer_index,
        'overlap_ratio': overlap_ratio,
        'selected_lung': selected_lung_name,
    }
    
    return severity_measurement_qa

def generate_comparison_qa(overlaps, lesion_name, num_option=4):
    
    size_ratio_threshold = 1.5
    
    answer_options = {
        1: 'Both lesions are similar in size.',
        2: 'The left lung lesion is larger than the right lung lesion.',
        3: 'The right lung lesion is larger than the left lung lesion.',
        4: 'The lesion is present in only one lung (left or right).',
    }
    
    # Build question with options
    options_text = "\n".join([f"({i}) {opt}" for i, opt in answer_options.items()])
    question = f"Assess and compare the sizes of the lesion across both lungs. If the lesion is present in only one lung, select the corresponding option. Note that the lung area excludes the heart area. A lesion is considered larger if it is {size_ratio_threshold} times or more the size of the other.\n\nOptions:\n{options_text}"
    
    # left lung과 right lung의 size 값 가져오기
    left_size = 0
    right_size = 0
    left_has_overlap = False
    right_has_overlap = False
    
    for suboptimal_component_id, overlap in overlaps.items():

        if overlap['left lung']['has_overlap']:
            left_has_overlap = True
            left_size += overlap['left lung']['size']
        
        if overlap['right lung']['has_overlap']:
            right_has_overlap = True
            right_size += overlap['right lung']['size']
        
    # 정답 결정
    # 4번: 한쪽에만 있음
    if (left_has_overlap and not right_has_overlap) or (not left_has_overlap and right_has_overlap):
        answer_index = 4
    # 둘 다 없으면 (이론적으로는 발생하지 않아야 하지만 안전장치)
    elif not left_has_overlap and not right_has_overlap:
        answer_index = 4
    # 둘 다 있으면 크기 비교
    else:
        # 0으로 나누기 방지
        if left_size == 0 and right_size == 0:
            answer_index = 1  # 둘 다 0이면 비슷함
        elif left_size == 0:
            answer_index = 3  # 오른쪽만 있음
        elif right_size == 0:
            answer_index = 2  # 왼쪽만 있음
        else:
            # 크기 비교 (1.5배 기준)
            size_ratio = max(left_size, right_size) / min(left_size, right_size)
            
            if size_ratio < 1.5:
                # 비슷함 (1.5배 미만 차이)
                answer_index = 1
            elif left_size >= right_size * 1.5:
                # 왼쪽이 큼
                answer_index = 2
            elif right_size >= left_size * 1.5:
                # 오른쪽이 큼
                answer_index = 3
            else:
                # 예외 케이스 (이론적으로는 발생하지 않아야 함)
                answer_index = 1
    
    comparison_qa = {
        'question': question,
        'answer_options': list(answer_options.values()),
        'answer': answer_options[answer_index],
        'answer_index': answer_index,
        'left_size': left_size,
        'right_size': right_size,
        'left_has_overlap': left_has_overlap,
        'right_has_overlap': right_has_overlap,
    }
    
    return comparison_qa

def generate_attribute_extraction_qa(overlaps, lesion_name, num_option=4):
    
    attribute_extraction_qa = {
        'distribution': None,
        'location': None,
        'severity/measurement': None,
        'comparison': None,
    }
    
    all_zones = [
        'right upper zone lung',
        'right mid zone lung',
        'right lung base',
        'left upper zone lung',
        'left mid zone lung',
        'left lung base',
    ]
    
    all_lung_regions = []
    
    for suboptimal_component_id, overlap in overlaps.items():
        for region in overlap['lung_regions'].keys():
            region_name, _ = preprocess_anatomy_name(region)
            all_lung_regions.append(region_name)
    
    all_lung_regions = list(set(all_lung_regions))

    pos_lung_regions = []
    pos_lung_zones = []

    for suboptimal_component_id, overlap in overlaps.items():
        for location_name, location_info in overlap['lung_regions'].items():
            if location_info['has_overlap']:
                pos_lung_region, pos_lung_zone = preprocess_anatomy_name(location_name)
                pos_lung_regions.append(pos_lung_region)
                pos_lung_zones.append(pos_lung_zone)
        
    pos_lung_regions = list(set(pos_lung_regions))
    pos_lung_zones = list(set(pos_lung_zones))
    neg_lung_regions = list(set(all_lung_regions) - set(pos_lung_regions))
    neg_lung_zones = list(set(overlap['zones'].keys()) - set(pos_lung_zones))
    
    attribute_extraction_qa['distribution'] = generate_distribution_qa(pos_lung_zones, lesion_name, num_option)
    attribute_extraction_qa['location'] = generate_location_qa(pos_lung_regions, neg_lung_regions, lesion_name, num_option)
    
    # location_qa의 정답 zone들을 severity_measurement_qa에 전달
    location_answer_zones = attribute_extraction_qa['location']['answer_zones'] if attribute_extraction_qa['location'] else None
    attribute_extraction_qa['severity/measurement'] = generate_severity_measurement_qa(overlaps, lesion_name, location_answer_zones, num_option)
    attribute_extraction_qa['comparison'] = generate_comparison_qa(overlaps, lesion_name, num_option)
    
    return attribute_extraction_qa

def generate_localize_qa(lesion_name, anatomy_name, operation_to_anatomy_names, center_point, fake_points_expansion, fake_points_contraction, operation, revision, all_available_fake_points=None, selected_component_id=None, num_option=4):
    """
    Args:
        lesion_name: 병변 이름
        anatomy_name: 해부학적 위치 이름
        operation_to_anatomy_names: operation별 anatomy 이름 리스트
        center_point: 정답 좌표
        fake_points_expansion: 현재 component의 expansion fake points
        fake_points_contraction: 현재 component의 contraction fake points
        operation: 현재 operation ('expansion' 또는 'contraction')
        revision: 수정 타입
        all_available_fake_points: 모든 component들에서 사용 가능한 fake_points 리스트 (dict 형태: {component_id: {'expansion': [...], 'contraction': [...]}})
        selected_component_id: 현재 선택된 component ID (현재 component를 제외하기 위해 사용)
        num_option: 총 옵션 개수
    """
    
    # 텍스트 모드 QA 생성
    location_hierarchy = {
        'right upper zone lung': [
            'right medial lung & right upper zone lung',
            'right lateral lung & right upper zone lung',   
            'right peripheral lung & right upper zone lung',
        ],
        'right mid zone lung': [
            'right medial lung & right mid zone lung',
            'right lateral lung & right mid zone lung',
            'right peripheral lung & right mid zone lung',
        ],
        'right lung base': [
            'right medial lung & right lung base',
            'right lateral lung & right lung base',
            'right peripheral lung & right lung base',
            'right costophrenic angle',
        ],
        'left upper zone lung': [
            'left medial lung & left upper zone lung',
            'left lateral lung & left upper zone lung',
            'left peripheral lung & left upper zone lung',
        ],
        'left mid zone lung': [
            'left medial lung & left mid zone lung',
            'left lateral lung & left mid zone lung',
            'left peripheral lung & left mid zone lung',
        ],
        'left lung base': [
            'left medial lung & left lung base',
            'left lateral lung & left lung base',
            'left peripheral lung & left lung base',
            'left costophrenic angle',
        ],
    }
    
    if lesion_name != 'cardiomegaly':
        all_location_names = []

        for anatomy_names in location_hierarchy.values():
            all_location_names.extend(anatomy_names)
    else:
        all_location_names = [
            'right lung base',
            'left lung base',
            'right mid zone lung',
            'left mid zone lung',
            'right upper zone lung',
            'left upper zone lung',
        ]
    
    all_location_names = list(set(all_location_names) - set(operation_to_anatomy_names[operation]))
    
    # anatomy_name과 겹치지 않는 location만 선택
    # 1. peripheral과 lateral은 같은 zone에 있을 때만 겹치므로 함께 나오면 안됨
    # 2. costophrenic angle은 lung base의 peripheral, lateral과 겹치므로 함께 나오면 안됨
    
    def get_zone_from_anatomy(anatomy):
        """anatomy에서 zone 정보 추출 (upper zone, mid zone, lung base)"""
        if 'costophrenic angle' in anatomy:
            return 'lung base'
        elif 'upper zone' in anatomy:
            return 'upper zone'
        elif 'mid zone' in anatomy:
            return 'mid zone'
        elif 'lung base' in anatomy:
            return 'lung base'
        return None
    
    def get_side_from_anatomy(anatomy):
        """anatomy에서 side 정보 추출 (left, right)"""
        if 'left' in anatomy:
            return 'left'
        elif 'right' in anatomy:
            return 'right'
        return None
    
    filtered_location_names = []
    
    # anatomy_name의 zone과 side 정보
    anatomy_zone = get_zone_from_anatomy(anatomy_name)
    anatomy_side = get_side_from_anatomy(anatomy_name)
    # anatomy_name이 peripheral을 포함하는지 확인
    anatomy_has_peripheral = 'peripheral' in anatomy_name
    # anatomy_name이 lateral을 포함하는지 확인
    anatomy_has_lateral = 'lateral' in anatomy_name
    # anatomy_name이 costophrenic angle인지 확인
    anatomy_is_costophrenic = 'costophrenic angle' in anatomy_name
    
    for loc_name in all_location_names:
        loc_zone = get_zone_from_anatomy(loc_name)
        loc_side = get_side_from_anatomy(loc_name)
        
        # 같은 zone이고 같은 side일 때만 충돌 체크
        if anatomy_zone and loc_zone and anatomy_zone == loc_zone:
            if anatomy_side and loc_side and anatomy_side == loc_side:
                # peripheral과 lateral이 같은 zone, 같은 side에 있으면 함께 나오지 않도록 필터링
                if anatomy_has_peripheral and 'lateral' in loc_name:
                    continue
                if anatomy_has_lateral and 'peripheral' in loc_name:
                    continue
                
                # costophrenic angle과 lung base의 peripheral/lateral이 함께 나오지 않도록 필터링
                if anatomy_is_costophrenic:
                    if 'peripheral' in loc_name or 'lateral' in loc_name:
                        continue
                if anatomy_has_peripheral or anatomy_has_lateral:
                    if 'costophrenic angle' in loc_name:
                        continue
        
        filtered_location_names.append(loc_name)
    
    # 정답 하나는 무조건 넣고.... 어떻게 할 지 생각해야한다
    num_to_select = min(num_option - 1, len(filtered_location_names))
    negative_location = random.sample(filtered_location_names, num_to_select) if num_to_select > 0 else []
    
    answer_text_options = [anatomy_name] + negative_location
    
    random.shuffle(answer_text_options)
    
    answer_text_index = answer_text_options.index(anatomy_name) + 1
    
    # Build question with text options
    text_options_processed = [preprocess_anatomy_name(opt) for opt in answer_text_options]
    text_options_text = "\n".join([f"({i}) {opt}" for i, opt in enumerate(text_options_processed, 1)])
    question_with_text_options = f"In which anatomical region does the mask need {revision}?\n\nOptions:\n{text_options_text}"

    # 좌표 모드 QA 생성
    # fake_points_expansion과 fake_points_contraction은 리스트 형태 [(x, y), ...]
    fake_points_expansion = fake_points_expansion if fake_points_expansion else []
    fake_points_contraction = fake_points_contraction if fake_points_contraction else []
    
    # 정답 외에 num_option - 1개의 fake points 선택
    num_fake_needed = num_option - 1
    fake_points_list = []
    
    # 다른 component의 point 사용 여부 추적
    used_other_component_point = False
    used_default_point = False
    
    # operation에 따라 우선순위 결정
    if operation == 'contraction':
        priority_fake_points = fake_points_expansion.copy()
        secondary_fake_points = fake_points_contraction.copy()
    elif operation == 'expansion':
        priority_fake_points = fake_points_contraction.copy()
        secondary_fake_points = fake_points_expansion.copy()
    else:
        # operation이 없으면 둘 다 동등하게 처리
        all_fake_points = fake_points_expansion + fake_points_contraction
        priority_fake_points = all_fake_points.copy()
        secondary_fake_points = []
    
    # 현재 component의 fake points가 충분한지 확인
    if len(priority_fake_points) + len(secondary_fake_points) >= num_fake_needed:
        # 충분하면 현재 것만 랜덤 선택하되, operation 기반으로 우선 선택
        # 1) priority에서 최대한 채우기
        if len(priority_fake_points) >= num_fake_needed:
            fake_points_list = random.sample(priority_fake_points, num_fake_needed)
        else:
            fake_points_list = priority_fake_points.copy()
            remaining = num_fake_needed - len(fake_points_list)
            if remaining > 0 and len(secondary_fake_points) > 0:
                num_to_add = min(remaining, len(secondary_fake_points))
                fake_points_list.extend(random.sample(secondary_fake_points, num_to_add))
    else:
        # 현재 것 모두 사용
        fake_points_list = priority_fake_points + secondary_fake_points
        
        # 다른 component의 fake_points에서 추가로 가져오기
        if all_available_fake_points:
            # all_available_fake_points의 구조: {component_id: {'expansion': [...], 'contraction': [...]}}
            # 현재 component는 제외
            all_other_fake_points_expansion = []
            all_other_fake_points_contraction = []
            for component_id, points_dict in all_available_fake_points.items():
                # 현재 component는 제외
                if component_id != selected_component_id:
                    if 'expansion' in points_dict:
                        all_other_fake_points_expansion.extend(points_dict['expansion'])
                    if 'contraction' in points_dict:
                        all_other_fake_points_contraction.extend(points_dict['contraction'])
            
            # operation에 따라 다른 component fake도 우선순위 적용
            if operation == 'contraction':
                priority_other_points = all_other_fake_points_expansion
                secondary_other_points = all_other_fake_points_contraction
            elif operation == 'expansion':
                priority_other_points = all_other_fake_points_contraction
                secondary_other_points = all_other_fake_points_expansion
            else:
                priority_other_points = all_other_fake_points_expansion + all_other_fake_points_contraction
                secondary_other_points = []
            
            # 중복 제거 (이미 사용한 것 제외)
            remaining_priority = [fp for fp in priority_other_points if fp not in fake_points_list]
            remaining_secondary = [fp for fp in secondary_other_points if fp not in fake_points_list]
            
            # 부족한 만큼 추가해야 하는 개수
            num_additional_needed = num_fake_needed - len(fake_points_list)
            if num_additional_needed > 0:
                additional_fake_points = []
                if len(remaining_priority) >= num_additional_needed:
                    additional_fake_points = random.sample(remaining_priority, num_additional_needed)
                else:
                    additional_fake_points = remaining_priority.copy()
                    still_needed = num_additional_needed - len(additional_fake_points)
                    if still_needed > 0 and len(remaining_secondary) > 0:
                        num_to_add = min(still_needed, len(remaining_secondary))
                        additional_fake_points.extend(random.sample(remaining_secondary, num_to_add))
                
                if additional_fake_points:
                    fake_points_list.extend(additional_fake_points)
                    used_other_component_point = True
    
    # 여전히 부족하면 default points 생성 (이미지 정중앙으로 고정)
    if len(fake_points_list) < num_fake_needed:
        needed = num_fake_needed - len(fake_points_list)
        # default points: 이미지 정중앙 (1024x1024 기준으로 (512, 512))
        default_point = (512, 512)
        default_points = [default_point] * needed
        fake_points_list.extend(default_points)
        used_default_point = True
    
    # 정답과 fake points를 합쳐서 옵션 생성
    answer_point_options = [center_point] + fake_points_list
    
    # 섞기
    random.shuffle(answer_point_options)
    
    # 정답 인덱스 찾기
    answer_point_index = answer_point_options.index(center_point) + 1
    
    # 텍스트 모드와 좌표 모드를 모두 포함한 하나의 딕셔너리로 반환
    localize_qa = {
        'question': question_with_text_options,
        'answer_text_options': answer_text_options,
        'answer_point_options': answer_point_options,
        'answer_text': anatomy_name,
        'answer_text_index': answer_text_index,
        'answer_point': center_point,
        'answer_point_index': answer_point_index,
        'used_other_component_point': used_other_component_point,
        'used_default_point': used_default_point,
    }
    
    
    return localize_qa

def generate_revision_qa(lesion_name, deformation, operation, all_available_fake_masks=None, all_component_ids=None, num_option=4):
    """
    Args:
        deformation: 현재 component의 deformation 데이터
        all_available_fake_masks: 다른 component들에서 사용 가능한 fake_mask 리스트 (dict 형태: {component_id: [fake_mask_paths]})
        all_component_ids: 모든 component ID 집합 (fallback용)
        num_option: 총 옵션 개수
    """
    
    gt_mask = deformation['revision_flow']['after revision']
    selected_component_id = deformation.get('suboptimal_component_id')
    
    # 현재 component의 fake_masks 수집
    current_fake_masks = []
    if 'fake_masks' in deformation and deformation['fake_masks']:
        current_fake_masks = [fm['fake_mask'] for fm in deformation['fake_masks']]
    
    # 필요한 fake_mask 개수 계산
    num_fake_needed = num_option - 1
    
    # 다른 component의 fake_mask 사용 여부 추적
    used_other_component_fake = False
    used_default_fake = False
    
    # 현재 component의 fake_mask가 부족하면 다른 component의 fake_mask 추가
    
    fake_masks_list = []
    if len(current_fake_masks) >= num_fake_needed:
        # 충분하면 현재 것만 랜덤 선택하되, operation 기반으로 정렬된 current_fake_masks에서 우선 선택
        fake_masks_list = []
        # 먼저 priority 부분에서 뽑고, 부족하면 나머지에서 보충

        if operation == 'expansion':
            priority_masks = [m for m in current_fake_masks if 'fake_contraction' in m]
            secondary_masks = [m for m in current_fake_masks if 'fake_expansion' in m]
        elif operation == 'contraction':
            priority_masks = [m for m in current_fake_masks if 'fake_expansion' in m]
            secondary_masks = [m for m in current_fake_masks if 'fake_contraction' in m]
        
        # 1) priority에서 최대한 채우기
        if len(priority_masks) >= num_fake_needed:
            fake_masks_list = random.sample(priority_masks, num_fake_needed)
        else:
            fake_masks_list = priority_masks.copy()
            remaining = num_fake_needed - len(fake_masks_list)
            if remaining > 0 and len(secondary_masks) > 0:
                num_to_add = min(remaining, len(secondary_masks))
                fake_masks_list.extend(random.sample(secondary_masks, num_to_add))
    else:
        # 현재 것 모두 사용 (이미 operation 기반으로 정렬되어 있음)
        fake_masks_list = current_fake_masks.copy()
        
        # 다른 component의 fake_mask에서 추가로 가져오기
        if all_available_fake_masks:
            # 모든 component의 fake_mask를 평탄화
            all_other_fake_masks = []
            for component_id, fake_mask_paths in all_available_fake_masks.items():
                # 현재 component는 제외
                if component_id != deformation.get('suboptimal_component_id'):
                    all_other_fake_masks.extend(fake_mask_paths)
            
            # 중복 제거 (이미 사용한 것 제외)
            remaining_fake_masks = [fm for fm in all_other_fake_masks if fm not in fake_masks_list]
            
            # 부족한 만큼 추가해야 하는 개수
            num_additional_needed = num_fake_needed - len(fake_masks_list)
            if num_additional_needed > 0 and len(remaining_fake_masks) > 0:
                # operation에 따라 다른 component fake도 우선순위 적용
                if operation == 'expansion':
                    # expansion round에서는 contraction 계열 fake를 먼저 보이게
                    priority_masks = [m for m in remaining_fake_masks if 'fake_contraction' in m]
                    secondary_masks = [m for m in remaining_fake_masks if m not in priority_masks]
                elif operation == 'contraction':
                    # contraction round에서는 expansion 계열 fake를 먼저 보이게
                    priority_masks = [m for m in remaining_fake_masks if 'fake_expansion' in m]
                    secondary_masks = [m for m in remaining_fake_masks if m not in priority_masks]
                else:
                    priority_masks = remaining_fake_masks
                    secondary_masks = []
                
                additional_fake_masks = []
                if len(priority_masks) >= num_additional_needed:
                    additional_fake_masks = random.sample(priority_masks, num_additional_needed)
                else:
                    additional_fake_masks = priority_masks.copy()
                    still_needed = num_additional_needed - len(additional_fake_masks)
                    if still_needed > 0 and len(secondary_masks) > 0:
                        num_to_add = min(still_needed, len(secondary_masks))
                        additional_fake_masks.extend(random.sample(secondary_masks, num_to_add))
                
                if additional_fake_masks:
                    fake_masks_list.extend(additional_fake_masks)
                    used_other_component_fake = True
    
    # 여전히 부족하면 기본값 사용 (하위 호환성)
    if len(fake_masks_list) < num_fake_needed:
        if lesion_name == 'cardiomegaly':
            default_fake_masks = ['default_keep', 'default_dilated', 'default_eroded']
        else:
            if 'right' in deformation['anatomy']:
                default_fake_masks = ['default_keep', 'default_right_lung_chex', 'default_right_lung_cxas']
            else:
                default_fake_masks = ['default_keep', 'default_left_lung_chex', 'default_left_lung_cxas']
        
        needed = num_fake_needed - len(fake_masks_list)
        
        if needed > 0:
            # default 후보들 중에서 랜덤으로 선택
            num_to_add = min(needed, len(default_fake_masks))
            fake_masks_list.extend(random.sample(default_fake_masks, num_to_add))
            used_default_fake = True
    
    # fake_mask가 어떤 component에 속하는지 매핑 생성
    fake_mask_to_component_id = {}
    # 현재 component의 fake_mask
    for fake_mask in current_fake_masks:
        fake_mask_to_component_id[fake_mask] = selected_component_id
    # 다른 component의 fake_mask
    if all_available_fake_masks:
        for component_id, fake_mask_paths in all_available_fake_masks.items():
            for fake_mask in fake_mask_paths:
                fake_mask_to_component_id[fake_mask] = component_id
    
    # 정답 mask와 fake mask들을 합쳐서 섞기 (임시 리스트)
    answer_options_temp = [{'mask': gt_mask, 'is_answer': True}]
    for fake_mask in fake_masks_list[:num_fake_needed]:
        answer_options_temp.append({'mask': fake_mask, 'is_answer': False})
    
    random.shuffle(answer_options_temp)
    
    # answer_index 찾기 및 relative_path 추가
    answer_index = None
    answer_options = []
    option_component_states = {}
    
    # 각 option에 대한 component state 정보 생성
    all_component_states = deformation.get('all_component_mask_states', {})
    
    # all_component_states가 비어있으면 all_component_ids를 사용해서 초기화
    if not all_component_states and all_component_ids:
        all_component_states = {comp_id: comp_id for comp_id in all_component_ids}  # 초기 상태는 suboptimal_component
    
    for opt_idx, option_data in enumerate(answer_options_temp, 1):
        mask_path = option_data['mask']
        is_answer = option_data['is_answer']
        
        # answer_index 찾기
        if is_answer:
            answer_index = opt_idx
            relative_path = f'option_{opt_idx}_mask_answer.png'
        else:
            relative_path = f'option_{opt_idx}_mask_fake.png'
        
        # answer_options에 딕셔너리 형태로 추가
        answer_options.append({
            'mask': mask_path,
            'relative_path': relative_path
        })
        
        # 각 option에 대해 component state 생성
        if all_component_states and selected_component_id:
            # 모든 component의 상태를 복사
            option_states = all_component_states.copy()
            
            # mask_path가 정답이면 selected_component_id 업데이트
            if mask_path == gt_mask:
                option_states[selected_component_id] = mask_path
            else:
                # fake_mask인 경우, 어떤 component에 속하는지 확인
                if mask_path in fake_mask_to_component_id:
                    # 해당 component의 상태를 업데이트
                    fake_mask_component_id = fake_mask_to_component_id[mask_path]
                    option_states[fake_mask_component_id] = mask_path
                else:
                    # default fake_mask인 경우, selected_component_id에 할당 (fallback)
                    option_states[selected_component_id] = mask_path
            
            option_component_states[opt_idx] = option_states
        elif selected_component_id:
            # fallback: selected_component만 포함하되, all_component_ids가 있으면 모든 component 포함
            if all_component_ids:
                option_states = {comp_id: comp_id for comp_id in all_component_ids}  # 초기 상태
                # mask_path가 정답이면 selected_component_id 업데이트
                if mask_path == gt_mask:
                    option_states[selected_component_id] = mask_path
                else:
                    # fake_mask인 경우, 어떤 component에 속하는지 확인
                    if mask_path in fake_mask_to_component_id:
                        fake_mask_component_id = fake_mask_to_component_id[mask_path]
                        option_states[fake_mask_component_id] = mask_path
                    else:
                        option_states[selected_component_id] = mask_path
                option_component_states[opt_idx] = option_states
            else:
                option_component_states[opt_idx] = {selected_component_id: mask_path}
        else:
            option_component_states[opt_idx] = {}
    
    # Build question with options
    options_text = "\n".join([f"({i}) Mask option {i}" for i in range(1, len(answer_options) + 1)])
    question = f"Choose the mask that reflects the modifications.\n\nOptions:\n{options_text}"
    
    revision_qa = {
        'question': question,
        'answer_options': answer_options,  # 이제 딕셔너리 리스트: [{'mask': '...', 'relative_path': '...'}, ...]
        'answer': gt_mask,
        'answer_index': answer_index,
        'answer_mask_path': gt_mask,
        'fake_mask_paths': fake_masks_list[:num_fake_needed],
        'used_other_component_fake': used_other_component_fake,
        'used_default_fake': used_default_fake,
        'option_component_states': option_component_states,  # 각 option별 component state 정보
        'selected_component_id': selected_component_id,
        'all_component_states_before': all_component_states,  # 수정 전 모든 component 상태
    }
    
    return revision_qa

def generate_initial_qa(revision, final=False, num_option=4):
    
    answer_options = [
        'expansion',
        'contraction',
        'no more modifications needed',
    ]
    
    # Build question with options
    options_text = "\n".join([f"({i}) {opt}" for i, opt in enumerate(answer_options, 1)])
    question = f"Does the segmentation mask need to be modified? If so, choose between expansion and contraction. If both are needed, choose expansion first.\n\nOptions:\n{options_text}"
    
    initial_qa = {
        'question': question,
        'answer_options': answer_options,
        'answer': 'no more modifications needed' if final else revision,
        'answer_index': 3 if final else (1 if revision == 'expansion' else 2),
    }
    
    return initial_qa

def generate_deformation_qa(qa_deformation_results, geometrical_mask_infos, lesion_name, num_option=4):
    
    operation_to_anatomy_names = {
        'expansion': [],
        'contraction': [],
    }
    
    # 모든 component의 fake_mask 수집
    all_available_fake_masks = {}
    for operation, deformation_results in qa_deformation_results.items():
        for suboptimal_component_id, deformations in deformation_results.items():
            component_fake_masks = []
            for deformation in deformations:
                if 'fake_masks' in deformation and deformation['fake_masks']:
                    component_fake_masks.extend([fm['fake_mask'] for fm in deformation['fake_masks']])
            if component_fake_masks:
                all_available_fake_masks[suboptimal_component_id] = component_fake_masks
    
    # 모든 component의 fake_points 수집
    all_available_fake_points = {}
    for operation, deformation_results in qa_deformation_results.items():
        for suboptimal_component_id, deformations in deformation_results.items():
            component_fake_points_expansion = []
            component_fake_points_contraction = []
            for deformation in deformations:
                if 'fake_points_expansion' in deformation and deformation['fake_points_expansion']:
                    component_fake_points_expansion.extend(deformation['fake_points_expansion'])
                if 'fake_points_contraction' in deformation and deformation['fake_points_contraction']:
                    component_fake_points_contraction.extend(deformation['fake_points_contraction'])
            if component_fake_points_expansion or component_fake_points_contraction:
                all_available_fake_points[suboptimal_component_id] = {
                    'expansion': component_fake_points_expansion,
                    'contraction': component_fake_points_contraction
                }
    
    # 모든 component ID 수집
    all_component_ids = []
    for i in geometrical_mask_infos:
        all_component_ids.append(f"suboptimal_component_{i['mask_component_id']}")
        
    num_deformations = 0
    for operation, deformation_results in qa_deformation_results.items():
        for suboptimal_component_id, deformations in deformation_results.items():
            for deformation in deformations:
                anatomy_name = deformation['anatomy']
                operation_to_anatomy_names[operation].append(anatomy_name)
                num_deformations += 1
    
    deformation_qa_sequence = {}
    
    for i in range(num_deformations + 1):
        deformation_qa_sequence[i+1] = None
    
    # 각 component의 현재 mask 상태 추적 (round별로 업데이트)
    component_mask_states = {}
    for component_id in all_component_ids:
        
        if component_id in qa_deformation_results['contraction']:
            component_mask_states[component_id] = qa_deformation_results['contraction'][component_id][-1]['revision_flow']['before revision']
        elif component_id in qa_deformation_results['expansion']:
            component_mask_states[component_id] = qa_deformation_results['expansion'][component_id][-1]['revision_flow']['before revision']
        else:
            component_mask_states[component_id] = component_id.split('sub')[1]

    idx = 0
    
    for operation in ['contraction', 'expansion']: # 아무런 deformation 없는 경우 생성도 고려해야함!!!!!!
        
        i = qa_deformation_results[operation]
        ids = list(i.keys())
        
        id_to_defomation_number = {}
        for suboptimal_component_id, deformation in i.items():
            id_to_defomation_number[suboptimal_component_id] = len(deformation)
        
        while sum(list(id_to_defomation_number.values())) != 0:
            
            selected_id = random.choice(ids)
            selected_deformation = i[selected_id][-1]
            i[selected_id] = i[selected_id][:-1]
            id_to_defomation_number[selected_id] -= 1
            
            if id_to_defomation_number[selected_id] == 0:
                ids.remove(selected_id)
            
            anatomy_name = selected_deformation['anatomy']
            center_point = selected_deformation['center_point']
            fake_points_expansion = selected_deformation['fake_points_expansion']
            fake_points_contraction = selected_deformation['fake_points_contraction']
            revision = selected_deformation['revision']
            
            # suboptimal_component_id를 deformation에 추가 (revision_qa에서 사용)
            selected_deformation['suboptimal_component_id'] = selected_id
            
            # 현재 round의 모든 component mask 상태 저장
            selected_deformation['all_component_mask_states'] = component_mask_states.copy()
            
            # 선택된 component의 mask 상태 업데이트 (after revision)
            after_revision_mask = selected_deformation['revision_flow']['after revision']
            component_mask_states[selected_id] = after_revision_mask

            initial_qa = generate_initial_qa(revision)
            localize_qa = generate_localize_qa(lesion_name, anatomy_name, operation_to_anatomy_names, center_point, fake_points_expansion, fake_points_contraction, operation, revision, all_available_fake_points, selected_id)
            revision_qa = generate_revision_qa(lesion_name,selected_deformation, operation, all_available_fake_masks, all_component_ids, num_option)
            
            selected_deformation['initial_qa'] = initial_qa
            selected_deformation['localize_qa'] = localize_qa
            selected_deformation['revision_qa'] = revision_qa
            deformation_qa_sequence[idx+1] = selected_deformation
            
            idx += 1

    # 마무리 질문
    final_round_data = {
        'initial_qa': generate_initial_qa(None, final=True),
        'all_component_mask_states': component_mask_states.copy(),  # 최종 모든 component 상태 저장
    }
    deformation_qa_sequence[idx+1] = final_round_data
    
    return deformation_qa_sequence

def generate_contour_qa(key_id, qa_deformation_results, geometrical_mask_infos, lesion_name, no_deformation_ratio=0.5, no_deformation_override=None):
    
    import random
    import math
    
    gt_points_expansion = []
    gt_points_contraction = []
    gt_anatomy_expansion = []
    gt_anatomy_contraction = []
    fake_points_expansion = []
    fake_points_contraction = []

    all_component_ids = set(list(qa_deformation_results['expansion'].keys()) + list(qa_deformation_results['contraction'].keys()))

    no_deformation_component_ids = set(list(qa_deformation_results['no_deformation'].keys()))

    revision_after_masks = []
    revision_before_masks = []

    all_fake_masks = {}
    all_true_masks = {}
    component_anatomies = {}  # component별 anatomy 정보 저장

    for component_id in list(all_component_ids):
        if len(qa_deformation_results['expansion']) > 0 and component_id in qa_deformation_results['expansion']:
            revision_after_mask = qa_deformation_results['expansion'][component_id][0]['revision_flow']['after revision']
            revision_after_masks.append(revision_after_mask)

            if component_id not in all_true_masks:
                all_true_masks[component_id] = []
            all_true_masks[component_id].append(revision_after_mask)

        elif len(qa_deformation_results['contraction']) > 0 and component_id in qa_deformation_results['contraction']:
            if component_id in qa_deformation_results['contraction']:
                revision_after_mask = qa_deformation_results['contraction'][component_id][0]['revision_flow']['after revision']
                revision_after_masks.append(revision_after_mask)
            
                if component_id not in all_true_masks:
                    all_true_masks[component_id] = []
                all_true_masks[component_id].append(revision_after_mask)
        else:
            raise ValueError("No deformation results found")

        if len(qa_deformation_results['contraction']) > 0 and component_id in qa_deformation_results['contraction']:
            if component_id in qa_deformation_results['contraction']:   
                revision_before_mask = qa_deformation_results['contraction'][component_id][-1]['revision_flow']['before revision']
                revision_before_masks.append(revision_before_mask)
        elif len(qa_deformation_results['expansion']) > 0 and component_id in qa_deformation_results['expansion']:
            if component_id in qa_deformation_results['expansion']:
                revision_before_mask = qa_deformation_results['expansion'][component_id][-1]['revision_flow']['before revision']
                revision_before_masks.append(revision_before_mask)
        else:
            raise ValueError("No deformation results found")

    for component_id in no_deformation_component_ids:
        revision_before_masks.append(qa_deformation_results['no_deformation'][component_id][0]['mask_path'])
        revision_after_masks.append(qa_deformation_results['no_deformation'][component_id][0]['mask_path'])
        all_true_masks[component_id] = [qa_deformation_results['no_deformation'][component_id][0]['mask_path']]

    for operation, deformation_results in qa_deformation_results.items():
        
        if operation == 'no_deformation':
            for idx, (suboptimal_component_id, deformations) in enumerate(deformation_results.items()):
                geometrical_mask_info = geometrical_mask_infos[int(suboptimal_component_id.split('_')[-1])]

                if geometrical_mask_info['overlap']['left lung']['overlap_ratio'] > 0.0:
                    component_anatomies[suboptimal_component_id] = 'left lung'
                elif geometrical_mask_info['overlap']['right lung']['overlap_ratio'] > 0.0:
                    component_anatomies[suboptimal_component_id] = 'right lung'
                else:
                    print(f"Invalid anatomy: {key_id}")
                    raise ValueError(f"Invalid anatomy: {geometrical_mask_info}")

            continue
        
        if len(deformation_results) == 0:
            continue
        
        for idx, (suboptimal_component_id, deformations) in enumerate(deformation_results.items()):

            for deformation in deformations:

                if deformation['revision'] == 'expansion':
                    gt_points_expansion.append(deformation['center_point'])
                    gt_anatomy_expansion.append(deformation['anatomy'])
                else:
                    gt_points_contraction.append(deformation['center_point'])
                    gt_anatomy_contraction.append(deformation['anatomy'])

                if suboptimal_component_id not in all_fake_masks:
                    all_fake_masks[suboptimal_component_id] = []

                # Anatomy 정보 저장
                if suboptimal_component_id not in component_anatomies:
                    component_anatomies[suboptimal_component_id] = deformation.get('anatomy', '')
                
                for mask in deformation['fake_masks']:
                    all_fake_masks[suboptimal_component_id].append(mask['fake_mask'])

                all_fake_masks[suboptimal_component_id].append(deformation['mask_path'])

                fake_points_expansion.extend(deformation['fake_points_expansion'])
                fake_points_contraction.extend(deformation['fake_points_contraction'])

    # GT points 합치기
    all_gt_points = gt_points_expansion + gt_points_contraction
    all_fake_points = fake_points_expansion + fake_points_contraction
    
    # Fake points 중 GT points와 거리 100 이상인 것만 필터링
    def calculate_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    filtered_fake_points = []
    for fake_point in all_fake_points:
        is_far_enough = True
        
        # GT points와의 거리 체크
        for gt_point in all_gt_points:
            if calculate_distance(fake_point, gt_point) < 100:
                is_far_enough = False
                break
        
        # 이미 선택된 fake points와의 거리 체크
        if is_far_enough:
            for selected_fake in filtered_fake_points:
                if calculate_distance(fake_point, selected_fake) < 100:
                    is_far_enough = False
                    break
        
        if is_far_enough:
            filtered_fake_points.append(fake_point)
    
    # 색상 리스트 정의
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown']
    max_points = len(colors)  # 최대 10개
    
    # GT points와 filtered fake points 합치기
    # 전체 포인트가 10개를 넘으면 GT를 포함해서 최대 10개만 선택
    if len(all_gt_points) + len(filtered_fake_points) > max_points:
        # GT points는 모두 포함
        num_fake_to_keep = max_points - len(all_gt_points)
        if num_fake_to_keep > 0:
            # Fake points 중 랜덤하게 선택
            filtered_fake_points = random.sample(filtered_fake_points, num_fake_to_keep)
        else:
            # GT points만으로도 10개를 넘으면 GT만 사용
            filtered_fake_points = []
    
    # GT points와 filtered fake points를 섞어서 색 배정
    all_points = all_gt_points + filtered_fake_points
    random.shuffle(all_points)
    
    # 각 포인트에 색상 배정
    point_colors = {}
    for i, point in enumerate(all_points):
        color = colors[i % len(colors)]  # 색상이 부족하면 순환
        point_colors[tuple(point)] = color
    
    # Answer options 생성 (모든 포인트와 색상) - 두 질문에서 공통으로 사용
    answer_options = []
    for point, color in point_colors.items():
        answer_options.append({
            'point': list(point),
            'color': color
        })
    
    # None 옵션 추가
    answer_options.append({
        'point': None,
        'color': 'None'
    })
    
    # Expansion QA 생성
    expansion_colors = [point_colors[tuple(pt)] for pt in gt_points_expansion]
    if expansion_colors:
        expansion_answer = ', '.join([f"{color} point" for color in expansion_colors])
        expansion_answer_indices = []
        for gt_point in gt_points_expansion:
            for idx, option in enumerate(answer_options):
                if option['point'] is not None and tuple(option['point']) == tuple(gt_point):
                    expansion_answer_indices.append(idx + 1)  # 1부터 시작
                    break
    else:
        expansion_answer = "None"
        expansion_answer_indices = [len(answer_options)]  # None 옵션의 인덱스 (1-based)
    
    # Contraction QA 생성
    contraction_colors = [point_colors[tuple(pt)] for pt in gt_points_contraction]
    if contraction_colors:
        contraction_answer = ', '.join([f"{color} point" for color in contraction_colors])
        contraction_answer_indices = []
        for gt_point in gt_points_contraction:
            for idx, option in enumerate(answer_options):
                if option['point'] is not None and tuple(option['point']) == tuple(gt_point):
                    contraction_answer_indices.append(idx + 1)  # 1부터 시작
                    break
    else:
        contraction_answer = "None"
        contraction_answer_indices = [len(answer_options)]  # None 옵션의 인덱스 (1-based)
    
    # Component별 true/fake mask 조합 생성
    
    # 각 component에 대해 true mask 또는 fake masks 중 하나를 선택하는 모든 조합 생성
    component_ids = sorted(all_true_masks.keys())
    
    # 각 component의 선택 가능한 mask 리스트 생성

    right_default = False
    left_default = False

    component_mask_choices = []
    for comp_id in component_ids:
        choices = []
        # True mask 추가
        if comp_id in all_true_masks and len(all_true_masks[comp_id]) > 0:
            choices.append(('true', all_true_masks[comp_id][0]))
        
        # Fake masks 추가
        fake_masks_for_comp = []
        if comp_id in all_fake_masks:
            fake_masks_for_comp = all_fake_masks[comp_id]
        
        # Fake mask가 부족하면 default mask 추가
        #num_fake_needed = 3  # 최소 3개의 fake mask 필요
        #if len(fake_masks_for_comp) < num_fake_needed:
            # Default fake masks 정의
        if lesion_name == 'cardiomegaly':
            default_fake_masks = ['default_keep', 'default_dilated', 'default_eroded']
        else:
            # Component의 anatomy 정보 확인
            anatomy = component_anatomies.get(comp_id, '')
            
            # Anatomy에 'right'가 포함되어 있으면 right lung default masks 사용
            if 'right' in anatomy.lower():
                if not right_default:
                    default_fake_masks = ['default_keep', 'default_right_lung_chex', 'default_right_lung_cxas']
                    right_default = True
                else:
                    default_fake_masks = ['default_keep']
            # 'left'가 포함되어 있으면 left lung default masks 사용
            elif 'left' in anatomy.lower():
                if not left_default:
                    default_fake_masks = ['default_keep', 'default_left_lung_chex', 'default_left_lung_cxas']
                    left_default = True
                else:
                    default_fake_masks = ['default_keep']
            else:
                # Anatomy 정보가 없으면 모든 default masks 사용
                raise ValueError(f"Invalid anatomy: {anatomy}")
                #default_fake_masks = ['default_keep', 'default_right_lung_chex', 'default_right_lung_cxas', 'default_left_lung_chex', 'default_left_lung_cxas']
            
            #needed = num_fake_needed - len(fake_masks_for_comp)
            # 이미 사용된 fake mask는 제외
        #available_defaults = [m for m in default_fake_masks if m not in fake_masks_for_comp]
        fake_masks_for_comp = fake_masks_for_comp + default_fake_masks
        #if needed > 0 and available_defaults:
        #     num_to_add = min(needed, len(available_defaults))
        #    fake_masks_for_comp = list(fake_masks_for_comp) + random.sample(available_defaults, num_to_add)
        
        # Fake masks를 choices에 추가
        for fake_mask in fake_masks_for_comp:
            choices.append(('fake', fake_mask))
        
        component_mask_choices.append(choices)
    
    # 모든 조합 생성 (각 component에서 하나씩 선택)
    all_combinations = list(itertools.product(*component_mask_choices))
    
    # 조합을 true/fake로 분류
    fake_combinations = []  # 하나라도 fake가 포함된 조합
    true_combination = None  # 모두 true인 조합
    
    for combination in all_combinations:
        # combination: [('true', mask1), ('fake', mask2), ...]
        has_fake = any(choice[0] == 'fake' for choice in combination)
        
        if has_fake:
            # 하나라도 fake면 fake 조합
            masks = [choice[1] for choice in combination]
            if set(masks) != set(revision_before_masks):
                fake_combinations.append(masks)
            else:
                continue
        else:
            # 모두 true면 true 조합
            masks = [choice[1] for choice in combination]
            true_combination = masks
    
    # Fake 조합 중 랜덤으로 3개 선택 (default mask가 적은 조합 우선)
    if len(fake_combinations) > 3:
        fake_combinations.sort(key=lambda combo: sum(1 for m in combo if str(m).startswith('default_')))
        grouped = {}
        for combo in fake_combinations:
            cnt = sum(1 for m in combo if str(m).startswith('default_'))
            grouped.setdefault(cnt, []).append(combo)
        selected_fake_combinations = []
        for cnt in sorted(grouped.keys()):
            needed = 3 - len(selected_fake_combinations)
            if needed <= 0:
                break
            group = grouped[cnt]
            if len(group) <= needed:
                selected_fake_combinations.extend(group)
            else:
                selected_fake_combinations.extend(random.sample(group, needed))
    else:
        selected_fake_combinations = fake_combinations
    
    # Revision result QA 생성 (true combination + fake combinations)
    revision_result_options_temp = []
    
    # True combination 추가 (정답)
    if true_combination:
        revision_result_options_temp.append({
            'type': 'true',
            'masks': true_combination
        })
    
    # Fake combinations 추가 (오답들)
    for fake_combo in selected_fake_combinations:
        revision_result_options_temp.append({
            'type': 'fake',
            'masks': fake_combo
        })
    
    # 옵션들을 섞기
    random.shuffle(revision_result_options_temp)
    
    # 정답 인덱스 찾기 및 relative_path 추가
    revision_result_answer_index = None
    revision_result_options = []
    
    for idx, option in enumerate(revision_result_options_temp, 1):
        is_answer = (option['type'] == 'true')
        if is_answer:
            revision_result_answer_index = idx  # 1-based
            relative_path = f'option_{idx}_mask_answer.png'
        else:
            relative_path = f'option_{idx}_mask_fake.png'
        
        revision_result_options.append({
            'type': option['type'],
            'masks': option['masks'],
            'relative_path': relative_path  # chexpercept에 저장될 상대 경로
        })
    
    # 각 포인트에 대한 anatomy 위치 QA 생성


    '''
    point_anatomy_qas = []
    
    # Helper function to get zone from anatomy
    def get_zone_from_anatomy(anatomy):
        """anatomy에서 zone 정보 추출 (upper zone, mid zone, lung base)"""
        if 'costophrenic angle' in anatomy:
            return 'lung base'
        elif 'upper zone' in anatomy:
            return 'upper zone'
        elif 'mid zone' in anatomy:
            return 'mid zone'
        elif 'lung base' in anatomy:
            return 'lung base'
        return None
    
    def get_side_from_anatomy(anatomy):
        """anatomy에서 side 정보 추출 (left, right)"""
        if 'left' in anatomy:
            return 'left'
        elif 'right' in anatomy:
            return 'right'
        return None
    
    def filter_fake_anatomies(correct_anatomy, all_anatomies):
        """겹치는 anatomy를 제외하고 fake anatomy 3개 선택"""
        anatomy_zone = get_zone_from_anatomy(correct_anatomy)
        anatomy_side = get_side_from_anatomy(correct_anatomy)
        anatomy_has_peripheral = 'peripheral' in correct_anatomy
        anatomy_has_lateral = 'lateral' in correct_anatomy
        anatomy_is_costophrenic = 'costophrenic angle' in correct_anatomy
        
        filtered_anatomies = []
        
        for candidate in all_anatomies:
            if candidate == correct_anatomy:
                continue
                
            loc_zone = get_zone_from_anatomy(candidate)
            loc_side = get_side_from_anatomy(candidate)
            
            # 같은 zone이고 같은 side일 때만 충돌 체크
            if anatomy_zone and loc_zone and anatomy_zone == loc_zone:
                if anatomy_side and loc_side and anatomy_side == loc_side:
                    # peripheral과 lateral이 같은 zone, 같은 side에 있으면 함께 나오지 않도록 필터링
                    if anatomy_has_peripheral and 'lateral' in candidate:
                        continue
                    if anatomy_has_lateral and 'peripheral' in candidate:
                        continue
                    
                    # costophrenic angle과 lung base의 peripheral/lateral이 함께 나오지 않도록 필터링
                    if anatomy_is_costophrenic:
                        if 'peripheral' in candidate or 'lateral' in candidate:
                            continue
                    if anatomy_has_peripheral or anatomy_has_lateral:
                        if 'costophrenic angle' in candidate:
                            continue
            
            filtered_anatomies.append(candidate)
        
        return filtered_anatomies
    
    # 모든 가능한 anatomy 리스트 정의
    if lesion_name != 'cardiomegaly':
        all_anatomies = [
            'right medial lung & right upper zone lung',
            'right lateral lung & right upper zone lung',
            'right peripheral lung & right upper zone lung',
            'right medial lung & right mid zone lung',
            'right lateral lung & right mid zone lung',
            'right peripheral lung & right mid zone lung',
            'right medial lung & right lung base',
            'right lateral lung & right lung base',
            'right peripheral lung & right lung base',
            'right costophrenic angle',
            'left medial lung & left upper zone lung',
            'left lateral lung & left upper zone lung',
            'left peripheral lung & left upper zone lung',
            'left medial lung & left mid zone lung',
            'left lateral lung & left mid zone lung',
            'left peripheral lung & left mid zone lung',
            'left medial lung & left lung base',
            'left lateral lung & left lung base',
            'left peripheral lung & left lung base',
            'left costophrenic angle',
        ]
    else:
        all_anatomies = [
            'right lung base',
            'left lung base',
            'right mid zone lung',
            'left mid zone lung',
            'right upper zone lung',
            'left upper zone lung',
        ]
    
    # Expansion points에 대한 anatomy QA
    for idx, (gt_point, gt_anatomy) in enumerate(zip(gt_points_expansion, gt_anatomy_expansion)):
        point_color = point_colors[tuple(gt_point)]
        
        # Fake anatomies 선택 (겹치지 않는 것만)
        filtered_anatomies = filter_fake_anatomies(gt_anatomy, all_anatomies)
        num_fake_to_select = min(3, len(filtered_anatomies))
        fake_anatomies = random.sample(filtered_anatomies, num_fake_to_select) if num_fake_to_select > 0 else []
        
        # 옵션 생성 (정답 + fake)
        anatomy_options = [gt_anatomy] + fake_anatomies
        random.shuffle(anatomy_options)
        
        # 정답 인덱스 찾기
        answer_idx = anatomy_options.index(gt_anatomy) + 1  # 1-based
        
        # 전처리된 anatomy 이름 적용
        anatomy_options_processed = [preprocess_anatomy_name(a) for a in anatomy_options]
        
        # Build question with options
        anatomy_options_text = "\n".join([f"({i}) {opt}" for i, opt in enumerate(anatomy_options_processed, 1)])
        question_with_options = f"Which anatomy is the {point_color} point closest to?\n\nOptions:\n{anatomy_options_text}"
        
        point_anatomy_qas.append({
            'point': gt_point,
            'point_color': point_color,
            'revision_type': 'expansion',
            'question': question_with_options,
            'answer_options': anatomy_options_processed,
            'answer': preprocess_anatomy_name(gt_anatomy),
            'answer_index': answer_idx
        })
    
    # Contraction points에 대한 anatomy QA
    for idx, (gt_point, gt_anatomy) in enumerate(zip(gt_points_contraction, gt_anatomy_contraction)):
        point_color = point_colors[tuple(gt_point)]
        
        # Fake anatomies 선택 (겹치지 않는 것만)
        filtered_anatomies = filter_fake_anatomies(gt_anatomy, all_anatomies)
        num_fake_to_select = min(3, len(filtered_anatomies))
        fake_anatomies = random.sample(filtered_anatomies, num_fake_to_select) if num_fake_to_select > 0 else []
        
        # 옵션 생성 (정답 + fake)
        anatomy_options = [gt_anatomy] + fake_anatomies
        random.shuffle(anatomy_options)
        
        # 정답 인덱스 찾기
        answer_idx = anatomy_options.index(gt_anatomy) + 1  # 1-based
        
        # 전처리된 anatomy 이름 적용
        anatomy_options_processed = [preprocess_anatomy_name(a) for a in anatomy_options]
        
        # Build question with options
        anatomy_options_text = "\n".join([f"({i}) {opt}" for i, opt in enumerate(anatomy_options_processed, 1)])
        question_with_options = f"Which anatomy is the {point_color} point closest to?\n\nOptions:\n{anatomy_options_text}"
        
        point_anatomy_qas.append({
            'point': gt_point,
            'point_color': point_color,
            'revision_type': 'contraction',
            'question': question_with_options,
            'answer_options': anatomy_options_processed,
            'answer': preprocess_anatomy_name(gt_anatomy),
            'answer_index': answer_idx
        })
    '''
    # Build initial context (shown once before expansion/contraction questions)
    if lesion_name.lower() != "cardiomegaly":
        adjustment_guideline = (
            "The colored points indicate potential locations where the mask may need to be adjusted. "
            "The mask may need to be expanded towards these points, contracted at these locations, or both. "
            "Note that lesions overlapping with the heart do not need to be considered."
        )
    else:
        adjustment_guideline = (
            "The colored points indicate potential locations where the mask may need to be adjusted. "
            "The mask may need to be expanded towards these points, contracted at these locations, or both."
        )

    initial_context = adjustment_guideline
    
    # Build expansion question with options
    expansion_instruction = f"Which point(s), if any, should be used to expand the mask toward areas of the {lesion_name} that are not currently covered? Select all points where expansion is needed. If no expansion is needed, select 'None'."
    expansion_options_text = "\n".join([f"({i}) {opt['color'] + ' point' if opt['color'] != 'None' else opt['color']}" for i, opt in enumerate(answer_options, 1)])
    expansion_question = f"{expansion_instruction}\n\nOptions:\n{expansion_options_text}"
    
    # Build contraction question with options (no context, just the question)
    contraction_instruction = f"Which point(s), if any, should be used to contract the mask away from areas that are not part of the {lesion_name}? Select all points where contraction is needed. If no contraction is needed, select 'None'."
    contraction_options_text = "\n".join([f"({i}) {opt['color'] + ' point' if opt['color'] != 'None' else opt['color']}" for i, opt in enumerate(answer_options, 1)])
    contraction_question = f"{contraction_instruction}\n\nOptions:\n{contraction_options_text}"
    
    # Build revision_result question with options
    ordinal_numbers = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th', 5: '5th', 6: '6th', 7: '7th', 8: '8th'}
    revision_result_options_text = "\n".join([f"({i}) {ordinal_numbers.get(i, f'{i}th')} mask" for i in range(1, len(revision_result_options) + 1)])
    revision_result_question = f"Choose the mask that reflects the modifications.\n\nOptions:\n{revision_result_options_text}"
    

    if no_deformation_override is not None:
        no_deformation = no_deformation_override
    else:
        no_deformation = random.random() < no_deformation_ratio

    if len(all_component_ids) == 0:
        print("all component ids are 0!!!!!!!!!!!!!!!")
        no_deformation = True

    contour_eval_answer_options = ['Yes', 'No']
    options_text = "\n".join([f"({i}) {opt}" for i, opt in enumerate(contour_eval_answer_options, 1)])
    
    if lesion_name.lower() != "cardiomegaly":
        constraint = (
            " Note that lesions overlapping with the heart do not need to be considered."
        )
    else:
        constraint = ""

    contour_eval_question = (
        f"The image shows a chest X-ray with a predicted mask for the {lesion_name} "
        f"(shown in the overlay). Is any major revision needed?{constraint}\n\n"
        f"Options:\n{options_text}"
    )

    revision_mask_answer_mapping = {
        1: '1st mask',
        2: '2nd mask',
        3: '3rd mask',
        4: '4th mask',
    }

    contour_qa = {
        'no_deformation': no_deformation,
        'initial_context': initial_context,  # 상황 설명 (expansion 전에 한 번만 제공)
        'revision_before_masks': revision_before_masks,
        'revision_after_masks': revision_after_masks,
        'true_masks': all_true_masks,
        'fake_masks': all_fake_masks,
        'true_combination': true_combination,  # 모든 component가 true인 조합
        'fake_combinations': selected_fake_combinations,  # 랜덤 선택된 3개의 fake 조합
        'contour_eval_qa':{
            'question': contour_eval_question,
            'answer_options': contour_eval_answer_options,
            'answer': 'No' if no_deformation else 'Yes',
            'answer_index': 2 if no_deformation else 1,
            'relative_path': os.path.join('contour_eval_qa', 'xray_with_mask.png'),
        },
        'contour_revision_qa_expansion': {
            'question': expansion_question,
            'answer_options': answer_options,
            'answer': expansion_answer,
            'answer_index': expansion_answer_indices
        },
        'contour_revision_qa_contraction': {
            'question': contraction_question,
            'answer_options': answer_options,
            'answer': contraction_answer,
            'answer_index': contraction_answer_indices
        },
        'contour_revision_qa_revision_result': {
            'question': revision_result_question,
            'answer_options': revision_result_options,  # [{'type': 'true'/'fake', 'masks': [...]}, ...]
            'answer': revision_mask_answer_mapping[revision_result_answer_index],
            'answer_index': revision_result_answer_index
        },
        #'point_anatomy': point_anatomy_qas  # 각 포인트에 대한 anatomy 위치 QA
    }

    return contour_qa


def generate_detection_qa(lesion_name, negative=False):

    answer_options = ['Yes', 'No']
    
    # Build question with options
    options_text = "\n".join([f"({i}) {opt}" for i, opt in enumerate(answer_options, 1)])

    if lesion_name in ['opacity', 'consolidation']:
        question = f"Is any {lesion_name} visible in the image?\n\nOptions:\n{options_text}"
    else:
        question = f"Is any finding suggestive of {lesion_name} visible in the image?\n\nOptions:\n{options_text}"
    
    detection_qa = {
        'question': question,
        'answer_options': answer_options,
        'answer': 'Yes' if not negative else 'No',
        'answer_index': 1 if not negative else 2
    }

    return detection_qa

def build_qa(key_id, results, lesion_name, generate_sequential_qa=True, no_deformation_ratio=0.5, no_deformation_override=None):
    """
    Build QA dictionary from results
    
    Args:
        results: deformation results
        lesion_name: name of the lesion
        generate_sequential_qa: whether to generate sequential QA (default: True)
        no_deformation_ratio: fallback probability when no_deformation_override is None (default: 0.5)
        no_deformation_override: if not None, directly sets no_deformation to this boolean value,
                                 bypassing random sampling (used for exact ratio control)
    
    Returns:
        qa: dictionary containing all QA types
    """
    qa = {}
    
    if isinstance(results, str) and results == 'negative':
        qa['detection_qa'] = generate_detection_qa(lesion_name, negative=True)
    else:
        qa_deformation_results = results['qa_deformation_results']
        geometrical_mask_infos = results['geometrical_mask_infos']
        
        overlaps = {}

        for i in geometrical_mask_infos:
        
            suboptimal_component_id = f"suboptimal_component_{i['mask_component_id']}"
            overlap = i['overlap']
            overlaps[suboptimal_component_id] = overlap
        
        detection_qa = generate_detection_qa(lesion_name)

        contour_qa = generate_contour_qa(
            key_id, qa_deformation_results, geometrical_mask_infos, lesion_name,
            no_deformation_ratio=no_deformation_ratio,
            no_deformation_override=no_deformation_override
        )
        
        if lesion_name != 'cardiomegaly':
            attribute_extraction_qa = generate_attribute_extraction_qa(overlaps, lesion_name)
        else:
            attribute_extraction_qa = None
        
        
        qa['detection_qa'] = detection_qa
        qa['contour_qa'] = contour_qa
        qa['attribute_extraction_qa'] = attribute_extraction_qa
        
        # Sequential QA는 옵션에 따라 생성
        if generate_sequential_qa:
            deformation_sequential_qa = generate_deformation_qa(qa_deformation_results, geometrical_mask_infos, lesion_name)
            qa['revision_sequential_qa'] = deformation_sequential_qa
        
    return qa