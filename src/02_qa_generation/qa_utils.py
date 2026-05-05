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
            # rename lung base to lower zone lung
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

    # Randomly pick up to (num_option-1) zones from pos_zones (0 is possible)
    max_select = min(num_option - 1, len(pos_zones))
    num_select = random.randint(0, max_select) if max_select > 0 else 0
    selected_pos_zones = random.sample(pos_zones, num_select) if num_select > 0 else []

    # selected_pos_zones must be included in the options;
    # fill the remaining options from neg_zones to reach a total of (num_option-1)
    num_remaining_options = num_option - 1 - len(selected_pos_zones)
    if num_remaining_options > 0 and len(neg_zones) > 0:
        num_neg_to_select = min(num_remaining_options, len(neg_zones))
        selected_neg_zones = random.sample(neg_zones, num_neg_to_select)
    else:
        selected_neg_zones = []

    # Merge selected_pos_zones and selected_neg_zones, then shuffle
    selected_zones = selected_pos_zones + selected_neg_zones

    # If selected_zones has fewer than (num_option-1) entries,
    # neg_zones was short, so top up by sampling more from the remaining pos_zones
    if len(selected_zones) < num_option - 1:
        remaining_pos_zones = [z for z in pos_zones if z not in selected_pos_zones]
        num_additional_needed = (num_option - 1) - len(selected_zones)
        if len(remaining_pos_zones) > 0:
            num_additional = min(num_additional_needed, len(remaining_pos_zones))
            additional_pos_zones = random.sample(remaining_pos_zones, num_additional)
            selected_zones.extend(additional_pos_zones)
            selected_pos_zones.extend(additional_pos_zones)  # also include in the answer

    random.shuffle(selected_zones)

    # Build answer_options (always (num_option-1) entries from 1 to num_option-1)
    answer_options = {}
    for i in range(1, num_option):
        answer_options[i] = selected_zones[i-1]

    # The last option (index num_option) is always "None of the above"
    option_range = f"1-{num_option-1}" if num_option > 2 else "1"
    answer_options[num_option] = f'None of the above (options {option_range})'

    # Build question with options
    options_text = "\n".join([f"({i}) {opt}" for i in range(1, num_option + 1) for opt in [answer_options[i]]])
    question = f"Where is the {lesion_name} located in the image? Select all locations where the lesion is present from the options.\n\nOptions:\n{options_text}"

    # Convert answer_options to a list in order (1 to num_option)
    answer_options_list = []
    for i in range(1, num_option + 1):
        if i in answer_options:
            answer_options_list.append(answer_options[i])

    # The answer is the indices of zones in selected_pos_zones (between 1 and num_option-1)
    answer_indices = []
    answer_texts = []
    for idx in range(1, num_option):
        if idx in answer_options and answer_options[idx] in selected_pos_zones:
            answer_indices.append(idx)
            answer_texts.append(answer_options[idx])

    # If there is no answer (selected_pos_zones is empty or none of them are in the picked options) -> num_option is the answer
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
    
    # Find lungs (left/right) with has_overlap=True
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
    
    # Pick the lung based on the answer zones from location_qa
    selected_lung_name = None
    overlap_ratio = 0.0

    if location_answer_zones:
        # Check left/right among the answer zones from location_qa
        has_left_zones = any('left' in zone for zone in location_answer_zones)
        has_right_zones = any('right' in zone for zone in location_answer_zones)

        # Prefer the lung that contains the answer zone
        if has_left_zones and has_right_zones:
            # If both sides are present, choose at random
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
            # If only left zones are present, select left lung
            if lungs_with_overlap['left lung']['has_overlap']:
                selected_lung_name = 'left lung'
                overlap_ratio = lungs_with_overlap['left lung']['overlap_ratio']
        elif has_right_zones:
            # If only right zones are present, select right lung
            if lungs_with_overlap['right lung']['has_overlap']:
                selected_lung_name = 'right lung'
                overlap_ratio = lungs_with_overlap['right lung']['overlap_ratio']

    # Fall back to the legacy logic if location_answer_zones is missing or no lung matches
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
            # No lesion present (should not happen in theory; safety fallback)
            overlap_ratio = 0.0
            selected_lung_name = 'left lung'  # default

    # Indicate the selected lung in answer_options
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
    
    # Determine answer_index from overlap_ratio
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
    
    # Fetch the size values for left lung and right lung
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
        
    # Determine the answer
    # Option 4: lesion only on one side
    if (left_has_overlap and not right_has_overlap) or (not left_has_overlap and right_has_overlap):
        answer_index = 4
    # If neither side has overlap (should not happen in theory; safety fallback)
    elif not left_has_overlap and not right_has_overlap:
        answer_index = 4
    # If both sides have overlap, compare sizes
    else:
        # Avoid division by zero
        if left_size == 0 and right_size == 0:
            answer_index = 1  # both zero, similar
        elif left_size == 0:
            answer_index = 3  # only right side
        elif right_size == 0:
            answer_index = 2  # only left side
        else:
            # Compare sizes (1.5x threshold)
            size_ratio = max(left_size, right_size) / min(left_size, right_size)

            if size_ratio < 1.5:
                # similar (less than 1.5x difference)
                answer_index = 1
            elif left_size >= right_size * 1.5:
                # left is larger
                answer_index = 2
            elif right_size >= left_size * 1.5:
                # right is larger
                answer_index = 3
            else:
                # exceptional case (should not happen in theory)
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
    
    # Pass the answer zones from location_qa to severity_measurement_qa
    location_answer_zones = attribute_extraction_qa['location']['answer_zones'] if attribute_extraction_qa['location'] else None
    attribute_extraction_qa['severity/measurement'] = generate_severity_measurement_qa(overlaps, lesion_name, location_answer_zones, num_option)
    attribute_extraction_qa['comparison'] = generate_comparison_qa(overlaps, lesion_name, num_option)
    
    return attribute_extraction_qa

def generate_localize_qa(lesion_name, anatomy_name, operation_to_anatomy_names, center_point, fake_points_expansion, fake_points_contraction, operation, revision, all_available_fake_points=None, selected_component_id=None, num_option=4):
    """
    Args:
        lesion_name: lesion name
        anatomy_name: anatomical location name
        operation_to_anatomy_names: list of anatomy names per operation
        center_point: ground-truth coordinates
        fake_points_expansion: expansion fake points for the current component
        fake_points_contraction: contraction fake points for the current component
        operation: current operation ('expansion' or 'contraction')
        revision: revision type
        all_available_fake_points: fake_points usable across all components (dict form: {component_id: {'expansion': [...], 'contraction': [...]}})
        selected_component_id: currently selected component ID (used to exclude the current component)
        num_option: total number of options
    """

    # Build text-mode QA
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

    # Keep only locations that do not conflict with anatomy_name
    # 1. peripheral and lateral overlap when they share a zone, so they should not appear together
    # 2. costophrenic angle overlaps with the peripheral/lateral of lung base, so they should not appear together

    def get_zone_from_anatomy(anatomy):
        """Extract zone info from anatomy (upper zone, mid zone, lung base)."""
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
        """Extract side info from anatomy (left, right)."""
        if 'left' in anatomy:
            return 'left'
        elif 'right' in anatomy:
            return 'right'
        return None

    filtered_location_names = []

    # zone and side info for anatomy_name
    anatomy_zone = get_zone_from_anatomy(anatomy_name)
    anatomy_side = get_side_from_anatomy(anatomy_name)
    # whether anatomy_name contains 'peripheral'
    anatomy_has_peripheral = 'peripheral' in anatomy_name
    # whether anatomy_name contains 'lateral'
    anatomy_has_lateral = 'lateral' in anatomy_name
    # whether anatomy_name is a costophrenic angle
    anatomy_is_costophrenic = 'costophrenic angle' in anatomy_name

    for loc_name in all_location_names:
        loc_zone = get_zone_from_anatomy(loc_name)
        loc_side = get_side_from_anatomy(loc_name)

        # Only check for conflicts when zone and side match
        if anatomy_zone and loc_zone and anatomy_zone == loc_zone:
            if anatomy_side and loc_side and anatomy_side == loc_side:
                # Filter out peripheral and lateral pairs in the same zone and side so they don't appear together
                if anatomy_has_peripheral and 'lateral' in loc_name:
                    continue
                if anatomy_has_lateral and 'peripheral' in loc_name:
                    continue

                # Filter so that costophrenic angle and lung base peripheral/lateral don't appear together
                if anatomy_is_costophrenic:
                    if 'peripheral' in loc_name or 'lateral' in loc_name:
                        continue
                if anatomy_has_peripheral or anatomy_has_lateral:
                    if 'costophrenic angle' in loc_name:
                        continue

        filtered_location_names.append(loc_name)

    # Always include the correct answer; details TBD
    num_to_select = min(num_option - 1, len(filtered_location_names))
    negative_location = random.sample(filtered_location_names, num_to_select) if num_to_select > 0 else []
    
    answer_text_options = [anatomy_name] + negative_location
    
    random.shuffle(answer_text_options)
    
    answer_text_index = answer_text_options.index(anatomy_name) + 1
    
    # Build question with text options
    text_options_processed = [preprocess_anatomy_name(opt) for opt in answer_text_options]
    text_options_text = "\n".join([f"({i}) {opt}" for i, opt in enumerate(text_options_processed, 1)])
    question_with_text_options = f"In which anatomical region does the mask need {revision}?\n\nOptions:\n{text_options_text}"

    # Build coordinate-mode QA
    # fake_points_expansion and fake_points_contraction are lists of [(x, y), ...]
    fake_points_expansion = fake_points_expansion if fake_points_expansion else []
    fake_points_contraction = fake_points_contraction if fake_points_contraction else []

    # Pick num_option - 1 fake points in addition to the correct answer
    num_fake_needed = num_option - 1
    fake_points_list = []

    # Track whether points from another component were used
    used_other_component_point = False
    used_default_point = False

    # Determine priority based on operation
    if operation == 'contraction':
        priority_fake_points = fake_points_expansion.copy()
        secondary_fake_points = fake_points_contraction.copy()
    elif operation == 'expansion':
        priority_fake_points = fake_points_contraction.copy()
        secondary_fake_points = fake_points_expansion.copy()
    else:
        # If there is no operation, treat both equally
        all_fake_points = fake_points_expansion + fake_points_contraction
        priority_fake_points = all_fake_points.copy()
        secondary_fake_points = []

    # Check whether the current component has enough fake points
    if len(priority_fake_points) + len(secondary_fake_points) >= num_fake_needed:
        # If enough, pick only from the current ones, preferring by operation
        # 1) fill from priority as much as possible
        if len(priority_fake_points) >= num_fake_needed:
            fake_points_list = random.sample(priority_fake_points, num_fake_needed)
        else:
            fake_points_list = priority_fake_points.copy()
            remaining = num_fake_needed - len(fake_points_list)
            if remaining > 0 and len(secondary_fake_points) > 0:
                num_to_add = min(remaining, len(secondary_fake_points))
                fake_points_list.extend(random.sample(secondary_fake_points, num_to_add))
    else:
        # Use everything from the current component
        fake_points_list = priority_fake_points + secondary_fake_points

        # Pull additional fake_points from other components
        if all_available_fake_points:
            # Structure of all_available_fake_points: {component_id: {'expansion': [...], 'contraction': [...]}}
            # exclude the current component
            all_other_fake_points_expansion = []
            all_other_fake_points_contraction = []
            for component_id, points_dict in all_available_fake_points.items():
                # exclude the current component
                if component_id != selected_component_id:
                    if 'expansion' in points_dict:
                        all_other_fake_points_expansion.extend(points_dict['expansion'])
                    if 'contraction' in points_dict:
                        all_other_fake_points_contraction.extend(points_dict['contraction'])

            # Apply operation-based priority for other-component fakes too
            if operation == 'contraction':
                priority_other_points = all_other_fake_points_expansion
                secondary_other_points = all_other_fake_points_contraction
            elif operation == 'expansion':
                priority_other_points = all_other_fake_points_contraction
                secondary_other_points = all_other_fake_points_expansion
            else:
                priority_other_points = all_other_fake_points_expansion + all_other_fake_points_contraction
                secondary_other_points = []

            # Remove duplicates (exclude points already in use)
            remaining_priority = [fp for fp in priority_other_points if fp not in fake_points_list]
            remaining_secondary = [fp for fp in secondary_other_points if fp not in fake_points_list]

            # Number still needed
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
    
    # If still not enough, create default points (fixed at the image center)
    if len(fake_points_list) < num_fake_needed:
        needed = num_fake_needed - len(fake_points_list)
        # default points: image center (for 1024x1024, that is (512, 512))
        default_point = (512, 512)
        default_points = [default_point] * needed
        fake_points_list.extend(default_points)
        used_default_point = True

    # Combine the correct answer and fake points to form the options
    answer_point_options = [center_point] + fake_points_list

    # Shuffle
    random.shuffle(answer_point_options)

    # Find the index of the correct answer
    answer_point_index = answer_point_options.index(center_point) + 1

    # Return a single dictionary containing both text-mode and coordinate-mode options
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
        deformation: deformation data for the current component
        all_available_fake_masks: fake_mask list usable from other components (dict form: {component_id: [fake_mask_paths]})
        all_component_ids: set of all component IDs (used as a fallback)
        num_option: total number of options
    """

    gt_mask = deformation['revision_flow']['after revision']
    selected_component_id = deformation.get('suboptimal_component_id')

    # Collect fake_masks from the current component
    current_fake_masks = []
    if 'fake_masks' in deformation and deformation['fake_masks']:
        current_fake_masks = [fm['fake_mask'] for fm in deformation['fake_masks']]

    # Compute the number of fake_masks needed
    num_fake_needed = num_option - 1

    # Track whether fake_masks from another component were used
    used_other_component_fake = False
    used_default_fake = False

    # If the current component does not have enough fake_masks, pull more from other components

    fake_masks_list = []
    if len(current_fake_masks) >= num_fake_needed:
        # If enough, sample from the current ones but preferring by operation
        fake_masks_list = []
        # First take from the priority bucket; if not enough, top up from the rest

        if operation == 'expansion':
            priority_masks = [m for m in current_fake_masks if 'fake_contraction' in m]
            secondary_masks = [m for m in current_fake_masks if 'fake_expansion' in m]
        elif operation == 'contraction':
            priority_masks = [m for m in current_fake_masks if 'fake_expansion' in m]
            secondary_masks = [m for m in current_fake_masks if 'fake_contraction' in m]

        # 1) fill from priority as much as possible
        if len(priority_masks) >= num_fake_needed:
            fake_masks_list = random.sample(priority_masks, num_fake_needed)
        else:
            fake_masks_list = priority_masks.copy()
            remaining = num_fake_needed - len(fake_masks_list)
            if remaining > 0 and len(secondary_masks) > 0:
                num_to_add = min(remaining, len(secondary_masks))
                fake_masks_list.extend(random.sample(secondary_masks, num_to_add))
    else:
        # Use everything from the current component (already ordered by operation)
        fake_masks_list = current_fake_masks.copy()

        # Pull additional fake_masks from other components
        if all_available_fake_masks:
            # Flatten the fake_masks from every component
            all_other_fake_masks = []
            for component_id, fake_mask_paths in all_available_fake_masks.items():
                # exclude the current component
                if component_id != deformation.get('suboptimal_component_id'):
                    all_other_fake_masks.extend(fake_mask_paths)

            # Remove duplicates (exclude items already in use)
            remaining_fake_masks = [fm for fm in all_other_fake_masks if fm not in fake_masks_list]

            # Number still needed
            num_additional_needed = num_fake_needed - len(fake_masks_list)
            if num_additional_needed > 0 and len(remaining_fake_masks) > 0:
                # Apply operation-based priority for other-component fakes too
                if operation == 'expansion':
                    # In an expansion round, surface contraction-style fakes first
                    priority_masks = [m for m in remaining_fake_masks if 'fake_contraction' in m]
                    secondary_masks = [m for m in remaining_fake_masks if m not in priority_masks]
                elif operation == 'contraction':
                    # In a contraction round, surface expansion-style fakes first
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
    
    # If still not enough, fall back to defaults (for backward compatibility)
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
            # Pick from the default candidates at random
            num_to_add = min(needed, len(default_fake_masks))
            fake_masks_list.extend(random.sample(default_fake_masks, num_to_add))
            used_default_fake = True

    # Build a mapping of fake_mask to its owning component
    fake_mask_to_component_id = {}
    # fake_mask of the current component
    for fake_mask in current_fake_masks:
        fake_mask_to_component_id[fake_mask] = selected_component_id
    # fake_mask from other components
    if all_available_fake_masks:
        for component_id, fake_mask_paths in all_available_fake_masks.items():
            for fake_mask in fake_mask_paths:
                fake_mask_to_component_id[fake_mask] = component_id

    # Combine the correct mask and fake masks into a temporary list and shuffle
    answer_options_temp = [{'mask': gt_mask, 'is_answer': True}]
    for fake_mask in fake_masks_list[:num_fake_needed]:
        answer_options_temp.append({'mask': fake_mask, 'is_answer': False})

    random.shuffle(answer_options_temp)

    # Find answer_index and add relative_path
    answer_index = None
    answer_options = []
    option_component_states = {}

    # Build component-state info for each option
    all_component_states = deformation.get('all_component_mask_states', {})

    # If all_component_states is empty, initialize using all_component_ids
    if not all_component_states and all_component_ids:
        all_component_states = {comp_id: comp_id for comp_id in all_component_ids}  # initial state is suboptimal_component

    for opt_idx, option_data in enumerate(answer_options_temp, 1):
        mask_path = option_data['mask']
        is_answer = option_data['is_answer']

        # Find answer_index
        if is_answer:
            answer_index = opt_idx
            relative_path = f'option_{opt_idx}_mask_answer.png'
        else:
            relative_path = f'option_{opt_idx}_mask_fake.png'

        # Append a dict entry to answer_options
        answer_options.append({
            'mask': mask_path,
            'relative_path': relative_path
        })

        # Build component state per option
        if all_component_states and selected_component_id:
            # Copy state of all components
            option_states = all_component_states.copy()

            # If mask_path is the correct answer, update selected_component_id
            if mask_path == gt_mask:
                option_states[selected_component_id] = mask_path
            else:
                # If it is a fake_mask, find which component it belongs to
                if mask_path in fake_mask_to_component_id:
                    # Update the state of that component
                    fake_mask_component_id = fake_mask_to_component_id[mask_path]
                    option_states[fake_mask_component_id] = mask_path
                else:
                    # For a default fake_mask, assign it to selected_component_id (fallback)
                    option_states[selected_component_id] = mask_path

            option_component_states[opt_idx] = option_states
        elif selected_component_id:
            # fallback: include only selected_component, but if all_component_ids exists, include all components
            if all_component_ids:
                option_states = {comp_id: comp_id for comp_id in all_component_ids}  # initial state
                # If mask_path is the correct answer, update selected_component_id
                if mask_path == gt_mask:
                    option_states[selected_component_id] = mask_path
                else:
                    # If it is a fake_mask, find which component it belongs to
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
        'answer_options': answer_options,  # now a list of dicts: [{'mask': '...', 'relative_path': '...'}, ...]
        'answer': gt_mask,
        'answer_index': answer_index,
        'answer_mask_path': gt_mask,
        'fake_mask_paths': fake_masks_list[:num_fake_needed],
        'used_other_component_fake': used_other_component_fake,
        'used_default_fake': used_default_fake,
        'option_component_states': option_component_states,  # per-option component state info
        'selected_component_id': selected_component_id,
        'all_component_states_before': all_component_states,  # state of all components before revision
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
    
    # Collect fake_masks from all components
    all_available_fake_masks = {}
    for operation, deformation_results in qa_deformation_results.items():
        for suboptimal_component_id, deformations in deformation_results.items():
            component_fake_masks = []
            for deformation in deformations:
                if 'fake_masks' in deformation and deformation['fake_masks']:
                    component_fake_masks.extend([fm['fake_mask'] for fm in deformation['fake_masks']])
            if component_fake_masks:
                all_available_fake_masks[suboptimal_component_id] = component_fake_masks
    
    # Collect fake_points from all components
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
    
    # Collect every component ID
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
    
    # Track each component's current mask state (updated per round)
    component_mask_states = {}
    for component_id in all_component_ids:
        
        if component_id in qa_deformation_results['contraction']:
            component_mask_states[component_id] = qa_deformation_results['contraction'][component_id][-1]['revision_flow']['before revision']
        elif component_id in qa_deformation_results['expansion']:
            component_mask_states[component_id] = qa_deformation_results['expansion'][component_id][-1]['revision_flow']['before revision']
        else:
            component_mask_states[component_id] = component_id.split('sub')[1]

    idx = 0
    
    for operation in ['contraction', 'expansion']: # Should also consider the case with no deformation!!!!!!
        
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
            
            # Add suboptimal_component_id to deformation (used in revision_qa)
            selected_deformation['suboptimal_component_id'] = selected_id

            # Save the mask states of all components for this round
            selected_deformation['all_component_mask_states'] = component_mask_states.copy()

            # Update the mask state of the selected component (after revision)
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

    # Final question
    final_round_data = {
        'initial_qa': generate_initial_qa(None, final=True),
        'all_component_mask_states': component_mask_states.copy(),  # save the final state of all components
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
    component_anatomies = {}  # store per-component anatomy info

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

                # Store anatomy info
                if suboptimal_component_id not in component_anatomies:
                    component_anatomies[suboptimal_component_id] = deformation.get('anatomy', '')
                
                for mask in deformation['fake_masks']:
                    all_fake_masks[suboptimal_component_id].append(mask['fake_mask'])

                all_fake_masks[suboptimal_component_id].append(deformation['mask_path'])

                fake_points_expansion.extend(deformation['fake_points_expansion'])
                fake_points_contraction.extend(deformation['fake_points_contraction'])

    # Merge GT points
    all_gt_points = gt_points_expansion + gt_points_contraction
    all_fake_points = fake_points_expansion + fake_points_contraction

    # Keep only fake points that are at least 100 away from GT points
    def calculate_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    filtered_fake_points = []
    for fake_point in all_fake_points:
        is_far_enough = True

        # Check distance against GT points
        for gt_point in all_gt_points:
            if calculate_distance(fake_point, gt_point) < 100:
                is_far_enough = False
                break

        # Check distance against already-selected fake points
        if is_far_enough:
            for selected_fake in filtered_fake_points:
                if calculate_distance(fake_point, selected_fake) < 100:
                    is_far_enough = False
                    break

        if is_far_enough:
            filtered_fake_points.append(fake_point)

    # Color list
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown']
    max_points = len(colors)  # at most 10

    # Combine GT points with filtered fake points
    # If the total exceeds 10, keep at most 10 (GT points are always included)
    if len(all_gt_points) + len(filtered_fake_points) > max_points:
        # always include all GT points
        num_fake_to_keep = max_points - len(all_gt_points)
        if num_fake_to_keep > 0:
            # randomly sample from fake points
            filtered_fake_points = random.sample(filtered_fake_points, num_fake_to_keep)
        else:
            # If GT points alone exceed 10, use only GT points
            filtered_fake_points = []

    # Mix GT points with filtered fake points and assign colors
    all_points = all_gt_points + filtered_fake_points
    random.shuffle(all_points)

    # Assign a color to each point
    point_colors = {}
    for i, point in enumerate(all_points):
        color = colors[i % len(colors)]  # cycle if there are not enough colors
        point_colors[tuple(point)] = color

    # Build answer options (all points and colors) - shared by both questions
    answer_options = []
    for point, color in point_colors.items():
        answer_options.append({
            'point': list(point),
            'color': color
        })
    
    # Add the None option
    answer_options.append({
        'point': None,
        'color': 'None'
    })

    # Build expansion QA
    expansion_colors = [point_colors[tuple(pt)] for pt in gt_points_expansion]
    if expansion_colors:
        expansion_answer = ', '.join([f"{color} point" for color in expansion_colors])
        expansion_answer_indices = []
        for gt_point in gt_points_expansion:
            for idx, option in enumerate(answer_options):
                if option['point'] is not None and tuple(option['point']) == tuple(gt_point):
                    expansion_answer_indices.append(idx + 1)  # 1-based index
                    break
    else:
        expansion_answer = "None"
        expansion_answer_indices = [len(answer_options)]  # index of the None option (1-based)

    # Build contraction QA
    contraction_colors = [point_colors[tuple(pt)] for pt in gt_points_contraction]
    if contraction_colors:
        contraction_answer = ', '.join([f"{color} point" for color in contraction_colors])
        contraction_answer_indices = []
        for gt_point in gt_points_contraction:
            for idx, option in enumerate(answer_options):
                if option['point'] is not None and tuple(option['point']) == tuple(gt_point):
                    contraction_answer_indices.append(idx + 1)  # 1-based index
                    break
    else:
        contraction_answer = "None"
        contraction_answer_indices = [len(answer_options)]  # index of the None option (1-based)

    # Build true/fake mask combinations per component

    # Build every combination by choosing one of true mask or fake masks per component
    component_ids = sorted(all_true_masks.keys())

    # Build the list of selectable masks for each component

    right_default = False
    left_default = False

    component_mask_choices = []
    for comp_id in component_ids:
        choices = []
        # Add true mask
        if comp_id in all_true_masks and len(all_true_masks[comp_id]) > 0:
            choices.append(('true', all_true_masks[comp_id][0]))

        # Add fake masks
        fake_masks_for_comp = []
        if comp_id in all_fake_masks:
            fake_masks_for_comp = all_fake_masks[comp_id]

        # Add default masks if there are not enough fake masks
        #num_fake_needed = 3  # at least 3 fake masks needed
        #if len(fake_masks_for_comp) < num_fake_needed:
            # Define default fake masks
        if lesion_name == 'cardiomegaly':
            default_fake_masks = ['default_keep', 'default_dilated', 'default_eroded']
        else:
            # Check the component's anatomy info
            anatomy = component_anatomies.get(comp_id, '')

            # If anatomy contains 'right', use right lung default masks
            if 'right' in anatomy.lower():
                if not right_default:
                    default_fake_masks = ['default_keep', 'default_right_lung_chex', 'default_right_lung_cxas']
                    right_default = True
                else:
                    default_fake_masks = ['default_keep']
            # If it contains 'left', use left lung default masks
            elif 'left' in anatomy.lower():
                if not left_default:
                    default_fake_masks = ['default_keep', 'default_left_lung_chex', 'default_left_lung_cxas']
                    left_default = True
                else:
                    default_fake_masks = ['default_keep']
            else:
                # If anatomy info is missing, use every default mask
                raise ValueError(f"Invalid anatomy: {anatomy}")
                #default_fake_masks = ['default_keep', 'default_right_lung_chex', 'default_right_lung_cxas', 'default_left_lung_chex', 'default_left_lung_cxas']

            #needed = num_fake_needed - len(fake_masks_for_comp)
            # exclude fake masks already in use
        #available_defaults = [m for m in default_fake_masks if m not in fake_masks_for_comp]
        fake_masks_for_comp = fake_masks_for_comp + default_fake_masks
        #if needed > 0 and available_defaults:
        #     num_to_add = min(needed, len(available_defaults))
        #    fake_masks_for_comp = list(fake_masks_for_comp) + random.sample(available_defaults, num_to_add)

        # Add fake masks to choices
        for fake_mask in fake_masks_for_comp:
            choices.append(('fake', fake_mask))

        component_mask_choices.append(choices)

    # Build every combination (one per component)
    all_combinations = list(itertools.product(*component_mask_choices))

    # Classify combinations as true/fake
    fake_combinations = []  # combinations containing at least one fake
    true_combination = None  # combination with all true entries

    for combination in all_combinations:
        # combination: [('true', mask1), ('fake', mask2), ...]
        has_fake = any(choice[0] == 'fake' for choice in combination)

        if has_fake:
            # At least one fake, so this is a fake combination
            masks = [choice[1] for choice in combination]
            if set(masks) != set(revision_before_masks):
                fake_combinations.append(masks)
            else:
                continue
        else:
            # All true, so this is the true combination
            masks = [choice[1] for choice in combination]
            true_combination = masks

    # Pick 3 fake combinations at random (prefer combinations with fewer default masks)
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
    
    # Build revision result QA (true combination + fake combinations)
    revision_result_options_temp = []

    # Add the true combination (correct answer)
    if true_combination:
        revision_result_options_temp.append({
            'type': 'true',
            'masks': true_combination
        })

    # Add fake combinations (distractors)
    for fake_combo in selected_fake_combinations:
        revision_result_options_temp.append({
            'type': 'fake',
            'masks': fake_combo
        })

    # Shuffle the options
    random.shuffle(revision_result_options_temp)

    # Find the answer index and add relative_path
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
            'relative_path': relative_path  # relative path used when stored under chexpercept
        })

    # Build per-point anatomy location QA


    '''
    point_anatomy_qas = []
    
    # Helper function to get zone from anatomy
    def get_zone_from_anatomy(anatomy):
        """Extract zone info from anatomy (upper zone, mid zone, lung base)."""
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
        """Extract side info from anatomy (left, right)."""
        if 'left' in anatomy:
            return 'left'
        elif 'right' in anatomy:
            return 'right'
        return None

    def filter_fake_anatomies(correct_anatomy, all_anatomies):
        """Pick 3 fake anatomies excluding any that conflict with correct_anatomy."""
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

            # Only check for conflicts when zone and side match
            if anatomy_zone and loc_zone and anatomy_zone == loc_zone:
                if anatomy_side and loc_side and anatomy_side == loc_side:
                    # Filter out peripheral and lateral pairs in the same zone and side so they don't appear together
                    if anatomy_has_peripheral and 'lateral' in candidate:
                        continue
                    if anatomy_has_lateral and 'peripheral' in candidate:
                        continue

                    # Filter so that costophrenic angle and lung base peripheral/lateral don't appear together
                    if anatomy_is_costophrenic:
                        if 'peripheral' in candidate or 'lateral' in candidate:
                            continue
                    if anatomy_has_peripheral or anatomy_has_lateral:
                        if 'costophrenic angle' in candidate:
                            continue

            filtered_anatomies.append(candidate)

        return filtered_anatomies

    # Define the list of all possible anatomies
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
    
    # Anatomy QA for expansion points
    for idx, (gt_point, gt_anatomy) in enumerate(zip(gt_points_expansion, gt_anatomy_expansion)):
        point_color = point_colors[tuple(gt_point)]

        # Pick fake anatomies (only non-conflicting ones)
        filtered_anatomies = filter_fake_anatomies(gt_anatomy, all_anatomies)
        num_fake_to_select = min(3, len(filtered_anatomies))
        fake_anatomies = random.sample(filtered_anatomies, num_fake_to_select) if num_fake_to_select > 0 else []

        # Build options (answer + fakes)
        anatomy_options = [gt_anatomy] + fake_anatomies
        random.shuffle(anatomy_options)

        # Find the answer index
        answer_idx = anatomy_options.index(gt_anatomy) + 1  # 1-based

        # Apply the preprocessed anatomy names
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
    
    # Anatomy QA for contraction points
    for idx, (gt_point, gt_anatomy) in enumerate(zip(gt_points_contraction, gt_anatomy_contraction)):
        point_color = point_colors[tuple(gt_point)]

        # Pick fake anatomies (only non-conflicting ones)
        filtered_anatomies = filter_fake_anatomies(gt_anatomy, all_anatomies)
        num_fake_to_select = min(3, len(filtered_anatomies))
        fake_anatomies = random.sample(filtered_anatomies, num_fake_to_select) if num_fake_to_select > 0 else []

        # Build options (answer + fakes)
        anatomy_options = [gt_anatomy] + fake_anatomies
        random.shuffle(anatomy_options)

        # Find the answer index
        answer_idx = anatomy_options.index(gt_anatomy) + 1  # 1-based

        # Apply the preprocessed anatomy names
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
        'initial_context': initial_context,  # context shown once before the expansion question
        'revision_before_masks': revision_before_masks,
        'revision_after_masks': revision_after_masks,
        'true_masks': all_true_masks,
        'fake_masks': all_fake_masks,
        'true_combination': true_combination,  # combination where all components are true
        'fake_combinations': selected_fake_combinations,  # 3 fake combinations chosen at random
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
        #'point_anatomy': point_anatomy_qas  # per-point anatomy location QA
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
        
        # Generate sequential QA according to the option
        if generate_sequential_qa:
            deformation_sequential_qa = generate_deformation_qa(qa_deformation_results, geometrical_mask_infos, lesion_name)
            qa['revision_sequential_qa'] = deformation_sequential_qa
        
    return qa