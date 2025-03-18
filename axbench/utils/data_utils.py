from dataclasses import dataclass
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import set_seed
import transformers, datasets, torch
from typing import Dict, Optional, Sequence, Union, List, Any
import numpy as np
import random
def parse_positions(positions: str):
    # parse position
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


def get_intervention_locations(**kwargs):
    """
    This function generates the intervention locations.

    For your customized dataset, you want to create your own function.
    """
    # parse kwargs
    share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
    last_position = kwargs["last_position"]
    if "positions" in kwargs:
        _first_n, _last_n = parse_positions(kwargs["positions"])
    else:
        _first_n, _last_n = kwargs["first_n"], kwargs["last_n"]
    num_interventions = kwargs["num_interventions"]
    pad_mode = kwargs["pad_mode"] if "pad_mode" in kwargs else "first"

    first_n = min(last_position // 2, _first_n)
    last_n = min(last_position // 2, _last_n)

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1 if pad_mode == "first" else last_position
    if share_weights or (first_n == 0 or last_n == 0):
        position_list = [i for i in range(first_n)] + \
            [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(pad_amount)]
        intervention_locations = [position_list]*num_interventions
    else:
        left_pad_amount = (_first_n - first_n)
        right_pad_amount = (_last_n - last_n)
        left_intervention_locations = [i for i in range(first_n)] + [pad_position for _ in range(left_pad_amount)]
        right_intervention_locations = [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(right_pad_amount)]
        # after padding, there could be still length diff, we need to do another check
        left_len = len(left_intervention_locations)
        right_len = len(right_intervention_locations)
        if left_len > right_len:
            right_intervention_locations += [pad_position for _ in range(left_len-right_len)]
        else:
            left_intervention_locations += [pad_position for _ in range(right_len-left_len)]
        intervention_locations = [left_intervention_locations]*(num_interventions//2) + \
            [right_intervention_locations]*(num_interventions//2)
    
    return intervention_locations


@dataclass
class InterventionDataCollator(object):
    """Collate examples for Intervention."""
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_intervention_len = max([len(inst["intervention_locations"][0]) for inst in instances])
        max_seq_len = max([len(inst["input_ids"]) for inst in instances])
        
        for inst in instances:
            non_pad_len = len(inst["input_ids"])

            _intervention_mask = torch.ones_like(inst["intervention_locations"][0])
            _intervention_location_paddings = torch.tensor(
                [[len(inst["input_ids"]) for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))]])
            _intervention_mask_paddings = torch.tensor(
                [0 for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))])
            inst["intervention_locations"] = torch.cat([inst["intervention_locations"], _intervention_location_paddings], dim=-1).int()
            inst["intervention_masks"] = torch.cat([_intervention_mask, _intervention_mask_paddings], dim=-1).int()
            inst["prompt_intervention_masks"] = inst["intervention_masks"].clone()
            inst["prompt_intervention_masks"][inst["prompt_lengths"]:] = 0 # mask out the intervention locations after prompt length

            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)])
            inst["input_ids"] = torch.cat((inst["input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _input_id_paddings)).int()

            _label_paddings = torch.tensor([-100 for _ in range(max_seq_len - non_pad_len+1)])
            inst["labels"] = torch.cat((inst["labels"], _label_paddings))
            
            inst["attention_mask"] = (inst["input_ids"] != self.tokenizer.pad_token_id).int()

        batch_inputs = self.data_collator(instances)
        return batch_inputs

@dataclass
class PreferenceInterventionDataCollator(object):
    """Collate examples for DPO with Intervention."""
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # this function will take care of padding by considering both winning and losing inputs and outputs
        max_intervention_len = max([max(
            len(inst["winning_intervention_locations"][0]), 
            len(inst["losing_intervention_locations"][0])) for inst in instances])
        max_seq_len = max([max(
            len(inst["winning_input_ids"]), 
            len(inst["losing_input_ids"])) for inst in instances]) 
        for inst in instances:
            winning_non_pad_len = len(inst["winning_input_ids"])
            losing_non_pad_len = len(inst["losing_input_ids"])
            
            _winning_intervention_mask = torch.ones_like(inst["winning_intervention_locations"][0])
            _winning_intervention_location_paddings = torch.tensor(
                [[winning_non_pad_len for _ in range(max_intervention_len - len(inst["winning_intervention_locations"][0]))]])
            _losing_intervention_mask = torch.ones_like(inst["losing_intervention_locations"][0])
            _losing_intervention_location_paddings = torch.tensor(
                [[losing_non_pad_len for _ in range(max_intervention_len - len(inst["losing_intervention_locations"][0]))]])
            _winning_intervention_mask_paddings = torch.tensor(
                [0 for _ in range(max_intervention_len - len(inst["winning_intervention_locations"][0]))])
            _losing_intervention_mask_paddings = torch.tensor(
                [0 for _ in range(max_intervention_len - len(inst["losing_intervention_locations"][0]))])
            
            inst["winning_intervention_locations"] = torch.cat(
                [inst["winning_intervention_locations"], _winning_intervention_location_paddings], dim=-1).int()
            inst["losing_intervention_locations"] = torch.cat(
                [inst["losing_intervention_locations"], _losing_intervention_location_paddings], dim=-1).int()
            inst["winning_intervention_masks"] = torch.cat([_winning_intervention_mask, _winning_intervention_mask_paddings], dim=-1).int()
            inst["losing_intervention_masks"] = torch.cat([_losing_intervention_mask, _losing_intervention_mask_paddings], dim=-1).int()
            inst["prompt_winning_intervention_masks"] = inst["winning_intervention_masks"].clone()
            inst["prompt_losing_intervention_masks"] = inst["losing_intervention_masks"].clone()
            inst["prompt_winning_intervention_masks"][inst["prompt_lengths"]:] = 0 # mask out the intervention locations after prompt length
            inst["prompt_losing_intervention_masks"][inst["prompt_lengths"]:] = 0 # mask out the intervention locations after prompt length
            
            _winning_input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - winning_non_pad_len)])
            _losing_input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - losing_non_pad_len)])
            inst["winning_input_ids"] = torch.cat(
                (inst["winning_input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _winning_input_id_paddings)).int()
            inst["losing_input_ids"] = torch.cat(
                (inst["losing_input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _losing_input_id_paddings)).int()  
            
            _winning_label_paddings = torch.tensor([-100 for _ in range(max_seq_len - winning_non_pad_len+1)])
            _losing_label_paddings = torch.tensor([-100 for _ in range(max_seq_len - losing_non_pad_len+1)])
            inst["winning_labels"] = torch.cat((inst["winning_labels"], _winning_label_paddings))
            inst["losing_labels"] = torch.cat((inst["losing_labels"], _losing_label_paddings))
            
            inst["winning_attention_mask"] = (inst["winning_input_ids"] != self.tokenizer.pad_token_id).int()
            inst["losing_attention_mask"] = (inst["losing_input_ids"] != self.tokenizer.pad_token_id).int()
            
        batch_inputs = self.data_collator(instances)
        return batch_inputs


@dataclass
class BiPreferenceInterventionDataCollator(object):
    """Collate examples for DPO with Intervention."""
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # this function will take care of padding by considering both winning and losing inputs and outputs
        max_intervention_len = max([max(
            len(inst["winning_intervention_locations"][0]), 
            len(inst["losing_intervention_locations"][0]),
            len(inst["steered_winning_intervention_locations"][0]),
            len(inst["steered_losing_intervention_locations"][0]),
            ) for inst in instances])
        max_seq_len = max([max(
            len(inst["winning_input_ids"]), 
            len(inst["losing_input_ids"]),
            len(inst["steered_winning_input_ids"]),
            len(inst["steered_losing_input_ids"]),
            ) for inst in instances]) 
        for inst in instances:
            winning_non_pad_len = len(inst["winning_input_ids"])
            losing_non_pad_len = len(inst["losing_input_ids"])
            steered_winning_non_pad_len = len(inst["steered_winning_input_ids"])
            steered_losing_non_pad_len = len(inst["steered_losing_input_ids"])
            
            _winning_intervention_mask = torch.ones_like(inst["winning_intervention_locations"][0])
            _winning_intervention_location_paddings = torch.tensor(
                [[winning_non_pad_len for _ in range(max_intervention_len - len(inst["winning_intervention_locations"][0]))]])
            _losing_intervention_mask = torch.ones_like(inst["losing_intervention_locations"][0])
            _losing_intervention_location_paddings = torch.tensor(
                [[losing_non_pad_len for _ in range(max_intervention_len - len(inst["losing_intervention_locations"][0]))]])
            _winning_intervention_mask_paddings = torch.tensor(
                [0 for _ in range(max_intervention_len - len(inst["winning_intervention_locations"][0]))])
            _losing_intervention_mask_paddings = torch.tensor(
                [0 for _ in range(max_intervention_len - len(inst["losing_intervention_locations"][0]))])
            inst["winning_intervention_locations"] = torch.cat(
                [inst["winning_intervention_locations"], _winning_intervention_location_paddings], dim=-1).int()
            inst["losing_intervention_locations"] = torch.cat(
                [inst["losing_intervention_locations"], _losing_intervention_location_paddings], dim=-1).int()
            inst["winning_intervention_masks"] = torch.cat([_winning_intervention_mask, _winning_intervention_mask_paddings], dim=-1).int()
            inst["losing_intervention_masks"] = torch.cat([_losing_intervention_mask, _losing_intervention_mask_paddings], dim=-1).int()
            inst["prompt_winning_intervention_masks"] = inst["winning_intervention_masks"].clone()
            inst["prompt_losing_intervention_masks"] = inst["losing_intervention_masks"].clone()
            inst["prompt_winning_intervention_masks"][inst["prompt_lengths"]:] = 0 # mask out the intervention locations after prompt length
            inst["prompt_losing_intervention_masks"][inst["prompt_lengths"]:] = 0 # mask out the intervention locations after prompt length
            
            # steered stuff for bidpo
            _steered_winning_intervention_mask = torch.ones_like(inst["steered_winning_intervention_locations"][0])
            _steered_winning_intervention_location_paddings = torch.tensor(
                [[steered_winning_non_pad_len for _ in range(max_intervention_len - len(inst["steered_winning_intervention_locations"][0]))]])
            _steered_losing_intervention_mask = torch.ones_like(inst["steered_losing_intervention_locations"][0])
            _steered_losing_intervention_location_paddings = torch.tensor(
                [[steered_losing_non_pad_len for _ in range(max_intervention_len - len(inst["steered_losing_intervention_locations"][0]))]])
            _steered_winning_intervention_mask_paddings = torch.tensor(
                [0 for _ in range(max_intervention_len - len(inst["steered_winning_intervention_locations"][0]))])
            _steered_losing_intervention_mask_paddings = torch.tensor(
                [0 for _ in range(max_intervention_len - len(inst["steered_losing_intervention_locations"][0]))])
            inst["steered_winning_intervention_locations"] = torch.cat(
                [inst["steered_winning_intervention_locations"], _steered_winning_intervention_location_paddings], dim=-1).int()
            inst["steered_losing_intervention_locations"] = torch.cat(
                [inst["steered_losing_intervention_locations"], _steered_losing_intervention_location_paddings], dim=-1).int()
            inst["steered_winning_intervention_masks"] = torch.cat([_steered_winning_intervention_mask, _steered_winning_intervention_mask_paddings], dim=-1).int()
            inst["steered_losing_intervention_masks"] = torch.cat([_steered_losing_intervention_mask, _steered_losing_intervention_mask_paddings], dim=-1).int()
            inst["prompt_steered_winning_intervention_masks"] = inst["steered_winning_intervention_masks"].clone()
            inst["prompt_steered_losing_intervention_masks"] = inst["steered_losing_intervention_masks"].clone()
            inst["prompt_steered_winning_intervention_masks"][inst["prompt_lengths"]:] = 0 # mask out the intervention locations after prompt length
            inst["prompt_steered_losing_intervention_masks"][inst["prompt_lengths"]:] = 0 # mask out the intervention locations after prompt length

            _winning_input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - winning_non_pad_len)])
            _losing_input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - losing_non_pad_len)])
            inst["winning_input_ids"] = torch.cat(
                (inst["winning_input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _winning_input_id_paddings)).int()
            inst["losing_input_ids"] = torch.cat(
                (inst["losing_input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _losing_input_id_paddings)).int()  
            
            _winning_label_paddings = torch.tensor([-100 for _ in range(max_seq_len - winning_non_pad_len+1)])
            _losing_label_paddings = torch.tensor([-100 for _ in range(max_seq_len - losing_non_pad_len+1)])
            inst["winning_labels"] = torch.cat((inst["winning_labels"], _winning_label_paddings))
            inst["losing_labels"] = torch.cat((inst["losing_labels"], _losing_label_paddings))
            
            inst["winning_attention_mask"] = (inst["winning_input_ids"] != self.tokenizer.pad_token_id).int()
            inst["losing_attention_mask"] = (inst["losing_input_ids"] != self.tokenizer.pad_token_id).int()
            
            # steered stuff for bidpo
            _steered_winning_input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - steered_winning_non_pad_len)])
            _steered_losing_input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - steered_losing_non_pad_len)])
            inst["steered_winning_input_ids"] = torch.cat(
                (inst["steered_winning_input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _steered_winning_input_id_paddings)).int()
            inst["steered_losing_input_ids"] = torch.cat(
                (inst["steered_losing_input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _steered_losing_input_id_paddings)).int()
            
            _steered_winning_label_paddings = torch.tensor([-100 for _ in range(max_seq_len - steered_winning_non_pad_len+1)])
            _steered_losing_label_paddings = torch.tensor([-100 for _ in range(max_seq_len - steered_losing_non_pad_len+1)])
            inst["steered_winning_labels"] = torch.cat((inst["steered_winning_labels"], _steered_winning_label_paddings))
            inst["steered_losing_labels"] = torch.cat((inst["steered_losing_labels"], _steered_losing_label_paddings))
            
            inst["steered_winning_attention_mask"] = (inst["steered_winning_input_ids"] != self.tokenizer.pad_token_id).int()
            inst["steered_losing_attention_mask"] = (inst["steered_losing_input_ids"] != self.tokenizer.pad_token_id).int()

        batch_inputs = self.data_collator(instances)
        return batch_inputs
    
    
def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer, df, 
    dataset_category="continuation",
    positions="all", # "all_prompt" or "all" or "f1+l1" (pyreft formatting)
    exclude_bos=True,
    prefix_length=1,
    **kwargs
):
    """Make dataset and collator for supervised fine-tuning with kl div loss."""
    if not exclude_bos:
        prefix_length = 0
    
    all_base_input_ids, all_intervention_locations, all_output_ids, all_concept_ids = [], [], [], []
    all_prompt_lengths = []
    all_examples_type = []
    for _, row in df.iterrows():
        _input, _output = row["input"], row["output"]
        _example_type = 1 if row["category"] == "positive" else -1


        # prepare input ids
        base_prompt = _input
        if isinstance(_output, float):
            _output = tokenizer.eos_token

        output_token = tokenizer(_output)["input_ids"]

        if len(output_token) < kwargs['output_length']:
            _output += "<|eot_id|>"

        base_input = base_prompt + _output
        base_prompt_ids = tokenizer(
            base_prompt, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_input_ids = tokenizer(
            base_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_length = len(base_input_ids)

        # output ids with prompt token mask
        output_ids = base_input_ids.clone()
        output_ids[:base_prompt_length] = -100

        if positions is None or positions == "all_prompt":
            intervention_locations = torch.tensor([[i for i in range(prefix_length, base_prompt_length)]])
        elif positions == "all":
            intervention_locations = torch.tensor([[i for i in range(prefix_length, base_length)]])
        
        else:
            first_n, last_n = parse_positions(positions)
            intervention_locations = get_intervention_locations(
                last_position=base_prompt_length - prefix_length, 
                first_n=first_n, 
                last_n=last_n,
                pad_mode="last",
                num_interventions=1,
                share_weights=True,
            )
            # shift intervention locations by prefix length
            shifted_intervention_locations = [[loc + prefix_length for loc in intervention_locations[0]]]
            intervention_locations = shifted_intervention_locations
        #if row["concept_id"] != -1:
        all_concept_ids.append(torch.tensor(row["concept_id"]))
        #else:
        #all_concept_ids.append(torch.tensor(random.choice(range(5))))
        all_intervention_locations.append(intervention_locations)
        all_base_input_ids.append(base_input_ids)
        all_output_ids.append(output_ids)
        all_examples_type.append(torch.tensor(_example_type))
        all_prompt_lengths.append(torch.tensor(base_prompt_length - 1)) # exclude bos token
    
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
        "prompt_lengths": all_prompt_lengths,
        "concept_id": all_concept_ids,
        "examples_types": all_examples_type,
    })
    train_dataset.set_format(
        type='torch', columns=[
            'input_ids', 'intervention_locations', 'prompt_lengths', 'concept_id', 'labels', 'examples_types'])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = InterventionDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_data_module_pos_neg(
    tokenizer: transformers.PreTrainedTokenizer, df, 
    dataset_category="continuation",
    positions="all", # "all_prompt" or "all" or "f1+l1" (pyreft formatting)
    exclude_bos=True,
    prefix_length=1,
    **kwargs
):
    """Make dataset and collator for supervised fine-tuning with kl div loss."""
    if not exclude_bos:
        prefix_length = 0
    
    all_base_input_ids, all_intervention_locations, all_output_ids, all_concept_ids = [], [], [], []
    all_prompt_lengths = []
    all_examples_type = []
    for _, row in df.iterrows():
        if random.random() < 0.5:
            _input, _output = row["input"], row["winning_output"]
            _example_type = 1 
        else:
            _input, _output = row["steered_input"], row["losing_output"]
            _example_type = -1


        # prepare input ids
        base_prompt = _input
        if isinstance(_output, float):
            _output = tokenizer.eos_token
        output_token = tokenizer(_output)["input_ids"]
        if len(output_token) < kwargs['output_length']:
            _output += "<|eot_id|>"

        base_input = base_prompt + _output
        base_prompt_ids = tokenizer(
            base_prompt, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_input_ids = tokenizer(
            base_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_length = len(base_input_ids)

        # output ids with prompt token mask
        output_ids = base_input_ids.clone()
        output_ids[:base_prompt_length] = -100

        if positions is None or positions == "all_prompt":
            intervention_locations = torch.tensor([[i for i in range(prefix_length, base_prompt_length)]])
        elif positions == "all":
            intervention_locations = torch.tensor([[i for i in range(prefix_length, base_length)]])
        
        else:
            first_n, last_n = parse_positions(positions)
            intervention_locations = get_intervention_locations(
                last_position=base_prompt_length - prefix_length, 
                first_n=first_n, 
                last_n=last_n,
                pad_mode="last",
                num_interventions=1,
                share_weights=True,
            )
            # shift intervention locations by prefix length
            shifted_intervention_locations = [[loc + prefix_length for loc in intervention_locations[0]]]
            intervention_locations = shifted_intervention_locations
        #if row["concept_id"] != -1:
        all_concept_ids.append(torch.tensor(row["concept_id"]))
        #else:
        #all_concept_ids.append(torch.tensor(random.choice(range(5))))
        all_intervention_locations.append(intervention_locations)
        all_base_input_ids.append(base_input_ids)
        all_output_ids.append(output_ids)
        all_examples_type.append(torch.tensor(_example_type))
        all_prompt_lengths.append(torch.tensor(base_prompt_length - 1)) # exclude bos token
    
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
        "prompt_lengths": all_prompt_lengths,
        "concept_id": all_concept_ids,
        "examples_types": all_examples_type,
    })
    train_dataset.set_format(
        type='torch', columns=[
            'input_ids', 'intervention_locations', 'prompt_lengths', 'concept_id', 'labels', 'examples_types'])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = InterventionDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_preference_data_module(
    tokenizer: transformers.PreTrainedTokenizer, df, 
    dataset_category="continuation",
    positions="all", # "all_prompt" or "all" or "f1+l1" (pyreft formatting)
    exclude_bos=True,
    prefix_length=1,
    **kwargs
):
    if not exclude_bos:
        prefix_length = 0
    
    all_winning_base_input_ids, all_winning_intervention_locations, all_winning_output_ids, \
        all_losing_base_input_ids, all_losing_intervention_locations, all_losing_output_ids = [], [], [], [], [], []
    all_prompt_lengths = []
    for _, row in df.iterrows():
        _input, _winning_output, _losing_output = row["input"], row["winning_output"], row["losing_output"]

        # prepare input ids
        base_prompt = _input
        if isinstance(_winning_output, float):
            _winning_output = tokenizer.eos_token
        if isinstance(_losing_output, float):
            _losing_output = tokenizer.eos_token
        base_winning_input = base_prompt + _winning_output
        base_losing_input = base_prompt + _losing_output

        base_prompt_ids = tokenizer(
            base_prompt, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        winning_base_input_ids = tokenizer(
            base_winning_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        losing_base_input_ids = tokenizer(
            base_losing_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]

        base_prompt_length = len(base_prompt_ids)
        winning_base_length = len(winning_base_input_ids)
        losing_base_length = len(losing_base_input_ids)

        # output ids with prompt token mask
        winning_output_ids = winning_base_input_ids.clone()
        winning_output_ids[:base_prompt_length] = -100
        losing_output_ids = losing_base_input_ids.clone()
        losing_output_ids[:base_prompt_length] = -100

        if positions is None or positions == "all_prompt":
            winning_intervention_locations = torch.tensor([[i for i in range(prefix_length, base_prompt_length)]])
            losing_intervention_locations = torch.tensor([[i for i in range(prefix_length, base_prompt_length)]])
        elif positions == "all":
            winning_intervention_locations = torch.tensor([[i for i in range(prefix_length, winning_base_length)]])
            losing_intervention_locations = torch.tensor([[i for i in range(prefix_length, losing_base_length)]])
        else:
            raise NotImplementedError(f"Positions {positions} not implemented")
        
        all_winning_base_input_ids.append(winning_base_input_ids)
        all_losing_base_input_ids.append(losing_base_input_ids)
        all_winning_intervention_locations.append(winning_intervention_locations)
        all_losing_intervention_locations.append(losing_intervention_locations)
        all_winning_output_ids.append(winning_output_ids)
        all_losing_output_ids.append(losing_output_ids)
        all_prompt_lengths.append(torch.tensor(base_prompt_length - 1)) # exclude bos token
    
    train_dataset = datasets.Dataset.from_dict({
        "winning_input_ids": all_winning_base_input_ids,
        "losing_input_ids": all_losing_base_input_ids,
        "winning_intervention_locations": all_winning_intervention_locations,
        "losing_intervention_locations": all_losing_intervention_locations,
        "winning_labels": all_winning_output_ids,
        "losing_labels": all_losing_output_ids,
        "prompt_lengths": all_prompt_lengths,
    })
    train_dataset.set_format(
        type='torch', columns=[
            'winning_input_ids', 
            'losing_input_ids', 
            'winning_intervention_locations', 
            'losing_intervention_locations', 
            'winning_labels', 
            'losing_labels', 
            'prompt_lengths'])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = PreferenceInterventionDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_bi_preference_data_module(
    tokenizer: transformers.PreTrainedTokenizer, df, 
    dataset_category="continuation",
    positions="all", # "all_prompt" or "all" or "f1+l1" (pyreft formatting)
    exclude_bos=True,
    prefix_length=1,
    **kwargs
):
    if not exclude_bos:
        prefix_length = 0
    
    all_winning_base_input_ids, all_winning_intervention_locations, all_winning_output_ids, \
        all_losing_base_input_ids, all_losing_intervention_locations, all_losing_output_ids = [], [], [], [], [], []
    # steered stuff for bidpo
    all_steered_winning_base_input_ids, all_steered_winning_intervention_locations, all_steered_winning_output_ids, \
        all_steered_losing_base_input_ids, all_steered_losing_intervention_locations, all_steered_losing_output_ids = [], [], [], [], [], []
    
    all_prompt_lengths = []
    all_steered_prompt_lengths = []
    all_concept_ids = []
    for _, row in df.iterrows():
        _input, _steered_input, _winning_output, _losing_output, concept_id = \
            row["input"], row["steered_input"], row["winning_output"], row["losing_output"], row["concept_id"]

        # prepare input ids
        base_prompt = _input
        steered_prompt = _steered_input
        if isinstance(_winning_output, float):
            _winning_output = tokenizer.eos_token
        if isinstance(_losing_output, float):
            _losing_output = tokenizer.eos_token
        base_winning_input = base_prompt + _winning_output
        base_losing_input = base_prompt + _losing_output

        # steered stuff for bidpo
        steered_winning_input = steered_prompt + _losing_output
        steered_losing_input = steered_prompt + _winning_output

        base_prompt_ids = tokenizer(
            base_prompt, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        steered_prompt_ids = tokenizer(
            steered_prompt, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        winning_base_input_ids = tokenizer(
            base_winning_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        losing_base_input_ids = tokenizer(
            base_losing_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        
        # steered stuff for bidpo
        steered_winning_input_ids = tokenizer(
            steered_winning_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        steered_losing_input_ids = tokenizer(
            steered_losing_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]

        base_prompt_length = len(base_prompt_ids)
        winning_base_length = len(winning_base_input_ids)
        losing_base_length = len(losing_base_input_ids)

        # steered stuff for bidpo
        steered_prompt_length = len(steered_prompt_ids)
        steered_winning_length = len(steered_winning_input_ids)
        steered_losing_length = len(steered_losing_input_ids)

        # output ids with prompt token mask
        winning_output_ids = winning_base_input_ids.clone()
        winning_output_ids[:base_prompt_length] = -100
        losing_output_ids = losing_base_input_ids.clone()
        losing_output_ids[:base_prompt_length] = -100

        # steered stuff for bidpo
        steered_winning_output_ids = steered_winning_input_ids.clone()
        steered_winning_output_ids[:steered_prompt_length] = -100
        steered_losing_output_ids = steered_losing_input_ids.clone()
        steered_losing_output_ids[:steered_prompt_length] = -100

        if positions is None or positions == "all_prompt":
            winning_intervention_locations = torch.tensor([[i for i in range(prefix_length, base_prompt_length)]])
            losing_intervention_locations = torch.tensor([[i for i in range(prefix_length, base_prompt_length)]])
            # steered stuff for bidpo
            steered_winning_intervention_locations = torch.tensor([[i for i in range(prefix_length, steered_prompt_length)]])
            steered_losing_intervention_locations = torch.tensor([[i for i in range(prefix_length, steered_prompt_length)]])
        elif positions == "all":
            winning_intervention_locations = torch.tensor([[i for i in range(prefix_length, winning_base_length)]])
            losing_intervention_locations = torch.tensor([[i for i in range(prefix_length, losing_base_length)]])
            # steered stuff for bidpo
            steered_winning_intervention_locations = torch.tensor([[i for i in range(prefix_length, steered_winning_length)]])
            steered_losing_intervention_locations = torch.tensor([[i for i in range(prefix_length, steered_losing_length)]])
        else:
            raise NotImplementedError(f"Positions {positions} not implemented")
        
        all_winning_base_input_ids.append(winning_base_input_ids)
        all_losing_base_input_ids.append(losing_base_input_ids)
        all_winning_intervention_locations.append(winning_intervention_locations)
        all_losing_intervention_locations.append(losing_intervention_locations)
        all_winning_output_ids.append(winning_output_ids)
        all_losing_output_ids.append(losing_output_ids)
        all_prompt_lengths.append(torch.tensor(base_prompt_length - 1)) # exclude bos token

        # steered stuff for bidpo
        all_steered_winning_base_input_ids.append(steered_winning_input_ids)
        all_steered_losing_base_input_ids.append(steered_losing_input_ids)
        all_steered_winning_intervention_locations.append(steered_winning_intervention_locations)
        all_steered_losing_intervention_locations.append(steered_losing_intervention_locations)
        all_steered_winning_output_ids.append(steered_winning_output_ids)
        all_steered_losing_output_ids.append(steered_losing_output_ids)
        all_steered_prompt_lengths.append(torch.tensor(steered_prompt_length - 1)) # exclude bos token
        all_concept_ids.append(torch.tensor(concept_id))
    
    train_dataset = datasets.Dataset.from_dict({
        "winning_input_ids": all_winning_base_input_ids,
        "losing_input_ids": all_losing_base_input_ids,
        "winning_intervention_locations": all_winning_intervention_locations,
        "losing_intervention_locations": all_losing_intervention_locations,
        "winning_labels": all_winning_output_ids,
        "losing_labels": all_losing_output_ids,
        "prompt_lengths": all_prompt_lengths,
        # steered stuff for bidpo
        "steered_winning_input_ids": all_steered_winning_base_input_ids,
        "steered_losing_input_ids": all_steered_losing_base_input_ids,
        "steered_winning_intervention_locations": all_steered_winning_intervention_locations,
        "steered_losing_intervention_locations": all_steered_losing_intervention_locations,
        "steered_winning_labels": all_steered_winning_output_ids,
        "steered_losing_labels": all_steered_losing_output_ids,
        "steered_prompt_lengths": all_steered_prompt_lengths,
        "concept_id": all_concept_ids,
    })
    train_dataset.set_format(
        type='torch', columns=[
            'winning_input_ids', 
            'losing_input_ids', 
            'winning_intervention_locations', 
            'losing_intervention_locations', 
            'winning_labels', 
            'losing_labels', 
            'prompt_lengths',
            # steered stuff for bidpo
            'steered_winning_input_ids',
            'steered_losing_input_ids',
            'steered_winning_intervention_locations',
            'steered_losing_intervention_locations',
            'steered_winning_labels', 'steered_losing_labels', 'steered_prompt_lengths', 'concept_id'])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = BiPreferenceInterventionDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

