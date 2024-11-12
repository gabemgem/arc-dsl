from unsloth import FastLanguageModel
import torch
max_seq_length = 4096
dtype=None
load_in_4bit=True

import sys
adapter = sys.argv[1]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = adapter,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    dtype = dtype,
    trust_remote_code = True, 
    )

merged_model = model.merge_and_unload()

output_dir = f"fullmodels/{adapter}"
merged_model.save_pretrained(output_dir, safe_serialization=False)
tokenizer.save_pretrained(output_dir)
