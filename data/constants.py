DATASETS = [
    "truthfulqa", "bigbench", "gsm8k", "math", "mmlu_pro",
]


BIGBENCH_SUBTASKS = [
#  'hinglish_toxicity',
#  'entailed_polarity_hindi',
#  'cryobiology_spanish',
 'logic_grid_puzzle',
#  'mnist_ascii',
#  "color",
#  'medical_questions_russian',
#  'english_russian_proverbs',
#  'misconceptions_russian',
#  'parsinlu_qa',
#  'persian_idioms',
#  'swedish_to_german_proverbs',
#  'swahili_english_proverbs',
 'strategyqa',
 'tracking_shuffled_objects',
#  'kanji_ascii',
#  'kannada',
#  'indic_cause_and_effect',
#  'hindu_knowledge',
 'contextual_parametric_knowledge_conflicts',
#  'checkmate_in_one',
]

MODEL_ORDER = ["gemma-2-2b-it", "gemma-2-9b-it", "gemma-2-27b-it", 
               "Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct",
               "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct"
              ]

CATEGORY_ORDER = ["name", "color", "education", "out-expert", "experts", "generic-expert", "in-expert", "expert_levels", "empty"]