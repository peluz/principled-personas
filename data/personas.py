import itertools

EDUCATION_PERSONAS = ["an uneducated person",
        "a person with primary school level education",
        "a person with middle school level education",
        "a person with high school level education",
        "a person with college-level education",
        "a person with graduate level education",]

EXPERTS = {
    "mmlu_pro": ['an expert in biology', 'an expert in business', 'an expert in chemistry', 'an expert in computer science', 'an expert in economics', 'an expert in engineering', 'an expert in health', 'an expert in history', 'an expert in law', 'an expert in math', 'an expert in miscellaneous fields including international relations, sociology, accounting, and human sexuality', 'an expert in philosophy', 'an expert in physics', 'an expert in psychology'],
    "truthfulqa": ["an expert in fact-checking", "an expert in criptography", "an expert in marine biology", "an expert in urban planning", "an expert in chess", "an expert in quantum mechanics"],
    "bigbench": ["an expert in logic grid puzzles", "an expert in multi-step implicit reasoning", "an expert in tracking shuffled objects", "an expert in applying contextual information", "an expert in sudoku", "an expert in inductive reasoning", "an expert in communicating effectively", "an expert in hunting"],
    "gsm8k": ["an expert in math", 'an expert in health', 'an expert in history', 'an expert in law', 'an expert in philosophy',  'an expert in psychology'],
    "math": ["an expert in math", "an expert in algebra", "an expert in counting and probability", "an expert in geometry", "an expert in intermediate algebra", "an expert in number theory", "an expert in prealgebra", "an expert in precalculus", 'an expert in health', 'an expert in history', 'an expert in law', 'an expert in philosophy',  'an expert in psychology'],
}

NAMES = ["Alexander",
        "Victor",
        "Muhammad",
        "Kai",
        "Amit",
        "Gustavo",
        "Anastasia",
        "Isabelle",
        "Fatima",
        "Yumi",
        "Aparna",
        "Larissa",]

COLOR_PERSONAS = ["favorite color is red", "favorite color is blue", "favorite color is green",
                  "favorite color is yellow", "favorite color is black", "favorite color is white"]

NO_PERSONA = [""]

def persona_to_cat(persona):
    if persona == "empty": return persona
    elif persona in itertools.chain.from_iterable(EXPERTS.values()): return "expert"
    elif persona in COLOR_PERSONAS: return "color"
    elif persona in NAMES: return "name"
    elif persona in EDUCATION_PERSONAS: return "education"
    elif persona in ["level1", "level2", "level3"]: return "expert_levels"