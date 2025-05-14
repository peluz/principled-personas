import re
from collections import Counter
from pylatexenc.latexwalker import LatexWalker, LatexMacroNode
import string
import signal


rep = {"#": "", ":": ""}
rep = dict((re.escape(k), v) for k, v in rep.items())
pattern = re.compile("|".join(rep.keys()))

def signal_handler(signum, frame):
    raise Exception("Timed out!")

def extract_answer(generation, n_options):
    '''
    generation: LLM response
    n_options: number of multiple choice options
    '''
    generation = re.sub(r'\n+', ' ', generation)
    generation = pattern.sub(lambda m: rep[re.escape(m.group(0))], generation) 
    alpha = string.ascii_uppercase
    option_range = f"A-{alpha[n_options-1]}"
    boxed = re.search(rf'(?<=\\boxed\{{)[{option_range}](?=\}})', generation)
    only = re.search(rf'^[{option_range}]$', generation)
    correct = re.search(rf'\([{option_range}]\)(?= is (the )*correct)', generation)
    answer = re.search(rf'\([{option_range}]\)(?= is the (answer|solution))', generation)
    inv_answer = re.search(rf'(?<=answer is )\([{option_range}]\)', generation)
    inv_solution = re.search(rf'(?<=solution is )\([{option_range}]\)', generation)
    cap_answer = re.search(rf'(?<=Answer )\([{option_range}]\)', generation)
    parenthesis = re.findall(rf'\([{option_range}]\)', generation)
    if boxed:
        return boxed.group()[0]
    elif correct:
        return correct.group()[1]
    elif answer:
        return answer.group()[1]
    elif inv_answer:
        return inv_answer.group()[1]
    elif inv_solution:
        return inv_solution.group()[1]
    elif cap_answer:
        return cap_answer.group()[1]
    elif only:
        return only.group()[0]
    elif len(parenthesis) > 0:
        if len(parenthesis) == 1: return parenthesis[0][1]
        else:
            options = Counter(parenthesis).most_common(1)
            return options[0][0][1]
    else:
        return extract_answer_fallback(generation, option_range)
    
def extract_answer_fallback(generation, option_range):
    '''
    generation: LLM response
    option_range: range of multiple choice options
    '''
    options = re.findall(rf'(?<![A-Za-z\.])[{option_range}][\.\) ]', generation)
    if len(options) == 0: return -1
    options = Counter(options).most_common(1)
    return options[0][0][0]

    
def extract_gsm8k(generation):
    '''
    generation: LLM response
    '''
    # last_line = generation.rstrip(" \n").split("\n")[-1]
    generation = re.sub(r'\.+', '.', generation).rstrip(".")
    generation = re.sub(r'\.,', '.', generation)
    number = re.findall(r'(-?\$?[0-9.,]{2,})|(-?[0-9]+)', generation)
    if len(number) >0:
        answer = number[-1]
        answer = answer[0] if answer[0] != "" else answer[1]
        answer_clean = answer.replace(",", "").replace("$", "").rstrip(".").replace(".0.0", ".0")
        try:
            return float(answer_clean)
        except:
            print(generation)
            print("------------")
            print(answer)
            print("------------")
            print(answer_clean)
            print("------------")
            raise Exception 
    else:
        return "nan"
    
def extract_math(generation):
    '''
    generation: LLM response
    '''
    if "boxed" not in generation:
        return "nan"
    number = re.search(r'(?<=\\boxed\{)\d+(?=\})', generation)
    if number: return number.group()
    try:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(1)   # 1 second
        try:
            nodes =  LatexWalker(generation).get_latex_nodes()[0]
        except Exception:
            return "nan"
        revisit_nodes = []
        for i, node in enumerate(nodes):
            if node.nodeType() == LatexMacroNode and node.macroname == "boxed":
                try:
                    return nodes[i+1].latex_verbatim()[1:-1]
                except IndexError:
                   return "nan"        
            if hasattr(node, 'nodelist'):
                revisit_nodes.append(node)
        for node in revisit_nodes:
            if hasattr(node, 'nodelist'):
                for i, n in enumerate(node.nodelist):
                    if n.nodeType() == LatexMacroNode and n.macroname == "boxed":
                        try:
                            return node.nodelist[i+1].latex_verbatim()[1:-1]
                        except IndexError:
                           return "nan"
    except RecursionError:
        return "nan"
    return "nan"

