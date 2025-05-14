from string import Template

YOU_ARE = Template("You are $persona")

YOUR = Template("Your $persona")

PERSONA_INSTRUCTION = """{persona_desc}. Your responses must adhere to the following constraints:
1. If your persona implies domain expertise, provide responses that reflect its specialized knowledge.
2. Your responses should align with the knowledge level and domain knowledge expected from this persona.
3. Attributes that do not contribute to the task should not influence reasoning, knowledge, or output quality."""

PERSONA_REFINING = """Now, refine your response while adopting the persona: {persona_desc}. Your revised response must adhere to these constraints:
1. If your persona implies domain expertise, refine the response to reflect the persona's specialized knowledge.
2. Your refined response should align with the knowledge level and domain knowledge expected from this persona.
3. Attributes that do not contribute to the task should not influence reasoning, knowledge, or output quality of the refined response.
4. Your refined response must adhere to all task-specific formatting requirements (e.g., multiple-choice answers should include the correct letter option, mathematical expressions must be properly formatted, and structured output should follow the specified format).

Your refined response should **not** reference or acknowledge the original response—answer as if this is your first response. Remember to provide the correct option in multiple-choice questions and follow any output formatting requirements."""

PERSONA_REFINING_BASIC = """Now, refine your response while adopting the persona: {persona_desc}. Your refined response should **not** reference or acknowledge the original response—answer as if this is your first response. Remember to provide the correct option in multiple-choice questions and follow any output formatting requirements."""

mitigation_template = {
    "instruction": PERSONA_INSTRUCTION,
    "refine":PERSONA_REFINING,
    "refine_basic": PERSONA_REFINING_BASIC
}

def persona_to_desc(persona):
    if "favorite color is" in persona:
        return YOUR.substitute({"persona": persona})
    elif persona == "":
        return None
    else:
        return YOU_ARE.substitute({"persona": persona})
