import os
from jinja2 import Environment, FileSystemLoader

# Get the directory where this file is located
PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))
env = Environment(loader=FileSystemLoader(PROMPT_DIR), autoescape=False)
# Add built-in zip function to Jinja environment for looping over multiple lists
env.globals.update(zip=zip)


def load_prompt(template_name: str, **kwargs) -> str:
    """Load and render a prompt template"""
    template = env.get_template(template_name)
    return template.render(**kwargs)
