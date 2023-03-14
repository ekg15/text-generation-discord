import gc
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import gradio as gr
import torch

import modules.chat as chat
import modules.extensions as extensions_module
import modules.shared as shared
import modules.ui as ui
from modules.html_generator import generate_chat_html
from modules.models import load_model, load_soft_prompt
from modules.text_generation import generate_reply

if (shared.args.chat or shared.args.cai_chat) and not shared.args.no_stream:
    print('Warning: chat mode currently becomes somewhat slower with text streaming on.\nConsider starting the web UI with the --no-stream option.\n')
    
# Loading custom settings
settings_file = None
if shared.args.settings is not None and Path(shared.args.settings).exists():
    settings_file = Path(shared.args.settings)
elif Path('settings.json').exists():
    settings_file = Path('settings.json')
if settings_file is not None:
    print(f"Loading settings from {settings_file}...")
    new_settings = json.loads(open(settings_file, 'r').read())
    for item in new_settings:
        shared.settings[item] = new_settings[item]

def get_available_models():
    if shared.args.flexgen:
        return sorted([re.sub('-np$', '', item.name) for item in list(Path('models/').glob('*')) if item.name.endswith('-np')], key=str.lower)
    else:
        return sorted([item.name for item in list(Path('models/').glob('*')) if not item.name.endswith(('.txt', '-np', '.pt'))], key=str.lower)

def get_available_presets():
    return sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('presets').glob('*.txt'))), key=str.lower)

def get_available_characters():
    return ['None'] + sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('characters').glob('*.json'))), key=str.lower)

def get_available_extensions():
    return sorted(set(map(lambda x : x.parts[1], Path('extensions').glob('*/script.py'))), key=str.lower)

def get_available_softprompts():
    return ['None'] + sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('softprompts').glob('*.zip'))), key=str.lower)

def load_model_wrapper(selected_model):
    if selected_model != shared.model_name:
        shared.model_name = selected_model
        shared.model = shared.tokenizer = None
        if not shared.args.cpu:
            gc.collect()
            torch.cuda.empty_cache()
        shared.model, shared.tokenizer = load_model(shared.model_name)

    return selected_model

def load_preset_values(preset_menu, return_dict=False):
    generate_params = {
        'do_sample': True,
        'temperature': 1,
        'top_p': 1,
        'typical_p': 1,
        'repetition_penalty': 1,
        'top_k': 50,
        'num_beams': 1,
        'penalty_alpha': 0,
        'min_length': 0,
        'length_penalty': 1,
        'no_repeat_ngram_size': 0,
        'early_stopping': False,
    }
    with open(Path(f'presets/{preset_menu}.txt'), 'r') as infile:
        preset = infile.read()
    for i in preset.splitlines():
        i = i.rstrip(',').strip().split('=')
        if len(i) == 2 and i[0].strip() != 'tokens':
            generate_params[i[0].strip()] = eval(i[1].strip())

    generate_params['temperature'] = min(1.99, generate_params['temperature'])

    if return_dict:
        return generate_params
    else:
        return generate_params['do_sample'], generate_params['temperature'], generate_params['top_p'], generate_params['typical_p'], generate_params['repetition_penalty'], generate_params['top_k'], generate_params['min_length'], generate_params['no_repeat_ngram_size'], generate_params['num_beams'], generate_params['penalty_alpha'], generate_params['length_penalty'], generate_params['early_stopping']

def upload_soft_prompt(file):
    with zipfile.ZipFile(io.BytesIO(file)) as zf:
        zf.extract('meta.json')
        j = json.loads(open('meta.json', 'r').read())
        name = j['name']
        Path('meta.json').unlink()

    with open(Path(f'softprompts/{name}.zip'), 'wb') as f:
        f.write(file)

    return name

available_models = get_available_models()
available_presets = get_available_presets()
available_characters = get_available_characters()
available_softprompts = get_available_softprompts()

# Default extensions
extensions_module.available_extensions = get_available_extensions()
if shared.args.chat or shared.args.cai_chat:
    for extension in shared.settings['chat_default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)
else:
    for extension in shared.settings['default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)
if shared.args.extensions is not None and len(shared.args.extensions) > 0:
    extensions_module.load_extensions()

# Default model
if shared.args.model is not None:
    shared.model_name = shared.args.model
else:
    if len(available_models) == 0:
        print('No models are available! Please download at least one.')
        sys.exit(0)
    elif len(available_models) == 1:
        i = 0
    else:
        print('The following models are available:\n')
        for i, model in enumerate(available_models):
            print(f'{i+1}. {model}')
        print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
        i = int(input())-1
        print()
    shared.model_name = available_models[i]
shared.model, shared.tokenizer = load_model(shared.model_name)

# Default UI settings
gen_events = []
default_preset = shared.settings['presets'][next((k for k in shared.settings['presets'] if re.match(k.lower(), shared.model_name.lower())), 'default')]
default_text = shared.settings['prompts'][next((k for k in shared.settings['prompts'] if re.match(k.lower(), shared.model_name.lower())), 'default')]
title ='Text generation web UI'
description = '\n\n# Text generation lab\nGenerate text using Large Language Models.\n'
suffix = '_pygmalion' if 'pygmalion' in shared.model_name.lower() else ''
