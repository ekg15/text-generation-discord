import gc
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import torch

import modules.chat as chat
import modules.extensions as extensions_module
import modules.shared as shared
import modules.ui as ui
from modules.html_generator import generate_chat_html
from modules.models import load_model, load_soft_prompt
from modules.text_generation import generate_reply

import discord
from discord.ext import commands
from typing import Tuple
import os,argparse
import json
from pathlib import Path

if not shared.args.cpu:
    gc.collect()
    torch.cuda.empty_cache()
    shared.model, shared.tokenizer = load_model("llama-30b-hf")

y = generate_reply("a recipe for a peanut butter and jelly sandwich is", 200, True, .7, 1, 1, 1.05, 40, 0, 0, 1, 0, 1, False, eos_token=None, stopping_string=None)
print(list(y))

@bot.event
async def on_message(message):
        #Default config values
            temperature= 0.9
            top_p= 0.75
            max_len=256
            repetition_penalty_range=1024
            repetition_penalty=1.15
            repetition_penalty_slope=0.7
            #semaphore
            global sem
            if message.author == bot.user:
                return
            if (sem==1):
                time.sleep(5)
            sem=1
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            botid=("<@%d>" % bot.user.id)
            if message.content.startswith(botid):
                query = message.content[len(botid):].strip()
                origquery=query
                query=query[:1024] #limit query lenght
                jsonEnd=query.find('}')
                rawpos=query.find('raw')
                if (jsonEnd > rawpos):
                    jsonEnd=0 # this is not configuration
                try:
                    if (jsonEnd>0): # json config present, parse
                        config=query[:jsonEnd+1]
                        query=query[jsonEnd+1:].strip()
                        config=json.loads(config)
                        if not (config.get('temperature') is None):
                            temperature=float(config['temperature'])
                        if not (config.get('top_p') is None):
                            top_p=float(config['top_p'])
                        if not (config.get('max_len') is None):
                            max_len=int(config['max_len'])
                            if (max_len>2048): max_len=2048
                        if not (config.get('repetition_penalty_range') is None):
                            repetition_penalty_range=int(config['repetition_penalty_range'])
                        if not (config.get('repetition_penalty') is None):
                            repetition_penalty_range=float(config['repetition_penalty'])
                        if not (config.get('repetition_penalty_slope') is None):
                            repetition_penalty_range=float(config['repetition_penalty_slope'])
                except Exception as e:
                    if local_rank == 0:
                        msg = f"{message.author.mention} Error parsing the Json config: %s" % str(e)
                        log(msg)
                        await message.channel.send(msg)
                    sem=0
                    return

                if (query.startswith('raw ')): # Raw prompt
                    query = query[4:]
                else: # Wrap prompt in question
                    query ='The answer for "%s" would be: ' % query
                prompts = [query]
                print(prompts)
                torch.cuda.empty_cache()
                async with message.channel.typing():
                    try:
                        results = generate_reply(query, 200, True, .7, 1, 1, 1.05, 40, 0, 0, 1, 0, 1, False, eos_token=None, stopping_string=None)
                    except:
                        if local_rank == 0:
                            await message.channel.send(msg)
                        sem=0
                        return
                if local_rank == 0:
                   sem=0
                   return
                log("---"+str(origquery))

                for result in results:
                    msg = f"{message.author.mention} %s" % result
                    log(msg)
                    if len(msg)>1500:
                        for i in range(0,len(msg),1500):
                            await message.channel.send(msg[i:i+1500])
                    else:
                        await message.channel.send(msg)
            sem=0
token=open('discordtoken.txt').read().strip()

bot.run(token)
