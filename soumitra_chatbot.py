# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 00:33:38 2019

@author: Soumitra
"""

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


chatbot = ChatBot("sample_chatbot")

# First, lets train our bot with some data
trainer = ChatterBotCorpusTrainer(chatbot)

trainer.train('chatterbot.corpus.english')
y=1
while(y):
    inp=input()
    if (inp):
        print(chatbot.get_response(inp))
    else:
        break



