#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:34:44 2024

@author: jyb
"""
import os
import openai
 
openai.api_key = os.getenv("OPENAI_API_KEY")

print(openai.api_key)

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)