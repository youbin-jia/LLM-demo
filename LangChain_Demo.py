#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:59:58 2024

@author: jyb
"""
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MyAgent:
    def __init__(self, model_name = "gpt-3.5-turbo", temperature = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name = model_name,
                              temperature = temperature)
        
        self.system_prompt = "You are world class technical documentation writer."
        
    def Chat(self, user_message, verbose=False):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("user", "{user_message}")
        ])
        chain = prompt | self.llm
        
        completion = chain.invoke({"user_message" : user_message,
                                   "system_prompt" : self.system_prompt})
        
        self.ShowUser(user_message)
        self.ShowCompletion(completion, verbose = verbose)
        
        
        
    def ShowUser(self, user_message):
        print("#USER# :<<")
        print(user_message, "\n\n")

    def ShowCompletion(self, completion, verbose=False):
        print("#GPT# :>>")
        print(completion.content)
        if verbose:
            self.ShowVerbose(completion)
            
            
    def ShowVerbose(self, completion):
        print("\n\n****************************verbose*************************")
        token_usage = completion.response_metadata["token_usage"]
        
        print("token_usage:")
        print("{:4}{:<20}:{:4}{:<20}".format(" ", "completion_tokens", " ", token_usage["completion_tokens"]))
        print("{:4}{:<20}:{:4}{:<20}".format(" ", "prompt_tokens", " ", token_usage["prompt_tokens"]))
        print("{:4}{:<20}:{:4}{:<20}".format(" ", "total_tokens", " ", token_usage["total_tokens"]))
        print("")
        
        model_name  = completion.response_metadata["model_name"]
        system_fingerprint = completion.response_metadata["system_fingerprint"]
        finish_reason = completion.response_metadata["finish_reason"]
        logprobs = completion.response_metadata["logprobs"]
        
        print("{:<20}:{:4}{:<50}".format("model_name", " ", model_name))
        print("{:<20}:{:4}{:<50}".format("system_fingerprint", " ", system_fingerprint))
        print("{:<20}:{:4}{:<50}".format("finish_reason", " ", finish_reason))
        if logprobs is not None:
            print("{:<20}:{:4}{:<50}".format("logprobs", " ", logprobs))
            
            
def main():
    """
    langchain demo
    """
    args = sys.argv[1:]
    if len(args) > 0:
        param1 = str(args[0])
        #print(param1)
    
    
    user_message = "how can langsmith help with testing?"
    
    agent = MyAgent()
    agent.Chat(user_message)
    
    
if __name__ == "__main__":
    main()
    





