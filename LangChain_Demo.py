#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:59:58 2024

@author: jyb
"""
import sys
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain



class MyAgent:
    def __init__(self, model_name = "gpt-3.5-turbo", temperature = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name = model_name,
                              temperature = temperature)
        
        self.system_prompt = "You are world class technical documentation writer."
        
    def Chat(self, user_message, verbose=False):
        if user_message is None:
            print("user_message is None !")
            return False
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", user_message)
        ])
        
        chain = prompt | self.llm
        
        completion = chain.invoke({"user_message" : user_message,
                                   "system_prompt" : self.system_prompt})
        
        self.ShowUser(user_message)
        self.ShowCompletion(completion, verbose = verbose)
        return True
        
        
        
    def ShowUser(self, user_message):
        print("#USER# :<<")
        print(user_message, "\n\n")
        return True
    
    def ShowGPT(self, gpt_message):
        print("#GPT# :>>")
        print(gpt_message)
        return True

    def ShowCompletion(self, completion, verbose=False):
        self.ShowGPT(completion.content)
        if verbose:
            self.ShowVerbose(completion)
        return True
            
            
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
        return True;
            
    def Retrieval(self, user_message):
        loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
        docs = loader.load()
        
        embeddings = OpenAIEmbeddings()
        
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            
            Question: {input}""")

        document_chain = create_stuff_documents_chain(self.llm, prompt)
        # gpt_message = document_chain.invoke({
        #        "input": user_message,
        #        "context": [Document(page_content="langsmith can let you visualize test results")]
        #     })
        
        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": user_message})
        
        self.ShowUser(user_message)
        self.ShowGPT(response["answer"])
        return True



def Run(operation):
    agent = MyAgent()
    
    
    chat_message = "how can langsmith help with testing?"
    retrieval_message = "how can langsmith help with testing ?"
    
    
    
    messages = {
        "Chat": chat_message,
        "Retrieval": retrieval_message
    }
    operations = {
        "Chat": agent.Chat,
        "Retrieval": agent.Retrieval
    }
    
    message = messages.get(operation, None)
    func = operations.get(operation, None)
    if func:
        return func(message)
    else:
        return "Unsupported operation"

        
def main():
    """
    langchain demo
    """
    args = sys.argv[1:]
    if len(args) > 0:
        param1 = str(args[0])
        #print(param1)
    
    
    operation = "Retrieval"
    
    Run(operation)

    
    
if __name__ == "__main__":
    main()
    





