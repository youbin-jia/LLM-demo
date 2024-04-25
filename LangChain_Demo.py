#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:59:58 2024

@author: jyb
"""
import os
import sys
import pytest
import requests
import traceback

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_openai import OpenAI
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
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent, create_react_agent
from langchain.agents import AgentExecutor

from gaode_navigation import route_planning, search_nearby


class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'




tavily_api_key = "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"


class SearchNearbyInput(BaseModel):
    keywords: str = Field(description="Search for nearby keywords and select target POI based on keywords")
    location: str = Field(description="Center point of nearby POI search")


def SearchNearby(keywords: str, location: str) -> str:
    ''' Search for surrounding/nearby target POI based on the center point and keywords '''
    print("keywords : ", keywords, "\nlocation : ", location)
    result = search_nearby(keywords, location)
    return result



class GetDirectionsInput(BaseModel):
    origin: str = Field(description="The first string parameter represents the origin")
    destination: str = Field(description="The second string parameter represents the destination")


def GetDirections(origin: str, destination: str) -> str:
    ''' Get directions from a given origin to a destination using a map service. '''
    print("origin : ", origin, "\ndestination : ", destination)
    result = route_planning(origin, destination)
    return result

class MyAgent:
    def __init__(self, model_name = "gpt-3.5-turbo", temperature = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name = model_name,
                              temperature = temperature)
        
        self.system_prompt = "You are world class technical documentation writer."
        self.operations = {
            "Chat": self.Chat,
            "Retrieval": self.Retrieval,
            "Conversation": self.Conversation,
            "Agent": self.Agent,
            "Navigation": self.Navigation
        }
        
    def Chat(self, user_message, verbose=False):
        if user_message is None:
            print("user_message is None !")
            return False
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("user", "{user_message}")
        ])
        
        chain = prompt | self.llm
        
        completion = chain.invoke({"user_message" : user_message,
                                   "system_prompt" : self.system_prompt})
        
        self.ShowUser(user_message)
        self.ShowCompletion(completion, verbose = verbose)
        return True
        
        
        
    def ShowUser(self, user_message):
        print(Color.GREEN + "\n #USER# :<<" + Color.ENDC)
        print(user_message, "\n\n")
        return True
    
    def ShowGPT(self, gpt_message):
        print(Color.GREEN + "\n #GPT# :>>" + Color.ENDC)
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

    def GetRetriever(self):
        loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
        docs = loader.load()
        
        embeddings = OpenAIEmbeddings()
        
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        return vector.as_retriever()
        
            
    def Retrieval(self, user_message):
        retriever = self.GetRetriever()
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            
            Question: {input}""")

        document_chain = create_stuff_documents_chain(self.llm, prompt)
        self.ShowUser(user_message)
        # gpt_message = document_chain.invoke({
        #        "input": user_message,
        #        "context": [Document(page_content="langsmith can let you visualize test results")]
        #     })
        # self.ShowGPT(gpt_message)
        
       
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": user_message})
        self.ShowGPT(response["answer"])
        return True

    def Conversation(self, user_message, verbose=False):
        retriever = self.GetRetriever()
        
        prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation") 
            ])
        retriever_chain = create_history_aware_retriever(self.llm, retriever, prompt)
        
        self.ShowUser(user_message)
        
        # chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
        # documents = retriever_chain.invoke({
        #         "chat_history": chat_history,
        #         "input": user_message
        #     })
        
        # self.ShowDocuments(documents, verbose)
        
        
        prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the user's questions based on the below context:\n\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ])
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
        chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
        gpt_message = retrieval_chain.invoke({
                "chat_history": chat_history,
                "input":  user_message
            })
        self.ShowGPT(gpt_message)
        
    def Agent(self, user_message, verbose = True):
        retriever = self.GetRetriever()
        retriever_tool = create_retriever_tool(
                retriever,
                "langsmith_search",
                "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
            )
        search = TavilySearchResults()
        tools = [retriever_tool, search]
        
        # Get the prompt to use - you can modify this!
        agent_prompt = hub.pull("hwchase17/openai-functions-agent")
        
        #print("agent prompt : \n", agent_prompt)
        
        agent = create_openai_functions_agent(self.llm, tools, agent_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
        
        chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
        agent_message = agent_executor.invoke({
                "chat_history": chat_history,
                "input": user_message
            })
        #print(type(agent_message))
        
        self.ShowUser(user_message)
        self.ShowGPT(agent_message)
        
    def Navigation(self, user_message, verbose = True):
        search = TavilySearchResults(max_results=1)
        
        searchnearby = StructuredTool.from_function(
            func=SearchNearby,
            name="SearchNearby",
            description="Search for surrounding/nearby target POI based on the center point and keywords",
            args_schema=SearchNearbyInput,
            return_direct=False,
            # coroutine= ... <- you can specify an async method if desired as well
        )
        
        directions = StructuredTool.from_function(
            func=GetDirections,
            name="GetDirections",
            description="Get directions from a given origin to a destination using a map service",
            args_schema=GetDirectionsInput,
            return_direct=False,
            # coroutine= ... <- you can specify an async method if desired as well
        )
        tools = [directions, search]
        
        # Get the prompt to use - you can modify this!
        #prompt = hub.pull("hwchase17/react")
        prompt = hub.pull("hwchase17/openai-tools-agent")
        #print(prompt['template'])
        
        
        

        # Choose the LLM to use
        #llm = OpenAI()
        agent =  create_openai_functions_agent(self.llm, tools, prompt)
        #agent =  create_react_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        #traceback.print_stack()
        
        chat_history = [SystemMessage(content='''
                                              You are a self driving route planning assistant 
                                              who needs to plan the most reasonable driving 
                                              route based on the starting point and destination,
                                              and can only consider self driving routes,
                                              Finally, provide detailed navigation information, 
                                              The navigation route needs to include all found locations
                                              including the raw information returned by the tool/function called.
                                              '''),
                        HumanMessage(content="我在南京南站"),
                        AIMessage(content="好的")]
        response = agent_executor.invoke({
                "chat_history": chat_history,
                "input": user_message
            })
        #print(type(agent_message))
        
        self.ShowUser(user_message)
        self.ShowGPT(response["output"])
        
        
    def ShowDocuments(self, documents, verbose=False):
         doc_str = ""
         for doc in documents:
             doc_str = doc_str + self.StrDocument(doc, verbose)
         self.ShowGPT(doc_str)
         
    def StrDocument(self, doc, verbose=False):
        text = doc.page_content
        metadata = doc.metadata
        doc_str = text
        verbose_str = ""
        if verbose:
            verbose_str = "\n\n***********************verbose******************" + \
                   "\n  title : " + metadata["source"] + \
                   "\n  source : " + metadata["title"] + \
                   "\n  description : " + metadata["description"] + \
                   "\n  language : " + metadata["language"] + \
                   "\n************************************************\n\n"
               
        return doc_str + verbose_str
         
    def Run(self, operation, message):
        func = self.operations.get(operation, None)
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

    agent = MyAgent()
    
    agent_type = "Navigation"
    message = "default"
    
    
    messages = {
        "Chat": "how can langsmith help with testing?",
        "Retrieval": "how can langsmith help with testing?",
        "Conversation" : "Tell me how",
        "Agent" : "Tell me how",    
        "Navigation" : "请帮忙导航到南京站，并寻找南京站附近的酒店和餐厅"
    }
    
    if agent_type in agent.operations:
        if message == "default":
            message = messages.get(agent_type, "")
        if message == "":
            print("user input empty !")
            return True
        agent.Run(agent_type, message)
    
    
if __name__ == "__main__":
    main()
    





