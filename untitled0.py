#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:09:13 2024

@author: jyb
"""
from abc import ABC
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
    cast,
)

from typing_extensions import NotRequired

from langchain_core.pydantic_v1 import BaseModel, PrivateAttr

class Serializable(BaseModel, ABC):
    """Serializable base class."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Is this class serializable?"""
        return False

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object.

        For example, if the class is `langchain.llms.openai.OpenAI`, then the
        namespace is ["langchain", "llms", "openai"]
        """
        return cls.__module__.split(".")

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example,
            {"openai_api_key": "OPENAI_API_KEY"}
        """
        return dict()

    @property
    def lc_attributes(self) -> Dict:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.
        """
        return {}

    @classmethod
    def lc_id(cls) -> List[str]:
        """A unique identifier for this class for serialization purposes.

        The unique identifier is a list of strings that describes the path
        to the object.
        """
        return [*cls.get_lc_namespace(), cls.__name__]

    class Config:
        extra = "ignore"


    _lc_kwargs = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        print("_lc_kwargs : ", self._lc_kwargs)
        print("kwargs : ", kwargs)
        self._lc_kwargs = kwargs
        
        print("\n\n_lc_kwargs : ", self._lc_kwargs)
        print("kwargs : ", kwargs)

class AgentExecutor(Serializable):
    """Agent that is using tools."""

    agent = "-1"
    """The agent to run for creating a plan and determining actions
    to take at each step of the execution loop."""
    tools: list[str]
    """The valid tools the agent can call."""
    verbose: bool = False
    """Whether to return the agent's trajectory of intermediate steps
    at the end in addition to the final output."""
    @classmethod
    def class_args(cls):
        print("agent: ", cls.agent, ", tools : ", cls.tools, ", verbose : ", cls.verbose)
    def object_args(self):
        print("agent: ", self.agent, ", tools : ", self.tools, ", verbose : ", self.verbose)
    
a = "aaa"
t = ["a", 'b', "c"]
v = True

print("class")
AgentExecutor.class_args()

agent_executor = AgentExecutor(agent=a, tools=t, verbose=v)
print("class")
AgentExecutor.class_args()
print("\nobj")
agent_executor.object_args()


