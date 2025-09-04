import json
import os
import pathlib
from rich.prompt import IntPrompt, Prompt
from rich.markup import escape
from envs import DialogueEnv
from utils import num_tokens

from matrix import Cli
from matrix.client import query_llm

MAX_PROMPT_LENGTH = 8192 # changed from original 4096 to avoid excessive OOC errors
MAX_RESPONSE_LENGTH = 512 # changed from original 128 due to model verbosity

class OutOfContextError(Exception):
    pass

class DryRunPlayer:

    def __init__(self, prompt, role, console, task="planning"):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.calls = 0
        self.task = task

    def observe(self, obs):
        self.prompt += obs

    def respond(self):
        self.calls += 1
        if self.role == "agent" and self.calls == 5:
            if self.task == "planning":
                return f" [propose] [Saul's, Cookies Cream, Mad Seoul]"
            elif self.task == "mediation":
                return f" [propose] User 0: [1], User 1: [15]"
        elif self.role == "user" and self.calls == 6:
            return f" [reject]"
        return f" [message] {self.calls}"

class LLMPlayer:

    def __init__(self, prompt, role, console, model='gpt-4o', model_kwargs=None,
                 prefix="\nYou:", optional=None):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.model = model
        self.optional = optional
        self.removed_optional = False
        if self.role in ["user", "agent", "user0", "user1"]:
            stop_tokens = ["User", "Agent", "You", "\n"]
        elif self.role in ["player-1", "player-2"]:
            stop_tokens = ["Partner", "You", "\n"]
        else:
            raise NotImplementedError
        self.model_kwargs = dict(
            model=self.model,
            temperature=0.1,
            top_p=.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_tokens,
        )
        if model_kwargs is not None:
            self.model_kwargs.update(**model_kwargs)
        self.prefix = prefix
        self.metadata = Cli().get_app_metadata(app_name=self.model)

    def observe(self, obs):
        self.prompt += obs

    def respond(self):
        self.console.rule(f"{self.role}'s turn")
        if not self.prompt.endswith(self.prefix):
            self.prompt += self.prefix
        #self.console.print(escape(self.prompt))
        remaining = MAX_PROMPT_LENGTH - num_tokens(self.prompt)
        if remaining < 0 and self.optional:
            self._remove_optional_context()
            remaining = MAX_PROMPT_LENGTH - num_tokens(self.prompt)
        # Still out of context after removing
        if remaining < 0:
            print("OUT OF CONTEXT! Remaining ", remaining)
            raise OutOfContextError()
            
        response = query_llm.batch_requests(
            url=self.metadata['endpoints']['head'],
            model=self.metadata['model_name'],
            app_name=self.metadata['name'],
            requests=[{"messages": [{'role': 'user', 'content': self.prompt}], **self.model_kwargs}],
            max_tokens=min(remaining, MAX_RESPONSE_LENGTH)
        )[0]['response']
        
        self.console.print("Response: ",
                           escape(response["text"][0].strip()))
        self.console.print("stop: ", response["finish_reason"][0])
        if response["finish_reason"] == "length":
            if not self.optional:
                raise OutOfContextError()
            self._remove_optional_context()
            response = query_llm.batch_requests(
                url=self.metadata['endpoints']['head'],
                model=self.metadata['model_name'],
                app_name=self.metadata['name'],
                requests=[{"messages": [{'role': 'user', 'content': self.prompt}], **self.model_kwargs}],
                max_tokens=min(remaining, MAX_RESPONSE_LENGTH)
            )
            self.console.print("Response: ",
                               escape(response["text"][0].strip()))
            self.console.print("stop: ", response["finish_reason"][0])
        self.console.print(response["usage"])
        return response["text"][0].strip()

    def _remove_optional_context(self):
        print("Cutting out optional context from prompt.")
        if self.removed_optional:
            print("!! already removed.")
            return
        self.prompt = (
            self.prompt[:self.prompt.index(self.optional)] +
            self.prompt[self.prompt.index(self.optional) + len(self.optional):])
        self.removed_optional = True

class HumanPlayer:

    def __init__(self, prompt, role, console, prefix="\nYou:"):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.prefix = prefix

    def observe(self, obs):
        self.prompt += obs

    def respond(self):
        if not self.prompt.endswith(self.prefix):
            self.prompt += self.prefix
        self.console.rule(f"Your turn ({self.role})")
        self.console.print(escape(self.prompt))
        resp = ""
        if self.prefix.strip().endswith("You to"):
            id_ = Prompt.ask(
                escape(f"Choose a player to talk to"),
                choices=["0","1","all"])
            resp += f" {id_}:"
        mtypes = ["[message]", "[propose]", "[accept]", "[reject]"]
        choices = " ".join(
                [f"({i}): {type_}" for i, type_ in enumerate(mtypes)])
        type_ = IntPrompt.ask(
                escape(
                    f"Choose one of the following message types:"
                    f"\n{choices}"),
                choices=["0","1","2","3"])
        message_type = mtypes[type_]
        if message_type not in ("[accept]", "[reject]"):
            content = Prompt.ask(escape(f"{message_type}"))
        else:
            content = ""
        resp += f" {message_type} {content}"
        return resp
