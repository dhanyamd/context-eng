import dspy 
import pprint
from pydantic import BaseModel, Field 
dspy.configure(lm=dspy.LM(model="gemini/gemini-1.5-flash", api_key="AIzaSyDbhpJ1u040K4hplUlGF3e2Yl3JlfuhzTM"))

class JokeIdea(BaseModel): 
    setup: str 
    contradiction: str 
    punchline: str 

class QueryToIdea(dspy.Signature):
    """You are a funny comedian and your goal is to generate a nice structure for a joke"""
    query: str = dspy.InputField()
    joke_idea: JokeIdea = dspy.OutputField() 

class IdeaToJoke(dspy.Signature):
    """You are a funny comedian who likes to tell stories before delivering a punchline 
       You are always funny and act on the input joke idea.   
    """
    joke_idea: JokeIdea = dspy.InputField()
    joke: str = dspy.OutputField(description="The full joke delivery in the comedian's voice")
    
class JokeGenerator(dspy.Module): 
    def __init__(self, n_attempts: int = 3):
        self.query_to_idea = dspy.Predict(QueryToIdea)
        self.idea_to_joke = dspy.Predict(IdeaToJoke)
        self.n_attempts = n_attempts
