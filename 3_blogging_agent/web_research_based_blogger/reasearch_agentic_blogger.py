from __future__ import annotations
import operator
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from utilities import sanitize_filename
from langchain_tavily import TavilySearch


class Task(BaseModel):
    """
    Defines data for individual tasks that are to be carried out under a plan.
    """
    id: int
    titlee: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section."
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3-6 concrete, non-overlapping subpoints to cover this section."
    )
    target_words: int = Field(
        ...,
        "Target word counts before this section (120-250)"
    )

    tags = List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False
    
class Plan(BaseModel):
    """
    Defines data for the blog plan of a certain topic.
    The LLM is asked to create a plan on a certain topic in this defined format.
    Note the sub-model "Task" here.
    """
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal['explainer', 'tutorial', 'news_roundmap', 'comparison', 'sytem_design'] = 'explainer'
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]

class EvidenceItem(BaseModel):
    """
    Model to be used by the search engine. The response given by the search engine will be in this format.
    """
    title: str
    url: str
    published_at: Optional[str]  # only in case travely is providing this info
    snippet: Optional[str]
    source: Optional[str]

class RouterDecision(BaseModel):
    """
    Model for containing all the information whether a topic is to be researched or not.
    """
    needs_research: bool
    mode: Literal['closed_book', 'hybrid', 'open_book']
    queries: List[str] = Field(default_factory=list)

class EvidencePack:
    evidence: List[EvidenceItem] = Field(default_factory=list)


# agent state
class State(TypedDict):
    topic: str

    # routing/research
    mode: str
    needs_research: str
    queries: str
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, str)
    final: str


llm = ChatOpenAI(model='gpt-4o-mini')


ROUTER_SYSTEM_PROMPT = """
You are a routing module for a technical blog planner.
Decide whether web research is needed before planning.

Modes:
    - closed book (needs_research=false)
        - Evergreen topics where correctness does not depend on recent topics.
    - hybrid (needs_research=true)
        - Mostly evergreen topics, but need up-to-date examples/tools to be useful.
    - open book (needs_research=false)
        - Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy_regulation etc

If needs_research=true:
    - Output 3-10 high-signal queries.
    - Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM")
    - If user asked for "last week/this week/latest", reflect those constraints IN THE QUERIES.
"""

# node for deciding the researchabilty related info - using llm call
def router_node(state: State) -> dict:
    """
    Purpose : Based on the topic provided in the state, calls the llm to decide if topic needs research.
              The output given by llm will be as per the Pydantic Model: RouterDecision
    """
    topic = state['topic']
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f'Topic: {topic}')
    )
    return {
        'needs_research': decision.needs_research,
        'mode': decision.mode,
        'queries': decision.queries
    }


# conditional node - for directing the flow.
def route_next(state: State) -> str:
    return 'research' if state['needs_research'] else 'orchestrator'


# research node : internet search - using tavily
def _tavily_search(query: str, max_results: int=5) -> List[dict]:
    tools = TavilySearch(max_results=max_results)
    results = tools.invoke({'query': query})
    normalized: List[dict] = []
    for r in results or []:
        normalized.append({
            'title': r.get('title') or '',
            'url': r.get('url') or '',
            'snippet': r.get('content') or r.get('snippet') or '',
            'published_at': r.get('published_date') or r.get('published_at'),
            'source': r.get('source')
        })
    return normalized
    
RESEARCH_SYSTEM_PROMPT = """
    You are a research synthesizer for a technical writing.
    Given raw web search results, produce a deduplicated list of EvidenceItem objects.

    Rules:
        - Only include items with a non-empty url
        - Prefer relevant + authoritative sources (company blogs, docs, reputable outlets)
        - If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD
            If missing or unclear, set published_at=null. Do NOT guess.
        - Keep snippets short.
        - Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    """
    Takes the state and makes the internet search for the query.
    At this point, it's already decided that the research is to be done.
    """
    # take the first 10 queries from state
    queries = (state.get('queries') or [])
    max_results = 6
    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))
    if not raw_results:
        return {'evidence': []}
    
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=f'raw_results:\n{raw_results}')
        ]
    )

    # Deduplicate by URL
    dedup  = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {'evidence': list(dedup.values())}
