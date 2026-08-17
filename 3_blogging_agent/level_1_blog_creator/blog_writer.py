from __future__ import annotations
import operator
from pathlib import Path
from typing import TypedDict, Annotated, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from ..utilities import sanitize_filename

load_dotenv()


class Task(BaseModel):
    id: str
    title: str
    brief: str = Field(..., description='What to cover')

class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]


# agent state definition
class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]
    final: str
llm = ChatOpenAI(model='gpt-4o-mini')


# orchestrator node
def orchestrator(state: State):
    """
    Calls the LLM to create a plan and their respective tasks for a given topic (mentioned in the state)
    Returns :
        updated state = {
            'topic': 'topic_name',
            'plan' : Plan Data prepared by the LLM in format of the mentioned Pydantic Structure.
            'sections': [],
            'final' : ''
        }
    """
    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(content="Create a blog with 5-7 sections on the following topic."),
            HumanMessage(content=f'Topic : {state['topic']}')
        ]
    )
    return {'plan': plan}


# conditional node (not just edge)
def fanout(state: State):
    """
    Takes the output of the orchestrator and prepares the data for worker node.
    At this stage: state = {
        'topic' : 'topic_name',
        'plan' : Plan Data prepared by the LLM in format of the mentioned Pydantic Structure.
        'sections': [],
        'final' : ''
    }
    """
    out_states = []
    for task in state['plan'].tasks:
        out_states.append(
            Send(
                'worker',
                {
                    'task': task,  # task = {'id_1', 'title_1', 'brief_1'}
                    'topic': state['topic'],
                    'plan': state['plan'],  # plan = {'blog_title', list_of_tasks}
                }
            )
        )
    return out_states


# worker node for invoking the llm for creating content for each task
def worker(payload: dict):
    """
    Creates data for the prompt based on the output of 'fanout' node. Then calls the llm using this prompt. 
    """
    task = payload['task']
    topic = payload['topic']
    plan = payload['plan']
    blog_title = plan.blog_title
    section_md = llm.invoke(
        [
            SystemMessage(content="Write one clean markdown section."),
            HumanMessage(
                content=(
                    f'Blog: {blog_title}\n'
                    f'Topic: {topic}\n'
                    f'Section: {task.title}\n'
                    f'Brief {task.brief}\n'
                    "Return only the section content in Markdown"
                )
            )
        ]
    ).content.strip()

    return {'sections': [section_md]}



# reducer node
def reducer(state: State):
    """
    Current State = {
        'topic' : 'topic_name',
        'plan' : Plan Data prepared by the LLM in format of the mentioned Pydantic Structure.
        'sections': [[title_1 content], [title_2 content]....[title_n content]],
        'final' : ''
    }
    Combined the sections into one content andsaves it in a persistent file.
    Returns:
        - {
            'topic' : 'topic_name',
            'plan' : Plan Data prepared by the LLM in format of the mentioned Pydantic Structure.
            'sections': [[title_1 content], [title_2 content]....[title_n content]],
            'final' : enire content on the topic
        }
    """
    title = state['plan'].blog_title
    body = '\n\n'.join(state['sections'])
    final_md = f'# {title}\n\n{body}\n'

    # Create a safe filename
    file_name = sanitize_filename(title)

    # Save next to the Python script
    output_path = Path(__file__).resolve().parent / file_name

    # Write the file safely
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(final_md)

    print('\n========== FINAL RESULT ==========\n')
    print(final_md)

    print('\n========== FILE INFO ==========')
    print(f'Filename : {output_path.name}')
    print(f'Location : {output_path}')
    print(f'Exists   : {output_path.exists()}')
    print(f'Size     : {output_path.stat().st_size} bytes')
    print('================================')

    return {'final': final_md}



# nodes
graph = StateGraph(State)
graph.add_node('orchestrator', orchestrator)
graph.add_node('worker', worker)
graph.add_node('reducer', reducer)

# edges
graph.add_edge(START, 'orchestrator')
graph.add_conditional_edges('orchestrator', fanout, ['worker'])
graph.add_edge('worker', 'reducer')
graph.add_edge('reducer', END)

app = graph.compile()

result = app.invoke({
    'topic': "Write a blog on self-attention.",
    'sections': [],
})
