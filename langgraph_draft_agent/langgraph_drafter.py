import os
from typing import Annotated, Sequence, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

load_dotenv()
document_content = ''

@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is :\n{document_content}"

@tool
def save(filename: str) -> str:
    """
    Save the current document to a single text file and finish the process.
    Args :
        - filename : name for the text file
    """
    if filename.endswith('.txt'):
        filename = f'{filename}.txt'
    
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f'\nDocument has been save to : {filename}')
        return "Document has been saved successfully."
    except Exception as e:
        return f'Error saving document : {str(e)}'


tool_list = [update, save]


model = ChatOpenAI(model='gpt-4o-mini').bind_tools(tools=tool_list)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content = f"""
        Your are drafter, a helpful writing assistant. Yor are going to hep the user update and modify document.
            - If the user wants to update or modify the content, use the 'update' tool with the complete update document.
            - If the user wants to save and finish, you need to use the 'save' tool.
            - Make sure to always show the curret document state after modification.
        The current document content is {document_content}
        """
    )
    if not state['messages']:
        user_input = input("I'm ready to help you create a document. What would you like to create ?")
    else:
        user_input = input("What would you like to do with the document ? ")
    
    user_message = HumanMessage(content=user_input)
    all_messages = [system_prompt] + state['messages'] + [user_message]
    response = model.invoke(all_messages)

    print(f'\nAI : {response.content}')
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f'USING TOOLS : {[tc['name'] for tc in response.tool_calls]}')

    return {'messages': state['messages'] + [user_message, response]}


# this is going to be the condtitional edge
def should_continue(state: AgentState) -> str:
    """Determine if the conversation is to be continued or to be ended."""
    messages = state['messages']
    if not messages:
        return 'continue'

    for message in reversed(messages):
        if isinstance(message, ToolMessage)\
            and 'saved' in message.content.lower()\
            and 'document' in message.content.lower():
            return 'end'
        
    return'continue'


def print_tool_message(messages):
    if not messages: return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f'Tool Result = {message.content}')


graph = StateGraph(AgentState)
graph.add_node('agent', our_agent)
graph.add_node('tool_node', ToolNode(tools=tool_list))

graph.add_edge(START, 'agent')
graph.add_edge('agent', 'tool_node')
graph.add_conditional_edges(
    'tool_node',
    should_continue,
    {
        'continue': 'agent',
        'end': END
    }
)
app = graph.compile()


def run_document_agent():
    print('\n==========DRAFTER==============')

    state = {'messages': []}

    for step in app.stream(state, stream_mode='values'):
        if 'messages' in step:
            print_tool_message(step['messages'])
    

    print('\n==========DRAFTER-FINISHED=============')


if __name__ == '__main__':
    run_document_agent()
