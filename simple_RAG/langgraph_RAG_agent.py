import os
from pathlib import Path
from typing import Annotated, Sequence, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# loading the document
pdf_path = Path(__file__).parent / "rag_knowledge_base_document.pdf"
if os.path.exists(pdf_path):
    raise FileNotFoundError(f'PDF file not found : {pdf_path}')
pdf_loader = PyPDFLoader(pdf_path)
try:
    pages = pdf_loader.load()
    print(f'PDF file loaded successfully. It has {len(pages)} pages')
except Exception as ex:
    print(f'Error loading PDF : {ex}')

# chunking the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
pages_split = text_splitter.split_documents(pages)

# storing the chunked documents in vector form in a chroma db collection
persistent_dir = '/workspaces/ai_mini_projects/simple_RAG/chroma_docs'
if not os.path(persistent_dir):
    os.makedirs(persistent_dir)
collection_name = 'rag_knowledge_collection'
try:
    vector_store = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persistent_directory=persistent_dir,
        collection_name=collection_name
    )
    print(f'\nCreated chroma_db vector store.')
except Exception as ex:
    print(f'\nError setting up the chorma_db : {str(ex)}')

retreiver = vector_store.as_retriever(
    search_type = 'similarity',
    search_kwargs={'k':5}  # k is the amount of chunks to return 
)



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# tool node
@tool
def retreiver_tool(query: str):
    """
    This tool searched the document and returns the information RAG knowlege.
    Args : Query: str
    """
    docs = retreiver.invoke(query)
    if not docs:
        return "I found no relevant information in RAG knowlege base"

    results = []
    for i, doc in enumerate(docs):
        results.append(f'Document {i+1} :\n{doc.content}')
    return "\n\n".join(results)

tools_list = [retreiver_tool]
llm = llm.bind_tools(tools=tools_list)


system_prompt = """
You are a helpful RAG-based assistant. Use the provided retrieval tool to find relevant information from the documents and answer the user's questions based only on the retrieved context.
Do not make up information or rely on outside knowledge when the answer is not supported by the documents.
For every answer, cite the specific source document and page/section (when available) from which the information was retrieved.
"""
# llm agent node
def call_llm(state: AgentState) -> AgentState:
    """
    Function to call the LLM with the current state.
    """
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    message = llm.invoke(messages)
    return {'messages': [message]}

# retreiver agent node
tools_dict = {our_tool.name for our_tool in tools_list}  # creating a dictionary of our tools
def take_action(state: AgentState):
    """Execute the LLM's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        if not['name'] in tools_dict:  # check if it's a valid tool
            print(f'\nTool : {t['name']} does not exist')
            result = "Incorrect Tool Name, Please Retry and select tool from list of Available Tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f'Result lenth: {len(str(result))}')

        # appends the tool message
        results.append(ToolMessage(tool_call_id=['id'], name=t['name'], content=str(result)))

    print('Tools Execution Complete. Back to the model!')
    return {'messages': results}


# conditional edge
def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    if hasattr(result, 'tool_calls') and len(result.tool_calls) > 0:
        return True
    return False


graph = StateGraph(AgentState)
graph.add_node('llm', call_llm)
graph.add_node('retreiver_agent', take_action)
graph.add_edge(START, 'llm')
llm.add_edge('llm', 'retreiver_agent')
graph.add_conditional_edges(
    'llm',
    should_continue,
    {True:'retreiver_agent', False: END}
)
rag_agent = graph.compile()


def run_agent():
    print('\n=== RUN AGENT ===')
    while True:
        user_input = input('\nWhat is your question: ')
        if user_input.lower() in ['exit', 'quit']:
            break

        messages = HumanMessage(content=user_input)
        result = rag_agent.invoke({'messages': messages})

    print('\n=== ANSWER ===')
    print(result['messages'][-1].content)

run_agent()
