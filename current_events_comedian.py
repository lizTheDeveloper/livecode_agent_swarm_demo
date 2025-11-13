import os
import warnings
# Suppress transformers warning about missing PyTorch/TensorFlow/Flax
# (not needed since we're using OpenAI API, not local models)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*Flax.*")

from langchain_openai import ChatOpenAI
from typing import TypedDict
import asyncio
from pydantic import BaseModel, Field
from typing_extensions import Literal
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from langgraph.graph.message import add_messages



async def main():
    print("Initializing MCP client...")
    import sys
    import os
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(script_dir, "venv", "bin", "python")
    thebluereport_script = os.path.join(script_dir, "thebluereport.py")
    
    client = MultiServerMCPClient(
        {
            "thebluereport": {
                "command": venv_python,
                "args": [thebluereport_script],
                "transport": "stdio"
            }
            
        }
    )
    print("Getting tools from MCP servers...")
    tools = await client.get_tools()
    print(f"Retrieved {len(tools)} tools from MCP servers")

    print("Initializing LLM...")
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)
    print("LLM initialized with tools")

    # Graph state - extend MessagesState to include messages for ToolNode
    class State(MessagesState):
        joke: str
        topic: str
        feedback: str
        done_or_not: str
        previous_jokes: list[str]
        best_joke: str


    # Schema for structured output to use in evaluation
    class Feedback(BaseModel):
        grade: Literal["generate_more", "finished"] = Field(
            description="Are we done generating jokes? Are these good enough? Use the humor transaction schema.",
        )
        feedback: str = Field(
            description="If the joke is not funny, provide feedback on how to improve it, use the humor transaction schema.",
        )
        best_joke: str = Field(
            description="Of all the jokes you've been given, choose the best joke, use the humor transaction schema.",
        )


    # Augment the LLM with schema for structured output
    evaluator = llm.with_structured_output(Feedback)


    # agent 1
    async def llm_call_generator(state: State):
        """LLM generates a joke"""
        
        print("Generating jokes...")
        # Build messages list from state
        messages = list(state.get("messages", []))
        
        # Add prompt based on feedback
        if state.get("feedback"):
            prompt = f"Write 5 jokes about current events, using the tools provided, but take into account the feedback: {state['feedback']}, here are the previous jokes you've already written: {state['previous_jokes']}. Give the corresponding probabilities for each joke."
        else:
            prompt = f"Write 5 jokes about current events, using the tools provided, with their corresponding probabilities"
        
        messages.append(HumanMessage(content=prompt))
        
        # Invoke LLM with messages (async)
        msg = await llm_with_tools.ainvoke(messages)
        print(f"LLM response: {msg.content[:200] if msg.content else 'No content (tool calls)'}...")
        
        # Extract content for state update
        joke_content = msg.content if msg.content else ""
        
        return {
            "messages": [msg],
            "joke": joke_content,
            "previous_jokes": state.get("previous_jokes", []) + [joke_content] if joke_content else state.get("previous_jokes", []),
        }


    ## agent 2
    def llm_call_evaluator(state: State):
        """LLM evaluates the joke"""

        evaluation = evaluator.invoke(f"Grade the jokes: {state['joke']}\n Decide if we're done generating jokes (because we got a really good / unique one), give feedback, and choose the best joke, here are the previous jokes: {state.get("previous_jokes", "no previous jokes")}. If a joke is not 'safe for work', reject it and give feedback on how to improve it.")
        state["best_joke"] = evaluation.best_joke
        return {"done_or_not": evaluation.grade, "feedback": evaluation.feedback, "best_joke": evaluation.best_joke}


    # Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
    def route_joke(state: State):
        """Route back to joke generator or end based upon feedback from the evaluator"""

        if state["done_or_not"] == "finished":
            return "Accepted"
        elif state["done_or_not"] == "generate_more":
            return "Rejected + Feedback"


    # Build workflow
    optimizer_builder = StateGraph(State)

    # Add the nodes
    optimizer_builder.add_node("llm_call_generator", llm_call_generator)
    optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)
    optimizer_builder.add_node("tools", ToolNode(tools))

    # Add edges to connect nodes
    optimizer_builder.add_edge(START, "llm_call_generator")
    
    # Conditional edge from generator: if tool calls, go to tools, else to evaluator
    optimizer_builder.add_conditional_edges(
        "llm_call_generator",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "llm_call_evaluator"
        }
    )
    
    # After tools execute, go back to generator
    optimizer_builder.add_edge("tools", "llm_call_generator")
    optimizer_builder.add_conditional_edges(
        "llm_call_evaluator",
        route_joke,
        {  # Name returned by route_joke : Name of next node to visit
            "Accepted": END,
            "Rejected + Feedback": "llm_call_generator",
        },
    )

    # Compile the workflow
    print("Compiling workflow...")
    optimizer_workflow = optimizer_builder.compile()
    print("Workflow compiled successfully")

    

    # Invoke (async since we're using async tools)
    print("Starting joke generation workflow...")
    state = await optimizer_workflow.ainvoke({
        "messages": [],
        "topic": "Programming",
        "joke": "",
        "feedback": "",
        "done_or_not": "",
        "previous_jokes": [],
        "best_joke": ""
    })
    print("\n" + "="*50)
    print("BEST JOKE:")
    print("="*50)
    print(state["best_joke"])
    
if __name__ == "__main__":
    asyncio.run(main())