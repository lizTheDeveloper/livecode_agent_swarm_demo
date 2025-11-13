# Livecode Agent Swarm Demo

A multi-agent system built with LangGraph that demonstrates agent swarms with tool calling capabilities. This project includes several agent examples:

## Projects

### Current Events Comedian
An agent swarm that generates jokes about current events using MCP (Model Context Protocol) tools to fetch real-time news stories.

**Features:**
- Multi-agent workflow with LangGraph
- MCP tool integration for fetching current events
- Iterative joke generation with feedback loop
- Tool calling with async support

**Usage:**
```bash
python current_events_comedian.py
```

**Requirements:**
- Set `OPENAI_API_KEY` environment variable
- Install dependencies: `pip install -r requirements.txt`

### Documentation Optimizer
An automated documentation generation system that reads a repository, generates structured documentation, and iteratively improves it through multiple rounds of AI-powered review and revision.

**Usage:**
```bash
python documentation_optimizer.py
```

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd livecode_agent_swarm_demo
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Requirements

See `requirements.txt` for full list of dependencies. Key packages:
- langchain
- langchain-openai
- langgraph
- langchain-mcp-adapters
- fastmcp
- beautifulsoup4
- requests

## Project Structure

```
.
├── current_events_comedian.py  # Main comedian agent swarm
├── thebluereport.py            # MCP server for fetching news
├── documentation_optimizer.py  # Documentation generation agent
├── evaluator_optimizer.py      # Simple evaluator example
├── mlx_llm_wrapper.py          # MLX LLM wrapper (if using local models)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## How It Works

### Current Events Comedian

1. **MCP Client Setup**: Connects to MCP servers (e.g., The Blue Report) to get tools
2. **Tool Binding**: Binds MCP tools to the LLM
3. **Agent Swarm**: 
   - Generator agent creates jokes using current events
   - Tools are called automatically when needed
   - Evaluator agent reviews jokes and provides feedback
   - Loop continues until a good joke is found

### Architecture

The system uses LangGraph's `StateGraph` with:
- **MessagesState**: Extended state that includes message history for tool calling
- **ToolNode**: Automatically executes tool calls from the LLM
- **Conditional Edges**: Routes between agents based on tool calls and evaluation results

## Notes

- All API keys should be set via environment variables, never hardcoded
- The MCP server paths are automatically resolved relative to the script location
- Async/await is used throughout for proper tool execution

## License

MIT
