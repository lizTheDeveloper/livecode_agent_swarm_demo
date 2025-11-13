"""
Documentation Writer and Reviewer System

This system reads a repository and generates structured documentation,
then iteratively improves it through multiple rounds of review and revision.
"""

import os
from pathlib import Path
from typing import TypedDict, Literal, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from IPython.display import Image, display

# MLX LM wrapper
from mlx_llm_wrapper import MLXLLM

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

# Initialize OpenTelemetry
resource = Resource.create({"service.name": "documentation-optimizer"})
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

# Add console exporter for development (can be replaced with OTLP exporter for production)
# Only add exporter if not in test mode to avoid I/O errors
import sys
if "pytest" not in sys.modules:
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    tracer_provider.add_span_processor(span_processor)

# Get tracer
tracer = trace.get_tracer(__name__)


# Graph state
class State(TypedDict):
    repo_path: str
    code_summary: str
    documentation: str
    feedback: str
    review_status: str
    revision_round: int


# Schema for structured output to use in evaluation
class DocumentationReview(BaseModel):
    status: Literal["approved", "needs_revision"] = Field(
        description="Whether the documentation is approved or needs revision.",
    )
    feedback: str = Field(
        description="Detailed feedback on what needs to be improved in the documentation.",
    )
    quality_score: int = Field(
        description="Quality score from 1-10, where 10 is excellent documentation.",
        ge=1,
        le=10,
    )


# Initialize MLX LLM
llm = MLXLLM(
    model_path="mlx-community/Qwen2-7B-Instruct-4bit",
    temperature=0.7,
    max_tokens=2048,
)

# For structured output with MLX_LM, we use JSON mode prompting
# MLX_LM supports tool calling through fine-tuning, but for inference we use JSON prompting
def get_structured_output(prompt: str, output_class: type) -> Any:
    """Get structured output from MLX model by prompting for JSON"""
    import json
    import re
    
    # Enhanced prompt for JSON output with clear schema
    json_prompt = f"""{prompt}

IMPORTANT: Respond with ONLY a valid JSON object, no markdown, no explanation, just JSON.
The JSON must have these exact fields:
- "status": either "approved" or "needs_revision" (string)
- "feedback": a detailed feedback string
- "quality_score": an integer between 1 and 10

JSON response:"""
    
    response = llm.invoke(json_prompt)
    content = response.content if hasattr(response, 'content') else str(response)
    
    # Clean up the response - remove markdown code blocks if present
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    content = content.strip()
    
    # Try multiple strategies to extract JSON
    # Strategy 1: Find JSON object with status field (more robust regex)
    json_match = re.search(r'\{[^{}]*(?:"status"[^{}]*)\}[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "status" in data and "feedback" in data and "quality_score" in data:
                return output_class(**data)
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Try to find any JSON object
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "status" in data and "feedback" in data and "quality_score" in data:
                return output_class(**data)
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Try parsing the whole response
    try:
        data = json.loads(content)
        if "status" in data and "feedback" in data and "quality_score" in data:
            return output_class(**data)
    except json.JSONDecodeError:
        pass
    
    # Last resort: create a default response
    return output_class(
        status="needs_revision",
        feedback=f"Could not parse structured output from model. Raw response: {content[:200]}",
        quality_score=5
    )

# Create reviewer function
def reviewer_invoke(prompt: str) -> DocumentationReview:
    """Invoke reviewer with structured output"""
    return get_structured_output(prompt, DocumentationReview)


def read_repository(repo_path: str) -> str:
    """
    Read repository files and create a summary of the codebase structure.
    Excludes common directories like .git, __pycache__, node_modules, etc.
    """
    with tracer.start_as_current_span("read_repository") as span:
        span.set_attribute("repo_path", repo_path)
        
        repo = Path(repo_path)
        if not repo.exists():
            span.record_exception(ValueError(f"Repository path does not exist: {repo_path}"))
            span.set_status(trace.Status(trace.StatusCode.ERROR, "Repository path does not exist"))
            raise ValueError(f"Repository path does not exist: {repo_path}")
    
    exclude_dirs = {
        ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
        ".pytest_cache", ".mypy_cache", ".idea", ".vscode", "dist", "build",
        ".next", ".nuxt", "coverage", ".coverage", "htmlcov"
    }
    
    exclude_extensions = {".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll"}
    
    code_files = []
    file_contents = []
    
    for root, dirs, files in os.walk(repo):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            file_path = Path(root) / file
            
            # Skip excluded extensions
            if file_path.suffix in exclude_extensions:
                continue
            
            # Skip hidden files
            if file_path.name.startswith('.'):
                continue
            
            try:
                # Read text files (limit size to avoid memory issues)
                if file_path.is_file() and file_path.stat().st_size < 100000:  # 100KB limit
                    relative_path = file_path.relative_to(repo)
                    code_files.append(str(relative_path))
                    
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        file_contents.append(f"=== {relative_path} ===\n{content}\n")
            except Exception:
                # Skip files that can't be read
                continue
    
        # Create a structured summary
        summary = f"Repository: {repo_path}\n"
        summary += f"Total files analyzed: {len(code_files)}\n\n"
        summary += "File structure:\n"
        for file_path in sorted(code_files)[:50]:  # Limit to first 50 files
            summary += f"  - {file_path}\n"
        if len(code_files) > 50:
            summary += f"  ... and {len(code_files) - 50} more files\n"
        
        summary += "\n\nCode content:\n"
        summary += "\n".join(file_contents[:20])  # Limit to first 20 files for context
        
        span.set_attribute("files_analyzed", len(code_files))
        span.set_attribute("summary_length", len(summary))
        span.set_status(trace.Status(trace.StatusCode.OK))
        
        return summary


def repo_reader(state: State):
    """Read the repository and create a code summary"""
    with tracer.start_as_current_span("repo_reader") as span:
        repo_path = state.get("repo_path", ".")
        span.set_attribute("repo_path", repo_path)
        
        try:
            code_summary = read_repository(repo_path)
            span.set_status(trace.Status(trace.StatusCode.OK))
            return {"code_summary": code_summary}
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def documentation_writer(state: State):
    """LLM generates structured documentation based on the codebase"""
    with tracer.start_as_current_span("documentation_writer") as span:
        code_summary = state.get("code_summary", "")
        revision_round = state.get("revision_round", 0)
        feedback = state.get("feedback", "")
        
        span.set_attribute("revision_round", revision_round)
        span.set_attribute("has_feedback", bool(feedback))
        span.set_attribute("code_summary_length", len(code_summary))
        
        if revision_round > 0 and feedback:
            prompt = f"""You are a technical documentation writer. Based on the following codebase analysis, 
write comprehensive, well-structured documentation. Take into account the feedback from the previous review:

Previous feedback:
{feedback}

Codebase analysis:
{code_summary[:10000]}  # Limit context size

Create documentation that includes:
1. Project Overview
2. Installation/Setup Instructions
3. Architecture/Structure
4. Key Components and their purposes
5. Usage Examples
6. API Reference (if applicable)
7. Contributing Guidelines

Make sure the documentation is clear, accurate, and follows best practices."""
        else:
            prompt = f"""You are a technical documentation writer. Based on the following codebase analysis, 
write comprehensive, well-structured documentation.

Codebase analysis:
{code_summary[:10000]}  # Limit context size

Create documentation that includes:
1. Project Overview
2. Installation/Setup Instructions
3. Architecture/Structure
4. Key Components and their purposes
5. Usage Examples
6. API Reference (if applicable)
7. Contributing Guidelines

Make sure the documentation is clear, accurate, and follows best practices."""
        
        try:
            msg = llm.invoke(prompt)
            documentation = msg.content
            span.set_attribute("documentation_length", len(documentation))
            span.set_status(trace.Status(trace.StatusCode.OK))
            return {
                "documentation": documentation,
                "revision_round": revision_round + 1
            }
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def documentation_reviewer(state: State):
    """LLM reviews the documentation and provides feedback"""
    with tracer.start_as_current_span("documentation_reviewer") as span:
        documentation = state.get("documentation", "")
        code_summary = state.get("code_summary", "")
        revision_round = state.get("revision_round", 0)
        
        span.set_attribute("revision_round", revision_round)
        span.set_attribute("documentation_length", len(documentation))
        span.set_attribute("code_summary_length", len(code_summary))
        
        review_prompt = f"""Review the following documentation for a codebase. Evaluate its quality, 
completeness, accuracy, and clarity.

Codebase context (first 5000 chars):
{code_summary[:5000]}

Documentation to review:
{documentation}

Evaluate based on:
1. Completeness - does it cover all important aspects?
2. Accuracy - is the information correct?
3. Clarity - is it easy to understand?
4. Structure - is it well-organized?
5. Examples - are there sufficient examples?
6. Technical depth - is it detailed enough?

Provide a quality score (1-10) and detailed feedback. Only approve if the score is 8 or higher."""
        
        try:
            review = reviewer_invoke(review_prompt)
            
            span.set_attribute("review_status", review.status)
            span.set_attribute("quality_score", review.quality_score)
            span.set_attribute("feedback_length", len(review.feedback))
            span.set_status(trace.Status(trace.StatusCode.OK))
            
            return {
                "review_status": review.status,
                "feedback": review.feedback,
            }
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


# Conditional edge function to route based on review status
def route_documentation(state: State):
    """Route based on review status and revision round"""
    with tracer.start_as_current_span("route_documentation") as span:
        review_status = state.get("review_status", "needs_revision")
        revision_round = state.get("revision_round", 0)
        max_rounds = 5  # Maximum number of revision rounds
        
        span.set_attribute("review_status", review_status)
        span.set_attribute("revision_round", revision_round)
        span.set_attribute("max_rounds", max_rounds)
        
        if review_status == "approved":
            route = "Accepted"
        elif revision_round >= max_rounds:
            route = "Max_Rounds_Reached"
        else:
            route = "Needs_Revision"
        
        span.set_attribute("route", route)
        span.set_status(trace.Status(trace.StatusCode.OK))
        return route


# Build workflow
optimizer_builder = StateGraph(State)

# Add the nodes
optimizer_builder.add_node("repo_reader", repo_reader)
optimizer_builder.add_node("documentation_writer", documentation_writer)
optimizer_builder.add_node("documentation_reviewer", documentation_reviewer)

# Add edges to connect nodes
optimizer_builder.add_edge(START, "repo_reader")
optimizer_builder.add_edge("repo_reader", "documentation_writer")
optimizer_builder.add_edge("documentation_writer", "documentation_reviewer")
optimizer_builder.add_conditional_edges(
    "documentation_reviewer",
    route_documentation,
    {
        "Accepted": END,
        "Needs_Revision": "documentation_writer",
        "Max_Rounds_Reached": END,
    },
)

# Compile the workflow
documentation_workflow = optimizer_builder.compile()


def generate_documentation(repo_path: str = ".", max_rounds: int = 5, show_graph: bool = False):
    """
    Generate documentation for a repository with iterative improvement.
    
    Args:
        repo_path: Path to the repository to document
        max_rounds: Maximum number of revision rounds (default: 5)
        show_graph: Whether to display the workflow graph (default: False)
    
    Returns:
        Final state containing the generated documentation
    """
    with tracer.start_as_current_span("generate_documentation") as span:
        span.set_attribute("repo_path", repo_path)
        span.set_attribute("max_rounds", max_rounds)
        span.set_attribute("show_graph", show_graph)
        
        if show_graph:
            try:
                display(Image(documentation_workflow.get_graph().draw_mermaid_png()))
            except Exception:
                print("Could not display graph. Install graphviz for visualization.")
        
        # Invoke the workflow
        initial_state = {
            "repo_path": repo_path,
            "code_summary": "",
            "documentation": "",
            "feedback": "",
            "review_status": "",
            "revision_round": 0,
        }
        
        try:
            final_state = documentation_workflow.invoke(initial_state)
            span.set_attribute("final_revision_round", final_state.get("revision_round", 0))
            span.set_attribute("final_status", final_state.get("review_status", ""))
            span.set_status(trace.Status(trace.StatusCode.OK))
            return final_state
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


if __name__ == "__main__":
    # Example usage
    result = generate_documentation(repo_path=".", show_graph=True)
    
    print("\n" + "="*80)
    print("FINAL DOCUMENTATION")
    print("="*80)
    print(result["documentation"])
    print("\n" + "="*80)
    print(f"Revision rounds: {result['revision_round']}")
    print(f"Final status: {result['review_status']}")
    if result.get("feedback"):
        print(f"Final feedback: {result['feedback']}")

