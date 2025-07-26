"""
LangGraph workflow for prompt optimization
"""

import os
from typing import Any
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from models import UserInput, OptimizationResult, PromptVersion
from core_components import LLMEvaluator, PromptGenerator, AnswerGenerator


class WorkflowState(TypedDict):
    """State maintained throughout the optimization workflow"""

    user_input: UserInput
    current_iteration: int
    current_prompt: str
    current_score: float
    current_reasoning: str
    current_feedback: str
    generated_answers: list[str]
    prompt_versions: list[PromptVersion]
    optimization_complete: bool
    error_message: str | None
    individual_metric_failure: str | None  # Message about failing individual metrics
    generation_llm: Any  # ChatGoogleGenerativeAI for prompt generation
    answer_generation_llm: Any  # ChatOllama for answer generation
    evaluation_llm: Any  # ChatGoogleGenerativeAI for evaluation


# Workflow Nodes
async def initialize_node(state: WorkflowState) -> WorkflowState:
    """Initialize the optimization workflow"""
    print("🚀 Initializing workflow...")

    try:
        # Get API key for Gemini (required for prompt generation and evaluation)
        api_key = os.getenv("GEMINI_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            state["error_message"] = (
                "API key not found. Please set GEMINI_KEY or GOOGLE_API_KEY."
            )
            state["optimization_complete"] = True
            return state

        # Initialize LLMs with different configurations
        # Prompt generation LLM (Gemini)
        generation_llm = ChatGoogleGenerativeAI(
            model=state["user_input"].config.generation_model_name,
            api_key=api_key,
            temperature=state["user_input"].config.generation_temperature,
        )

        # Answer generation LLM (Ollama)
        answer_generation_llm = ChatOllama(
            model=state["user_input"].config.answer_generation_model_name,
            temperature=state["user_input"].config.answer_generation_temperature,
            base_url="http://localhost:11434",  # Default Ollama URL
        )

        # Evaluation LLM (Gemini)
        evaluation_llm = ChatGoogleGenerativeAI(
            model=state["user_input"].config.evaluation_model_name,
            api_key=api_key,
            temperature=state["user_input"].config.evaluation_temperature,
        )

        state["generation_llm"] = generation_llm
        state["answer_generation_llm"] = answer_generation_llm
        state["evaluation_llm"] = evaluation_llm
        state["current_feedback"] = ""
        state["generated_answers"] = []
        state["individual_metric_failure"] = None

        print(f"✅ Initialized with {len(state['user_input'].examples)} examples")

    except Exception as e:
        state["error_message"] = f"Failed to initialize: {str(e)}"
        state["optimization_complete"] = True

    return state


async def generate_initial_prompt_node(state: WorkflowState) -> WorkflowState:
    """Generate the initial prompt"""
    print("📝 Generating initial prompt...")

    try:
        generator = PromptGenerator(state["generation_llm"])

        initial_prompt = await generator.generate_initial_prompt(
            state["user_input"].requirement, state["user_input"].examples
        )

        state["current_prompt"] = initial_prompt
        print("✅ Initial prompt generated successfully")

    except Exception as e:
        state["error_message"] = f"Failed to generate initial prompt: {str(e)}"
        state["optimization_complete"] = True

    return state


async def generate_answers_node(state: WorkflowState) -> WorkflowState:
    """Generate answers for all examples using the current prompt"""
    print(f"🔄 Generating answers with current prompt...")

    try:
        answer_generator = AnswerGenerator(state["answer_generation_llm"])

        generated_answers = await answer_generator.generate_answers(
            state["current_prompt"], state["user_input"].examples
        )

        state["generated_answers"] = generated_answers
        print(f"✅ Generated {len(generated_answers)} answers")

    except Exception as e:
        state["error_message"] = f"Failed to generate answers: {str(e)}"
        state["optimization_complete"] = True

    return state


async def evaluate_prompt_node(state: WorkflowState) -> WorkflowState:
    """Evaluate the generated answers using LLM-as-judge"""
    print(f"🔍 Evaluating answers (iteration {state['current_iteration'] + 1})...")

    try:
        evaluator = LLMEvaluator(state["evaluation_llm"], state["generation_llm"])

        # Evaluate the generated answers
        avg_score, reasoning, feedback, evaluation_results = (
            await evaluator.evaluate_prompt(
                state["user_input"].examples, state["generated_answers"]
            )
        )

        # Update state
        state["current_score"] = avg_score
        state["current_reasoning"] = reasoning
        state["current_feedback"] = feedback

        # Create and store prompt version
        version = PromptVersion(
            version=state["current_iteration"] + 1,
            prompt_text=state["current_prompt"],
            average_score=avg_score,
            evaluation_results=evaluation_results,
            improvements_made=[f"Version {state['current_iteration'] + 1} created"],
        )
        state["prompt_versions"].append(version)

        print(f"✅ Evaluation complete. Score: {avg_score:.2f}")

    except Exception as e:
        state["error_message"] = f"Failed to evaluate prompt: {str(e)}"
        state["optimization_complete"] = True

    return state


async def check_completion_node(state: WorkflowState) -> WorkflowState:
    """Update state with completion decision"""
    print("🔍 Checking completion criteria...")

    # Check for errors
    if state.get("error_message"):
        print("❌ Stopping due to error")
        state["optimization_complete"] = True
        return state

    # Check individual metric thresholds
    if state["prompt_versions"]:
        latest_version = state["prompt_versions"][-1]
        if latest_version.evaluation_results:
            # Calculate average scores across all examples for each metric
            avg_relevance = sum(
                result.relevance_score for result in latest_version.evaluation_results
            ) / len(latest_version.evaluation_results)
            avg_accuracy = sum(
                result.accuracy_score for result in latest_version.evaluation_results
            ) / len(latest_version.evaluation_results)
            avg_completeness = sum(
                result.completeness_score
                for result in latest_version.evaluation_results
            ) / len(latest_version.evaluation_results)
            avg_coherence = sum(
                result.coherence_score for result in latest_version.evaluation_results
            ) / len(latest_version.evaluation_results)
            avg_format_match = sum(
                result.format_match_score
                for result in latest_version.evaluation_results
            ) / len(latest_version.evaluation_results)

            min_threshold = state["user_input"].config.min_individual_score
            failing_metrics = []

            if avg_relevance < min_threshold:
                failing_metrics.append(f"Relevance: {avg_relevance:.3f}")
            if avg_accuracy < min_threshold:
                failing_metrics.append(f"Accuracy: {avg_accuracy:.3f}")
            if avg_completeness < min_threshold:
                failing_metrics.append(f"Completeness: {avg_completeness:.3f}")
            if avg_coherence < min_threshold:
                failing_metrics.append(f"Coherence: {avg_coherence:.3f}")
            if avg_format_match < min_threshold:
                failing_metrics.append(f"Format Match: {avg_format_match:.3f}")

            if failing_metrics:
                failure_message = f"Individual metrics below {min_threshold}: {', '.join(failing_metrics)}"
                print(f"❌ {failure_message}")
                state["individual_metric_failure"] = failure_message
                state["optimization_complete"] = False
                return state
            else:
                state["individual_metric_failure"] = None

    # Check if target score reached
    if state["current_score"] >= state["user_input"].config.target_score:
        print(f"🎯 Target score {state['user_input'].config.target_score} reached!")
        state["optimization_complete"] = True
        return state

    # Check if max iterations reached
    if state["current_iteration"] >= state["user_input"].config.max_iterations - 1:
        print(
            f"⏰ Max iterations ({state['user_input'].config.max_iterations}) reached"
        )
        state["optimization_complete"] = True
        return state

    # Continue optimization
    print(
        f"🔄 Score {state['current_score']:.2f} < {state['user_input'].config.target_score}, continuing..."
    )
    state["optimization_complete"] = False
    return state


def route_completion(state: WorkflowState) -> str:
    """Route based on completion status"""
    if state["optimization_complete"]:
        return "finalize"
    else:
        return "improve"


async def improve_prompt_node(state: WorkflowState) -> WorkflowState:
    """Improve the current prompt based on evaluation feedback"""
    print(f"⚡ Improving prompt (iteration {state['current_iteration'] + 1})...")

    try:
        generator = PromptGenerator(state["generation_llm"])

        improved_prompt = await generator.improve_prompt(
            state["current_prompt"],
            state["current_score"],
            state["current_feedback"],
            state["user_input"].requirement,
        )

        state["current_prompt"] = improved_prompt
        state["current_iteration"] += 1

        print("✅ Prompt improved successfully")

    except Exception as e:
        state["error_message"] = f"Failed to improve prompt: {str(e)}"
        state["optimization_complete"] = True

    return state


async def finalize_results_node(state: WorkflowState) -> WorkflowState:
    """Finalize optimization results"""
    print("🏁 Finalizing optimization results...")

    state["optimization_complete"] = True

    # Find best version
    if state["prompt_versions"]:
        best_version = max(
            state["prompt_versions"], key=lambda v: v.average_score or 0.0
        )
        print(f"✅ Optimization complete! Best score: {best_version.average_score:.2f}")
    else:
        print("⚠️ No prompt versions created")

    return state


class PromptOptimizationWorkflow:
    """LangGraph workflow for prompt optimization"""

    def __init__(self):
        self.workflow = None
        self.memory = MemorySaver()

    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""

        # Create the workflow graph
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("initialize", initialize_node)
        workflow.add_node("generate_initial_prompt", generate_initial_prompt_node)
        workflow.add_node("generate_answers", generate_answers_node)
        workflow.add_node("evaluate_prompt", evaluate_prompt_node)
        workflow.add_node("check_completion", check_completion_node)
        workflow.add_node("improve_prompt", improve_prompt_node)
        workflow.add_node("finalize_results", finalize_results_node)

        # Add edges
        workflow.add_edge("initialize", "generate_initial_prompt")
        workflow.add_edge("generate_initial_prompt", "generate_answers")
        workflow.add_edge("generate_answers", "evaluate_prompt")
        workflow.add_edge("evaluate_prompt", "check_completion")

        # Conditional edges from check_completion
        workflow.add_conditional_edges(
            "check_completion",
            route_completion,
            {"improve": "improve_prompt", "finalize": "finalize_results"},
        )

        # Continue the optimization loop (improve -> generate answers -> evaluate)
        workflow.add_edge("improve_prompt", "generate_answers")
        workflow.add_edge("finalize_results", END)

        # Set entry point
        workflow.set_entry_point("initialize")

        return workflow

    async def run_optimization(self, user_input: UserInput) -> OptimizationResult:
        """Run the complete optimization workflow"""

        # Build workflow if not already built
        if self.workflow is None:
            self.workflow = self.build_workflow()

        # Compile the workflow
        app = self.workflow.compile(checkpointer=self.memory)

        # Initial state
        initial_state = {
            "user_input": user_input,
            "current_iteration": 0,
            "current_prompt": "",
            "current_score": 0.0,
            "current_reasoning": "",
            "current_feedback": "",
            "generated_answers": [],
            "prompt_versions": [],
            "optimization_complete": False,
            "error_message": None,
            "generation_llm": None,
            "evaluation_llm": None,
        }

        # Run the workflow
        try:
            result = await app.ainvoke(initial_state, {"thread_id": "optimization"})
            print("prompt_versions", result["prompt_versions"])

            # Find best prompt
            if result["prompt_versions"]:
                best_version = max(
                    result["prompt_versions"], key=lambda v: v.average_score or 0.0
                )

                return OptimizationResult(
                    success=True,
                    final_prompt=best_version.prompt_text,
                    final_score=best_version.average_score or 0.0,
                    reasoning=result["current_reasoning"],
                    iterations_completed=result["current_iteration"] + 1,
                    all_versions=result["prompt_versions"],
                )
            else:
                return OptimizationResult(
                    success=False,
                    final_prompt="",
                    final_score=0.0,
                    reasoning="No prompt versions were created",
                    iterations_completed=0,
                    all_versions=[],
                    error_message=result.get("error_message", "Unknown error"),
                )

        except Exception as e:
            return OptimizationResult(
                success=False,
                final_prompt="",
                final_score=0.0,
                reasoning="Workflow execution failed",
                iterations_completed=0,
                all_versions=[],
                error_message=str(e),
            )
