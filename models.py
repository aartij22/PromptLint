"""
Pydantic models for the prompt optimization system
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QAExample(BaseModel):
    """Single QA example with question, context, and expected answer"""
    question: str = Field(..., description="The question to ask (can be empty for summarization)")
    context: str = Field(..., description="The context/document to answer from")
    answer: str = Field(..., description="The expected answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EvaluationMetrics(BaseModel):
    """Detailed evaluation metrics for a single QA example"""
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="How relevant is the answer to the question")
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="How accurate is the answer based on context")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="How complete is the answer")
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="How coherent and well-structured is the answer")
    format_match_score: float = Field(..., ge=0.0, le=1.0, description="How closely the answer matches the expected format, length, and style")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    reasoning: str = Field(..., description="Why these scores were assigned")
    feedback: str = Field(..., description="Specific suggestions for prompt improvement")


class PromptVersion(BaseModel):
    """Represents a version of the prompt with its performance metrics"""
    version: int = Field(..., description="Version number")
    prompt_text: str = Field(..., description="The actual prompt text")
    average_score: Optional[float] = Field(None, description="Average score across all examples")
    evaluation_results: List[EvaluationMetrics] = Field(default_factory=list, description="Individual evaluation results")
    improvements_made: List[str] = Field(default_factory=list, description="List of improvements made in this version")


class OptimizationConfig(BaseModel):
    """Configuration for the optimization process"""
    max_iterations: int = Field(3, ge=1, le=10, description="Maximum number of optimization iterations")
    target_score: float = Field(0.6, ge=0.0, le=1.0, description="Target score to achieve")
    min_individual_score: float = Field(0.5, ge=0.0, le=1.0, description="Minimum score required for each individual metric")
    generation_model_name: str = Field("gemini-2.0-flash", description="Model for prompt generation")
    answer_generation_model_name: str = Field("granite3.3", description="Ollama model for answer generation")
    evaluation_model_name: str = Field("gemini-2.5-flash", description="Model for evaluation (LLM-as-judge)")
    generation_temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation tasks")
    answer_generation_temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for answer generation")
    evaluation_temperature: float = Field(0.1, ge=0.0, le=2.0, description="Temperature for evaluation tasks")


class UserInput(BaseModel):
    """Complete user input for optimization"""
    requirement: str = Field(..., description="Description of what the prompt should accomplish")
    examples: List[QAExample] = Field(..., min_items=5, max_items=10, description="QA examples for evaluation")
    config: OptimizationConfig = Field(default_factory=OptimizationConfig, description="Optimization configuration")


class OptimizationResult(BaseModel):
    """Final result of the optimization process"""
    success: bool = Field(..., description="Whether optimization completed successfully")
    final_prompt: str = Field(..., description="The best prompt found")
    final_score: float = Field(..., description="Final average score achieved")
    reasoning: str = Field(..., description="Reasoning for the final evaluation")
    iterations_completed: int = Field(..., description="Number of iterations actually completed")
    all_versions: List[PromptVersion] = Field(default_factory=list, description="All prompt versions tried")
    error_message: Optional[str] = Field(None, description="Error message if optimization failed")
