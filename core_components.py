"""
Core components for the prompt optimization system
"""

import json
from typing import List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

from models import QAExample, EvaluationMetrics


class LLMEvaluator:
    """LLM-as-judge evaluator for prompt performance"""

    def __init__(
        self,
        evaluation_llm: ChatGoogleGenerativeAI,
        generation_llm: ChatGoogleGenerativeAI,
    ):
        self.evaluation_llm = evaluation_llm
        self.generation_llm = generation_llm

    def create_evaluation_prompt(
        self, example: QAExample, generated_answer: str
    ) -> str:
        """Create evaluation prompt for LLM-as-judge"""
        return f"""
You are an expert evaluator for any given use case, such as summarization, classification, or QA. Your task is to evaluate the quality of a generated answer based on the provided inputs.

Note:
- The **question** field may be empty depending on the use case (e.g., summarization or title generation).
- Pay special attention to the **format and length** of the expected answer. If the expected answer is concise (e.g., a single word or short phrase), but the generated answer is overly verbose or stylistically mismatched, this should be penalized.
- If the generated answer includes unnecessary elaboration or deviates from the expected format, reduce the score accordingly and explain this in your feedback.

Inputs:
**Question:** {example.question}  
**Context:** {example.context}  
**Expected Answer:** {example.answer}  
**Generated Answer:** {generated_answer}

Evaluate on the following criteria (0.0–1.0 scale):

1. **Relevance**: How well does the answer address the question or align with the context?
2. **Accuracy**: How factually correct is the answer based on the context?
3. **Completeness**: Does the answer sufficiently and fully respond to the input?
4. **Coherence**: Is the answer well-structured, grammatically correct, and logically consistent?
5. **Format Match**: Does the generated answer match the expected length, structure, and style? Penalize heavily if the generated answer does not match the expected format.

Provide your evaluation in JSON format:

**Required JSON Output:**
{{
    "relevance_score": 0.0,
    "accuracy_score": 0.0,
    "completeness_score": 0.0,
    "coherence_score": 0.0,
    "format_match_score": 0.0,
    "reasoning": "Explain why you assigned these specific scores - what made the answer strong or weak in each area",
    "feedback": "Specific actionable suggestions for improving the prompt to generate better answers for this type of question"
}}
"""

    async def evaluate_answer(
        self, example: QAExample, generated_answer: str
    ) -> EvaluationMetrics:
        """Evaluate a single generated answer"""
        prompt = self.create_evaluation_prompt(example, generated_answer)

        messages = [
            SystemMessage(
                content="You are an expert evaluator. Always respond with valid JSON."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.evaluation_llm.ainvoke(messages)

        try:
            # Extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].strip()

            result = json.loads(content)

            # Validate and create EvaluationMetrics
            return EvaluationMetrics(
                relevance_score=float(result.get("relevance_score", 0.0)),
                accuracy_score=float(result.get("accuracy_score", 0.0)),
                completeness_score=float(result.get("completeness_score", 0.0)),
                coherence_score=float(result.get("coherence_score", 0.0)),
                format_match_score=float(result.get("format_match_score", 0.0)),
                overall_score=(
                    float(result.get("relevance_score", 0.0))
                    + float(result.get("accuracy_score", 0.0))
                    + float(result.get("completeness_score", 0.0))
                    + float(result.get("coherence_score", 0.0))
                    + float(result.get("format_match_score", 0.0))
                )
                / 5,
                reasoning=str(result.get("reasoning", "No reasoning provided")),
                feedback=str(result.get("feedback", "No feedback provided")),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing evaluation response: {e}")
            print(f"Raw response: {response.content}")

            # Return default low scores if parsing fails
            return EvaluationMetrics(
                relevance_score=0.0,
                accuracy_score=0.0,
                completeness_score=0.0,
                coherence_score=0.0,
                format_match_score=0.0,
                overall_score=0.0,
                reasoning=f"Evaluation failed due to parsing error: {str(e)}",
                feedback="Unable to provide feedback due to evaluation error",
            )

    async def evaluate_prompt(
        self, examples: List[QAExample], generated_answers: List[str]
    ) -> Tuple[float, str, str, List[EvaluationMetrics]]:
        """Evaluate the prompt performance across all examples"""
        evaluation_results = []

        for example, generated_answer in zip(examples, generated_answers):
            result = await self.evaluate_answer(example, generated_answer)
            evaluation_results.append(result)

        # Calculate average score
        avg_score = sum(result.overall_score for result in evaluation_results) / len(
            evaluation_results
        )

        # Create combined reasoning and feedback
        reasoning = await self._create_combined_reasoning(evaluation_results)
        combined_feedback = await self._create_combined_feedback(evaluation_results)

        return avg_score, reasoning, combined_feedback, evaluation_results

    async def _create_combined_reasoning(
        self, evaluations: List[EvaluationMetrics]
    ) -> str:
        """Create combined reasoning from individual evaluations using generation LLM"""

        # Calculate average scores for context
        avg_scores = {
            "relevance": sum(e.relevance_score for e in evaluations) / len(evaluations),
            "accuracy": sum(e.accuracy_score for e in evaluations) / len(evaluations),
            "completeness": sum(e.completeness_score for e in evaluations)
            / len(evaluations),
            "coherence": sum(e.coherence_score for e in evaluations) / len(evaluations),
            "format_match": sum(e.format_match_score for e in evaluations)
            / len(evaluations),
            "overall": sum(e.overall_score for e in evaluations) / len(evaluations),
        }

        # Collect individual reasonings
        individual_reasonings = [
            f"Example {i+1}: {eval.reasoning}" for i, eval in enumerate(evaluations)
        ]

        prompt = f"""
Based on the evaluation of {len(evaluations)} QA examples, provide a comprehensive reasoning for the overall performance.

**Average Scores:**
- Relevance: {avg_scores['relevance']:.3f}
- Accuracy: {avg_scores['accuracy']:.3f}  
- Completeness: {avg_scores['completeness']:.3f}
- Coherence: {avg_scores['coherence']:.3f}
- Format Match: {avg_scores['format_match']:.3f}
- Overall: {avg_scores['overall']:.3f}

**Individual Example Reasonings:**
{chr(10).join(individual_reasonings)}

Please provide a comprehensive reasoning that explains the overall performance across all examples. Focus on patterns, strengths, and areas for improvement. Be specific about what the prompt does well and where it falls short.
"""

        messages = [
            SystemMessage(
                content="You are an expert evaluator providing comprehensive analysis."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.generation_llm.ainvoke(messages)
        return response.content.strip()

    async def _create_combined_feedback(
        self, evaluations: List[EvaluationMetrics]
    ) -> str:
        """Create combined actionable feedback from individual evaluations"""

        individual_feedback = [
            f"Example {i+1}: {eval.feedback}" for i, eval in enumerate(evaluations)
        ]

        prompt = f"""
Based on the feedback from evaluating {len(evaluations)} QA examples, provide consolidated, actionable recommendations for improving the prompt.

**Individual Feedback:**
{chr(10).join(individual_feedback)}

Please synthesize this feedback into clear, actionable recommendations for prompt improvement. Focus on:
1. Specific changes to make to the prompt structure or instructions
2. Areas where the prompt needs more guidance or constraints
3. Ways to improve consistency across different types of questions

Provide practical suggestions that can be directly implemented in the prompt.
"""

        messages = [
            SystemMessage(
                content="You are an expert prompt engineer providing actionable recommendations."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.generation_llm.ainvoke(messages)
        return response.content.strip()


class PromptGenerator:
    """Generator for creating and improving prompts"""

    def __init__(self, generation_llm: ChatGoogleGenerativeAI):
        self.generation_llm = generation_llm

    async def generate_initial_prompt(
        self, requirement: str, examples: List[QAExample]
    ) -> str:
        """Generate the initial prompt based on requirements and examples"""

        # Analyze examples to understand the task
        example_analysis = self._analyze_examples(examples)

        prompt = f"""
Create a high-quality prompt for the below requirement:

**Requirement:** {requirement}

**Sample Examples:**
{example_analysis}

**Guidelines:**
1. Create a clear, specific prompt for question-answering
2. Use the placeholder {{context}} in all prompts, and {{question}} only if the task involves a user query.
3. Include instructions for handling cases where context doesn't contain the answer
4. Emphasize accuracy, relevance, and completeness
5. Make it robust and generalizable
6. If your response contains JSON references, use double curly braces ({{ and }}).

**Output the prompt directly (no explanations):**
"""

        messages = [
            SystemMessage(
                content="You are an expert prompt engineer. Create clear, effective prompts."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.generation_llm.ainvoke(messages)
        return response.content.strip()

    async def improve_prompt(
        self, current_prompt: str, current_score: float, feedback: str, requirement: str
    ) -> str:
        """Improve the current prompt based on feedback"""

        prompt = f"""
Improve the below prompt based on performance feedback.

**Original Requirement:** {requirement}

**Current Prompt:**
{current_prompt}

**Performance Analysis:**
- Current Score: {current_score:.2f}

**Specific Improvement Recommendations:**
{feedback}

**Improvement Instructions:**
1. Address each recommendation in the feedback above
2. Make the prompt more specific and clear based on the feedback
3. Add constraints to prevent the identified issues
4. Improve structure and instructions as suggested
5. Keep placeholders {{question}} and {{context}}
6. Note - If your response contains JSON references apart from the placeholders, use double curly braces ({{ and }}).

**Output the improved prompt directly (no explanations):**
"""

        messages = [
            SystemMessage(
                content="You are an expert prompt engineer focused on iterative improvement."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.generation_llm.ainvoke(messages)
        return response.content.strip()

    def _analyze_examples(self, examples: List[QAExample]) -> str:
        """Analyze the provided examples to understand the task better"""
        analysis = []

        # Basic statistics
        analysis.append(f"Number of examples: {len(examples)}")

        # Question types
        empty_questions = sum(1 for ex in examples if not ex.question.strip())
        if empty_questions > 0:
            analysis.append(
                f"- {empty_questions} examples are for summarization (empty questions)"
            )

        analysis.append(
            f"- {len(examples) - empty_questions} examples have specific questions"
        )

        # Context and answer lengths (rough)
        avg_context_len = sum(len(ex.context.split()) for ex in examples) / len(
            examples
        )
        avg_answer_len = sum(len(ex.answer.split()) for ex in examples) / len(examples)

        analysis.append(f"- Average context length: ~{avg_context_len:.0f} words")
        analysis.append(f"- Average answer length: ~{avg_answer_len:.0f} words")

        return "\n".join(analysis)


class AnswerGenerator:
    """Generator for creating answers using the current prompt"""

    def __init__(self, answer_generation_llm):
        self.answer_generation_llm = answer_generation_llm

    async def generate_answers(
        self, prompt: str, examples: List[QAExample]
    ) -> List[str]:
        """Generate answers for all examples using the current prompt"""
        answers = []

        for example in examples:
            # Format the prompt with the specific question and context
            formatted_prompt = prompt.format(
                question=example.question, context=example.context
            )

            messages = [
                SystemMessage(
                    content="You are a helpful AI assistant. Follow the prompt instructions carefully."
                ),
                HumanMessage(content=formatted_prompt),
            ]

            response = await self.answer_generation_llm.ainvoke(messages)
            answers.append(response.content.strip())

        return answers
