"""
PromptLint - Professional Prompt Optimization System
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import json
import os
import asyncio
from datetime import datetime
from typing import List

# Import our modular components
from models import QAExample, UserInput, OptimizationConfig
from workflow import PromptOptimizationWorkflow
from examples.examples_normal import get_normal_examples
from examples.examples_medical import get_medical_examples
from examples.examples_summarization import get_summarization_examples
from helpers import export_results_to_json
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PromptLint", layout="wide", initial_sidebar_state="expanded"
)

# Professional styling for PromptLint
st.markdown(
    """
<style>
/* Hide Streamlit branding and reduce padding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Remove all default Streamlit spacing */
.css-18e3th9, .css-1d391kg, .css-1lcbmhc, .css-1outpf7 {
    padding-top: 0.5rem !important;
    margin-top: 0rem !important;
}

/* Target the main app container */
section.main > .block-container {
    padding-top: 1rem !important;
    margin-top: 0rem !important;
}

.main > div {
    padding-top: 1rem !important;
}

/* Remove any default Streamlit padding */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 0rem !important;
}

/* Professional header styling */
.promptlint-header {
    background: linear-gradient(135deg, #8c3f45 0%, #1f1f1f 100%);
    color: white;
    padding: 0.2rem 1rem;
    margin: -0.5rem -0.2rem 0.8rem -0.4rem;
}

.promptlint-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 600;
}

.promptlint-header p {
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
    font-size: 1rem;
}

/* Progress container with fixed height */
.progress-container {
    background-color: #e9ecef;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 1rem;
    height: 320px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    white-space: pre-wrap;
    color: #495057;
}

.progress-running {
    background-color: #fff3cd;
    border-color: #ffeaa7;
}

.progress-complete {
    background-color: #d4edda;
    border-color: #c3e6cb;
}

.progress-error {
    background-color: #f8d7da;
    border-color: #f5c6cb;
}

/* Compact form styling */
.stTextArea textarea {
    min-height: 100px !important;
}

/* Results styling */
.results-container {
    background-color: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #ced4da;
}

/* Fix button styling */
.stButton > button {
    width: 100%;
    border-radius: 6px;
    border: none;
    background: linear-gradient(135deg, #8c3f45 0%, #1f1f1f 100%);
    color: white;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* Metrics styling */
.metric-container {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e9ecef;
    text-align: center;
    margin: 0.5rem 0;
}

/* Status indicators */
.status-ready { color: #28a745; font-weight: 500; }
.status-warning { color: #ffc107; font-weight: 500; }
.status-error { color: #dc3545; font-weight: 500; }
</style>
""",
    unsafe_allow_html=True,
)


class PromptOptimizerStreamlitApp:
    """Main Streamlit application class"""

    def __init__(self):
        self.workflow = PromptOptimizationWorkflow()

    def parse_json_examples(
        self, file_content: str
    ) -> tuple[bool, str, List[QAExample]]:
        """Parse uploaded JSON content into QAExample objects"""
        try:
            data = json.loads(file_content)

            # Validate that it's a list
            if not isinstance(data, list):
                return False, "JSON must be a list of examples", []

            if len(data) < 5 or len(data) > 10:
                return False, f"Must have 5-10 examples, got {len(data)}", []

            examples = []
            for i, item in enumerate(data):
                try:
                    # Validate required fields
                    if not all(
                        key in item for key in ["question", "context", "answer"]
                    ):
                        return (
                            False,
                            f"Example {i+1} missing required fields (question, context, answer)",
                            [],
                        )

                    example = QAExample(
                        question=item.get("question", ""),
                        context=item["context"],
                        answer=item["answer"],
                        metadata=item.get("metadata", {}),
                    )
                    examples.append(example)
                except Exception as e:
                    return False, f"Error parsing example {i+1}: {str(e)}", []

            return True, f"Successfully loaded {len(examples)} examples", examples

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}", []
        except Exception as e:
            return False, f"Error reading file: {str(e)}", []

    async def run_optimization_with_progress(
        self, user_input: UserInput, progress_callback
    ):
        """Run optimization with clean, focused progress updates"""

        # Import workflow components directly for manual stepping
        from workflow import (
            initialize_node,
            generate_initial_prompt_node,
            generate_answers_node,
            evaluate_prompt_node,
            check_completion_node,
            improve_prompt_node,
            finalize_results_node,
        )

        # Initial state setup
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
            "individual_metric_failure": None,
            "generation_llm": None,
            "answer_generation_llm": None,
            "evaluation_llm": None,
        }

        current_state = initial_state.copy()

        # Step 1: Initialize
        progress_callback("Initializing optimization system...")
        current_state = await initialize_node(current_state)
        if current_state.get("error_message"):
            raise Exception(current_state["error_message"])
        progress_callback("System initialized successfully")

        # Step 2: Generate initial prompt
        progress_callback("Generating initial prompt...")
        current_state = await generate_initial_prompt_node(current_state)
        if current_state.get("error_message"):
            raise Exception(current_state["error_message"])
        progress_callback("Initial prompt created")

        # Main optimization loop
        iteration = 1
        while (
            not current_state["optimization_complete"]
            and iteration <= user_input.config.max_iterations
        ):

            # Step 3: Generate answers
            progress_callback(f"Iteration {iteration}: Generating answers...")
            current_state = await generate_answers_node(current_state)
            if current_state.get("error_message"):
                raise Exception(current_state["error_message"])
            progress_callback(
                f"Generated {len(current_state['generated_answers'])} answers"
            )

            # Step 4: Evaluate answers
            progress_callback(f"Iteration {iteration}: Evaluating answers...")
            current_state = await evaluate_prompt_node(current_state)
            if current_state.get("error_message"):
                raise Exception(current_state["error_message"])

            score = current_state["current_score"]
            target = user_input.config.target_score
            progress_callback(f"Score: {score:.3f} (target: {target:.3f})")

            # Step 5: Check completion
            progress_callback("Checking completion criteria...")
            current_state = await check_completion_node(current_state)

            # Display individual metric failure information if any
            if current_state.get("individual_metric_failure"):
                progress_callback(f"{current_state['individual_metric_failure']}")
            else:
                # Check if we have evaluation results to show individual metric status
                if current_state["prompt_versions"]:
                    latest_version = current_state["prompt_versions"][-1]
                    if latest_version.evaluation_results:
                        min_threshold = current_state[
                            "user_input"
                        ].config.min_individual_score
                        progress_callback(
                            f"All individual metrics above {min_threshold}"
                        )

            if current_state["optimization_complete"]:
                if score >= target:
                    progress_callback("Target score achieved!")
                elif iteration >= user_input.config.max_iterations:
                    progress_callback("Maximum iterations reached")
                break

            if iteration < user_input.config.max_iterations:
                # Step 6: Improve prompt
                progress_callback(f"Iteration {iteration}: Improving prompt...")
                current_state = await improve_prompt_node(current_state)
                if current_state.get("error_message"):
                    raise Exception(current_state["error_message"])
                progress_callback("Prompt enhanced for next iteration")
                iteration += 1
            else:
                break

        # Step 7: Finalize
        progress_callback("Finalizing results...")
        current_state = await finalize_results_node(current_state)

        # Create final result
        if current_state["prompt_versions"]:
            best_version = max(
                current_state["prompt_versions"], key=lambda v: v.average_score or 0.0
            )

            from models import OptimizationResult

            result = OptimizationResult(
                success=True,
                final_prompt=best_version.prompt_text,
                final_score=best_version.average_score or 0.0,
                reasoning=current_state["current_reasoning"],
                iterations_completed=current_state["current_iteration"] + 1,
                all_versions=current_state["prompt_versions"],
            )

            progress_callback(
                f"Optimization complete! Final score: {result.final_score:.3f}"
            )
            return result
        else:
            raise Exception("No prompt versions were created")


def main():
    """Main PromptLint application"""

    # Professional header
    st.markdown(
        """
    <div class="promptlint-header">
        <h1>PromptLint</h1>
        <h5>AI-powered prompt optimization</h5>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "app_instance" not in st.session_state:
        st.session_state.app_instance = PromptOptimizerStreamlitApp()
    if "progress_lines" not in st.session_state:
        st.session_state.progress_lines = []
    if "optimization_complete" not in st.session_state:
        st.session_state.optimization_complete = False
    if "optimization_result" not in st.session_state:
        st.session_state.optimization_result = None

    # Sidebar for configuration and resources
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Advanced settings
        with st.expander("Optimization Settings", expanded=True):
            max_iterations = st.slider("Max Iterations", 1, 5, 3)
            target_score = st.slider("Target Score", 0.5, 1.0, 0.8, 0.05)
            min_individual_score = st.slider(
                "Min Individual Score",
                0.0,
                1.0,
                0.5,
                0.05,
                help="Minimum score required for each individual metric (relevance, accuracy, completeness, coherence, format match)",
            )

        # Examples and templates
        with st.expander("Example Templates", expanded=False):
            st.markdown("**Sample JSON Templates:**")

            if st.button("📚 General Knowledge", use_container_width=True):
                sample_data = [
                    {
                        "question": "What is the capital of France?",
                        "context": "France is a country in Western Europe. Its capital and largest city is Paris...",
                        "answer": "Paris",
                    },
                    {
                        "question": "How many continents are there?",
                        "context": "The Earth is divided into seven continents...",
                        "answer": "Seven continents",
                    },
                ]
                st.json(sample_data)

            if st.button("🏥 Medical QA", use_container_width=True):
                sample_data = [
                    {
                        "question": "What are symptoms of diabetes?",
                        "context": "Type 2 diabetes often develops gradually with symptoms like increased thirst...",
                        "answer": "Common symptoms include increased thirst, frequent urination, fatigue...",
                    }
                ]
                st.json(sample_data)

            if st.button("📰 Summarization", use_container_width=True):
                sample_data = [
                    {
                        "question": "",
                        "context": "India's lunar mission Chandrayaan-3 successfully landed near the Moon's south pole, making it the first country to achieve this feat...",
                        "answer": '{"title": "India\'s Chandrayaan-3 Lands on Moon\'s South Pole"}',
                    }
                ]
                st.json(sample_data)

        # Usage tips
        with st.expander("💡 Usage Tips", expanded=False):
            st.markdown(
                """
            **For best results:**
            
            • Be specific in requirements
            
            • Use 5-10 diverse examples
            
            • Include edge cases if relevant
            
            • Set appropriate target scores
            
            **Typical process:**
            
            • Upload takes ~30 seconds
            
            • Optimization runs 2-5 minutes
            
            • Higher scores need more iterations
            """
            )

        # API status
        st.markdown("### 🔗 System Status")
        api_key = os.getenv("GEMINI_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            st.markdown(
                '<p class="status-ready">✅ API Connected</p>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<p class="status-error">❌ API Key Missing</p>', unsafe_allow_html=True
            )

    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Input Configuration")

        # Requirement input
        requirement = st.text_area(
            "Prompt Requirement",
            placeholder="Describe your QA assistant requirements...\n\nExample: Create a professional medical QA assistant that provides accurate answers while reminding users to consult healthcare professionals.",
            height=100,
            help="Be specific about task, tone, and format requirements",
        )

        # File upload option - Fixed layout
        st.markdown("**Choose Examples Source:**")
        upload_option = st.radio(
            "Examples source:",
            ["Upload JSON file", "Use sample examples"],
            horizontal=True,
            label_visibility="collapsed",
        )

        # Fixed container for upload/sample selection
        upload_container = st.container()
        with upload_container:
            uploaded_file = None
            examples = None

            if upload_option == "Upload JSON file":
                uploaded_file = st.file_uploader(
                    "Upload QA Examples (JSON)",
                    type=["json"],
                    help="5-10 examples with question, context, answer fields",
                )
            else:
                sample_type = st.selectbox(
                    "Select sample type:",
                    ["General Knowledge", "Medical QA", "Summarization"],
                )

                if sample_type == "General Knowledge":
                    examples = get_normal_examples()
                    st.success(f"✅ {len(examples)} examples loaded")
                elif sample_type == "Medical QA":
                    examples = get_medical_examples()
                    st.success(f"✅ {len(examples)} examples loaded")
                else:
                    examples = get_summarization_examples()
                    st.success(f"✅ {len(examples)} examples loaded")

        # Status check
        can_optimize = bool(
            requirement and (uploaded_file is not None or examples is not None)
        )

        if not can_optimize:
            if not requirement:
                st.markdown(
                    '<p class="status-warning">⚠️ Requirement needed</p>',
                    unsafe_allow_html=True,
                )
            elif upload_option == "Upload JSON file" and uploaded_file is None:
                st.markdown(
                    '<p class="status-warning">⚠️ JSON file needed</p>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<p class="status-ready">✅ Ready to optimize</p>',
                unsafe_allow_html=True,
            )

        # Optimization button
        optimize_button = st.button(
            "Start Optimization", type="primary", disabled=not can_optimize
        )

    with col2:
        st.markdown("### Progress & Results")

        # Progress and results area
        progress_placeholder = st.empty()

        if not st.session_state.optimization_complete:
            # Show progress container
            with progress_placeholder.container():
                progress_container_class = "progress-container"
                if len(st.session_state.progress_lines) > 0:
                    progress_container_class += " progress-running"

                progress_text = "\n".join(st.session_state.progress_lines)
                if not progress_text:
                    progress_text = "Ready to start optimization..."

                st.markdown(
                    f'<div class="{progress_container_class}">{progress_text}</div>',
                    unsafe_allow_html=True,
                )

        else:
            # Show collapsed progress and results
            with st.expander("📝 View Progress Log", expanded=False):
                progress_text = "\n".join(st.session_state.progress_lines)
                st.code(progress_text, language=None)

            # Show results
            results_container = st.container()
            if st.session_state.optimization_result:
                result = st.session_state.optimization_result
                with results_container:
                    if result.success:
                        st.success("✅ Optimization Complete!")

                        col_metric1, col_metric2 = st.columns(2)
                        with col_metric1:
                            st.metric("Final Score", f"{result.final_score:.3f}")
                        with col_metric2:
                            st.metric("Iterations", f"{result.iterations_completed}")

                        # Final prompt
                        st.markdown("**Optimized Prompt:**")
                        st.text_area(
                            "Copy your optimized prompt:",
                            value=result.final_prompt,
                            height=150,
                            label_visibility="collapsed",
                        )

                        # Additional details in expander
                        with st.expander("📊 Detailed Results", expanded=False):
                            st.markdown("**Evaluation Reasoning:**")
                            st.markdown(result.reasoning)

                            if len(result.all_versions) > 1:
                                st.markdown("**Version History:**")
                                for version in result.all_versions:
                                    st.write(
                                        f"• Version {version.version}: {version.average_score:.3f}"
                                    )

                        # Export and download
                        col_download, col_reset = st.columns([2, 1])
                        with col_download:
                            filename = export_results_to_json(result)
                            if filename:
                                with open(filename, "rb") as f:
                                    st.download_button(
                                        "📥 Download Results (JSON)",
                                        data=f.read(),
                                        file_name=filename,
                                        mime="application/json",
                                        use_container_width=True,
                                    )

                        with col_reset:
                            if st.button(
                                "🔄 New Optimization", use_container_width=True
                            ):
                                st.session_state.optimization_complete = False
                                st.session_state.optimization_result = None
                                st.session_state.progress_lines = []
                                st.rerun()
                    else:
                        st.error("❌ Optimization Failed")
                        if result.error_message:
                            st.error(f"Error: {result.error_message}")

    # Handle optimization
    if optimize_button and can_optimize:

        # Reset optimization state
        st.session_state.optimization_complete = False
        st.session_state.optimization_result = None
        st.session_state.progress_lines = []
        st.session_state.progress_placeholder = progress_placeholder

        # Validate API key
        api_key = os.getenv("GEMINI_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            error_msg = (
                "[ERROR] API Key Missing - Please set GEMINI_KEY environment variable"
            )
            st.session_state.progress_lines.append(error_msg)
            with progress_placeholder.container():
                st.markdown(
                    f'<div class="progress-container progress-error">{error_msg}</div>',
                    unsafe_allow_html=True,
                )
            return

        # Parse examples if uploaded
        if uploaded_file is not None:
            file_content = uploaded_file.read().decode("utf-8")
            success, message, parsed_examples = (
                st.session_state.app_instance.parse_json_examples(file_content)
            )

            if not success:
                error_msg = f"[ERROR] File Error: {message}"
                st.session_state.progress_lines.append(error_msg)
                with progress_placeholder.container():
                    st.markdown(
                        f'<div class="progress-container progress-error">{error_msg}</div>',
                        unsafe_allow_html=True,
                    )
                return

            examples = parsed_examples

        # Create configuration
        config = OptimizationConfig(
            max_iterations=max_iterations,
            target_score=target_score,
            min_individual_score=min_individual_score,
            generation_model_name="gemini-2.0-flash",
            answer_generation_model_name="granite3.3",
            evaluation_model_name="gemini-2.5-flash",
            generation_temperature=0.7,
            answer_generation_temperature=0.7,
            evaluation_temperature=0.1,
        )

        user_input = UserInput(
            requirement=requirement.strip(), examples=examples, config=config
        )

        def clean_progress_callback(message):
            """Clean, focused progress callback with live updates"""
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_line = f"[{timestamp}] {message}"
            st.session_state.progress_lines.append(formatted_line)

            # Update progress display in real-time
            if hasattr(st.session_state, "progress_placeholder"):
                with st.session_state.progress_placeholder.container():
                    progress_text = "\n".join(st.session_state.progress_lines)
                    st.markdown(
                        f'<div class="progress-container progress-running">{progress_text}</div>',
                        unsafe_allow_html=True,
                    )

        try:
            # Run optimization
            async def run_optimization():
                return (
                    await st.session_state.app_instance.run_optimization_with_progress(
                        user_input, clean_progress_callback
                    )
                )

            # Execute optimization
            result = asyncio.run(run_optimization())

            # Set completion state
            st.session_state.optimization_complete = True
            st.session_state.optimization_result = result

            # Force final UI update
            st.rerun()

        except Exception as e:
            error_msg = f"[ERROR] Critical Error: {str(e)}"
            st.session_state.progress_lines.append(error_msg)
            st.session_state.optimization_complete = True
            with progress_placeholder.container():
                st.markdown(
                    f'<div class="progress-container progress-error">{error_msg}</div>',
                    unsafe_allow_html=True,
                )
            st.rerun()


if __name__ == "__main__":
    main()
