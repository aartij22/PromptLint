"""
Medical QA examples for testing the prompt optimization system
"""

from typing import List
from models import QAExample


def get_medical_examples() -> List[QAExample]:
    """Get medical QA examples for testing"""
    return [
        QAExample(
            question="What are the symptoms of diabetes?",
            context="Type 2 diabetes often develops gradually, and symptoms may be mild and go unnoticed for years. Common symptoms include increased thirst, frequent urination, increased hunger, fatigue, blurred vision, slow-healing cuts or wounds, and tingling or numbness in hands or feet.",
            answer="Common symptoms of diabetes include increased thirst, frequent urination, increased hunger, fatigue, blurred vision, slow-healing wounds, and tingling in hands or feet."
        ),
        QAExample(
            question="How is blood pressure measured?",
            context="Blood pressure is measured using a device called a sphygmomanometer. It records two numbers: systolic pressure (when the heart beats) and diastolic pressure (when the heart rests between beats). A normal reading is typically around 120/80 mmHg.",
            answer="Blood pressure is measured with a sphygmomanometer, recording systolic (heart beating) and diastolic (heart resting) pressures, with normal readings around 120/80 mmHg."
        ),
        QAExample(
            question="What is the function of the liver?",
            context="The liver is a vital organ that performs over 500 functions. Key functions include filtering toxins from blood, producing bile for digestion, storing vitamins and minerals, regulating blood sugar levels, producing proteins including blood clotting factors, and metabolizing medications.",
            answer="The liver performs over 500 functions including filtering toxins, producing bile, storing vitamins, regulating blood sugar, producing proteins and clotting factors, and metabolizing medications."
        ),
        QAExample(
            question="What is high blood pressure?", 
            context="Hypertension, commonly known as high blood pressure, is a chronic medical condition where blood pressure in arteries is persistently elevated. It is often called the 'silent killer' because it typically has no symptoms but can lead to serious complications like heart disease, stroke, and kidney problems if left untreated.",
            answer="Hypertension or high blood pressure is a chronic medical condition where blood pressure in arteries is persistently elevated."
        ),
        QAExample(
            question="What causes heart attacks?",
            context="Heart attacks occur when blood flow to part of the heart muscle is blocked, usually by a blood clot in a coronary artery. This blockage prevents oxygen-rich blood from reaching heart tissue, causing that part of the heart muscle to die. Risk factors include high cholesterol, high blood pressure, smoking, diabetes, and family history.",
            answer="Heart attacks occur when blood clots block coronary arteries, preventing oxygen from reaching heart muscle and causing tissue death. Risk factors include high cholesterol, blood pressure, smoking, and diabetes."
        )
    ] 
