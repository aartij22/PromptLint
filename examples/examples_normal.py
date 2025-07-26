"""
Normal/General QA examples for testing the prompt optimization system
"""

from typing import List
from models import QAExample


def get_normal_examples() -> List[QAExample]:
    """Get general knowledge QA examples for testing"""
    return [
        QAExample(
            question="What is the capital of France?",
            context="France is a country in Western Europe. Its capital and largest city is Paris, located in the north-central part of the country. Paris is known for its cultural landmarks including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
            answer="Paris"
        ),
        QAExample(
            question="How many continents are there?",
            context="The Earth is divided into seven continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia/Oceania. These continents contain all the world's countries and are separated by oceans.",
            answer="Seven continents"
        ),
        QAExample(
            question="What is photosynthesis?",
            context="Photosynthesis is the process by which plants and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose. This process uses carbon dioxide from the air and water from the soil, producing oxygen as a byproduct.",
            answer="Photosynthesis is the process by which plants convert light energy into chemical energy (glucose), using carbon dioxide and water, while producing oxygen."
        ),
        QAExample(
            question="Who wrote Romeo and Juliet?",
            context="Romeo and Juliet is a tragic play written by the English playwright William Shakespeare. It was written early in his career, around 1594-1596, and tells the story of two young star-crossed lovers whose deaths ultimately unite their feuding families.",
            answer="William Shakespeare"
        ),
        QAExample(
            question="What is the largest ocean?",
            context="The Pacific Ocean is the largest and deepest ocean on Earth, covering about 46% of the world's water surface and about one-third of the total surface area of the planet. It extends from the Arctic to the Antarctic and is bounded by Asia and Australia on the west and the Americas on the east.",
            answer="The Pacific Ocean"
        ),
        QAExample(
            question="What are some usecases related to AI?",
            context="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. Modern AI applications include virtual assistants, recommendation systems, autonomous vehicles, and medical diagnosis tools.",
            answer="AI is the simulation of human intelligence in machines, enabling them to perform tasks like perception, recognition, decision-making, and translation. Applications include virtual assistants, recommendations, autonomous vehicles, and medical diagnosis."
        )
    ] 