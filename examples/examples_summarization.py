"""
Summarization QA examples for testing the prompt optimization system
"""

from typing import List
from models import QAExample


def get_summarization_examples() -> List[QAExample]:
    """Get summarization QA examples for testing"""
    return [
        QAExample(
            question="",
            context="India's lunar mission Chandrayaan-3 successfully landed near the Moon's south pole, making it the first country to achieve this feat. The mission aims to study the Moon's surface and gather data on its mineral composition and seismic activity.",
            answer='{"title": "India\'s Chandrayaan-3 Lands on Moon\'s South Pole"}'
        ),
        QAExample(
            question="",
            context="The European Central Bank raised interest rates by 0.25%, marking the ninth consecutive hike to combat persistent inflation in the Eurozone. Analysts believe this may be the final increase in the current cycle.",
            answer='{"title": "ECB Implements Ninth Consecutive Rate Hike"}'
        ),
        QAExample(
            question="",
            context="A major wildfire in California's Yosemite National Park has spread to over 10,000 acres, forcing evacuations and threatening ancient sequoia trees. Firefighters are working around the clock to contain the blaze amid challenging weather conditions.",
            answer='{"title": "Wildfire Threatens Sequoias in Yosemite National Park"}'
        ),
        QAExample(
            question="",
            context="Apple has announced the launch of its Vision Pro mixed-reality headset, blending augmented and virtual reality features. The device, priced at $3,499, is aimed at developers and early adopters and will be available next year.",
            answer='{"title": "Apple Unveils Vision Pro Mixed-Reality Headset"}'
        ),
        QAExample(
            question="",
            context="Scientists have detected methane emissions on Saturn's moon Enceladus, strengthening the hypothesis that microbial life could exist in its subsurface ocean. The findings come from data collected by the Cassini spacecraft.",
            answer='{"title": "Methane Found on Enceladus Sparks Life Possibility"}'
        ),
        QAExample(
            question="",
            context="Tesla has reported record quarterly deliveries of 466,140 vehicles, surpassing Wall Street expectations despite ongoing supply chain challenges. The electric vehicle manufacturer continues to expand its global production capacity.",
            answer='{"title": "Tesla Reports Record Quarterly Vehicle Deliveries"}'
        )
    ] 