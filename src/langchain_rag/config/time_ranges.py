"""
Available cooking time configuration.

This module contains the list of time ranges that users can select from.
"""

# List of cooking time ranges
COOKING_TIME_RANGES = [
    {"id": "quick", "name": "15 minutes or less", "description": "Very quick recipes for when you're in a hurry"},
    {"id": "short", "name": "15-30 minutes", "description": "Short cooking time, great for weekday meals"},
    {"id": "medium", "name": "30-60 minutes", "description": "Medium cooking time, good for more complex dishes"},
    {"id": "long", "name": "Up to 2 hours", "description": "Longer cooking time for more elaborate recipes"},
    {"id": "extended", "name": "Up to 5 hours", "description": "Extended cooking time for slow-cooked dishes"},
]
