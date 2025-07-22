"""
Meal types configuration.

This module contains definitions for different meal types.
"""

# Meal type options
MEAL_TYPES = [
    {"id": "breakfast", "name": "Breakfast", "description": "Morning meals to start the day"},
    {"id": "brunch", "name": "Brunch", "description": "Late morning meals combining breakfast and lunch"},
    {"id": "lunch", "name": "Lunch", "description": "Midday meals"},
    {"id": "dinner", "name": "Dinner", "description": "Evening meals"},
    {"id": "appetizer", "name": "Appetizer", "description": "Starters and hors d'oeuvres"},
    {"id": "snack", "name": "Snack", "description": "Small bites between meals"},
    {"id": "dessert", "name": "Dessert", "description": "Sweet treats to end a meal"},
    {"id": "drink", "name": "Drink", "description": "Beverages, cocktails, and other drinks"},
    {"id": "side_dish", "name": "Side Dish", "description": "Accompaniments to main courses"},
]

# Default meal type (None means no selection)
DEFAULT_MEAL_TYPE = None
