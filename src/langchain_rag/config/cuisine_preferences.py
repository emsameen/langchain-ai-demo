"""
Cuisine and taste preferences configuration.

This module contains:
- Cuisines (organized by continent)
- Taste preferences (organized by category)
"""

# ────────────────────────────────────────────────────────────────────────────────
# CUISINE CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
# Cuisines organized by continent/region
CUISINES = {
    "Europe": [
        {"id": "italian", "name": "Italian", "description": "Pasta, pizza, risotto, and Mediterranean flavors"},
        {"id": "french", "name": "French", "description": "Rich sauces, pastries, and refined techniques"},
        {"id": "spanish", "name": "Spanish", "description": "Paella, tapas, and vibrant flavors"},
        {"id": "greek", "name": "Greek", "description": "Olive oil, feta, and fresh Mediterranean ingredients"},
        {"id": "mediterranean", "name": "Mediterranean", "description": "Olive oil, seafood, and fresh herbs"},
        {"id": "polish", "name": "Polish", "description": "Pierogis, kielbasa, and hearty comfort food"},
        {"id": "german", "name": "German", "description": "Sausages, pretzels, and hearty comfort food"},
        {"id": "british", "name": "British", "description": "Pies, roasts, fish & chips, and comforting puddings"},
        {"id": "scandinavian", "name": "Scandinavian", "description": "Seafood, berries, rye bread, and preserved foods"},
        {"id": "russian", "name": "Russian", "description": "Hearty stews, borscht, blini, and rich flavors"},
        {"id": "hungarian", "name": "Hungarian", "description": "Paprika, goulash, and hearty dishes"},
        {"id": "portuguese", "name": "Portuguese", "description": "Seafood, peri-peri, and Mediterranean influences"},
        {"id": "turkish", "name": "Turkish", "description": "Kebabs, mezze, baklava, and robust flavors"},
    ],
    "Asia": [
        {"id": "chinese", "name": "Chinese", "description": "Stir-fry, dumplings, and balanced flavors"},
        {"id": "japanese", "name": "Japanese", "description": "Sushi, ramen, and delicate flavors"},
        {"id": "thai", "name": "Thai", "description": "Curry, noodles, and sweet-spicy-sour balance"},
        {"id": "indian", "name": "Indian", "description": "Curry, naan, rich spices and aromatic dishes"},
        {"id": "korean", "name": "Korean", "description": "Kimchi, barbecue, and fermented flavors"},
        {"id": "vietnamese", "name": "Vietnamese", "description": "Fresh herbs, noodles, and balanced flavors"},
        {"id": "malaysian", "name": "Malaysian", "description": "Spice blends, satay, and multicultural influences"},
        {"id": "indonesian", "name": "Indonesian", "description": "Satay, sambal, and rich tropical flavors"},
        {"id": "filipino", "name": "Filipino", "description": "Adobo, sweet and sour, and fusion dishes"},
        {"id": "nepalese", "name": "Nepalese", "description": "Momos, dal bhat, and mountain cuisine"},
        {"id": "taiwanese", "name": "Taiwanese", "description": "Street food, bubble tea, and rich umami flavors"},
        {"id": "sri_lankan", "name": "Sri Lankan", "description": "Hoppers, curry, and coconut-based dishes"},
    ],
    "Americas": [
        {"id": "american", "name": "American", "description": "Burgers, comfort food, and hearty dishes"},
        {"id": "mexican", "name": "Mexican", "description": "Tacos, enchiladas, spicy and vibrant flavors"},
        {"id": "brazilian", "name": "Brazilian", "description": "Grilled meats, beans, and tropical influences"},
        {"id": "caribbean", "name": "Caribbean", "description": "Jerk spices, tropical fruits, and vibrant flavors"},
        {"id": "peruvian", "name": "Peruvian", "description": "Ceviche, potato dishes, and Andean flavors"},
        {"id": "argentinian", "name": "Argentinian", "description": "Grilled meats, chimichurri, and robust flavors"},
        {"id": "chilean", "name": "Chilean", "description": "Seafood, corn-based dishes, and coastal flavors"},
        {"id": "colombian", "name": "Colombian", "description": "Arepas, hearty stews, and diverse regional dishes"},
        {"id": "cuban", "name": "Cuban", "description": "Rice and beans, sandwiches, and citrus flavors"},
        {"id": "cajun", "name": "Cajun/Creole", "description": "Gumbo, jambalaya, and bold spicy flavors"},
    ],
    "Africa & Middle East": [
        {"id": "middle_eastern", "name": "Middle Eastern", "description": "Hummus, kebabs, and aromatic spices"},
        {"id": "moroccan", "name": "Moroccan", "description": "Tagines, couscous, and aromatic spices"},
        {"id": "ethiopian", "name": "Ethiopian", "description": "Injera, stews, and berbere spice"},
        {"id": "egyptian", "name": "Egyptian", "description": "Ful medames, koshari, and ancient flavors"},
        {"id": "lebanese", "name": "Lebanese", "description": "Mezze, tabbouleh, and fresh mountain cuisine"},
        {"id": "persian", "name": "Persian/Iranian", "description": "Saffron, stews, and elegant spice combinations"},
        {"id": "south_african", "name": "South African", "description": "Braai, bobotie, and multicultural fusion"},
        {"id": "nigerian", "name": "Nigerian", "description": "Jollof rice, plantains, and hearty stews"},
        {"id": "tunisian", "name": "Tunisian", "description": "Couscous, harissa, and Mediterranean influences"},
    ],
    "Oceania & Pacific": [
        {"id": "australian", "name": "Australian", "description": "Grilled meats, seafood, and multicultural influences"},
        {"id": "new_zealand", "name": "New Zealand", "description": "Lamb, seafood, and Pacific Rim cuisine"},
        {"id": "hawaiian", "name": "Hawaiian", "description": "Poke, spam musubi, and tropical flavors"},
        {"id": "pacific_islands", "name": "Pacific Islands", "description": "Coconut, taro, and fresh seafood"},
    ]
}

# ────────────────────────────────────────────────────────────────────────────────
# TASTE PREFERENCES CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
# Taste preferences organized by category
TASTES = {
    "Basic Tastes": [
        {"id": "spicy", "name": "Spicy", "description": "Foods with heat from chilies or peppers"},
        {"id": "sweet", "name": "Sweet", "description": "Dishes with sweet flavors or ingredients"},
        {"id": "savory", "name": "Savory", "description": "Rich, umami-forward dishes"},
        {"id": "tangy", "name": "Tangy/Sour", "description": "Dishes with acidic or sour notes"},
        {"id": "bitter", "name": "Bitter", "description": "Foods with bitter notes like some vegetables"},
        {"id": "umami", "name": "Umami-rich", "description": "Deep savory flavors like mushrooms or aged cheese"},
        {"id": "salty", "name": "Salty", "description": "Prominently salt-enhanced foods"},
    ],
    "Textures": [
        {"id": "creamy", "name": "Creamy", "description": "Smooth, rich, dairy-based textures"},
        {"id": "crunchy", "name": "Crunchy", "description": "Foods with crispy or crunchy textures"},
        {"id": "silky", "name": "Silky", "description": "Smooth, velvety texture"},
        {"id": "chewy", "name": "Chewy", "description": "Dishes with satisfying chewy textures"},
        {"id": "crispy", "name": "Crispy", "description": "Foods with a thin, crisp outer texture"},
        {"id": "juicy", "name": "Juicy", "description": "Foods that release juices when bitten"},
    ],
    "Flavor Profiles": [
        {"id": "herbaceous", "name": "Herbaceous", "description": "Prominent herb flavors"},
        {"id": "citrusy", "name": "Citrusy", "description": "Bright citrus notes and flavors"},
        {"id": "garlicky", "name": "Garlicky", "description": "Strong garlic presence"},
        {"id": "smoky", "name": "Smoky", "description": "Foods with smoky flavors or cooking methods"},
        {"id": "fermented", "name": "Fermented", "description": "Foods with fermented flavors like kimchi or yogurt"},
        {"id": "nutty", "name": "Nutty", "description": "Foods with nut-like flavors"},
        {"id": "caramelized", "name": "Caramelized", "description": "Foods with sweet-savory browned flavors"},
        {"id": "earthy", "name": "Earthy", "description": "Mushrooms, root vegetables, and forest flavors"},
        {"id": "aromatic", "name": "Aromatic", "description": "Foods with fragrant spices and herbs"},
        {"id": "fruity", "name": "Fruity", "description": "Dishes with prominent fruit flavors"},
    ],
    "Meal Characteristics": [
        {"id": "hearty", "name": "Hearty", "description": "Filling, substantial dishes"},
        {"id": "light", "name": "Light", "description": "Fresher, less heavy dishes"},
        {"id": "comfort_food", "name": "Comfort Food", "description": "Nostalgic, satisfying, homestyle dishes"},
        {"id": "healthy", "name": "Healthy", "description": "Nutritious, balanced meals"},
        {"id": "fusion", "name": "Fusion", "description": "Dishes combining multiple culinary traditions"},
        {"id": "gourmet", "name": "Gourmet", "description": "Refined, elegant dishes with premium ingredients"},
        {"id": "one_pot", "name": "One-Pot Meals", "description": "Complete meals cooked in a single container"},
    ],
    "Cooking Methods": [
        {"id": "grilled", "name": "Grilled", "description": "Foods cooked over direct heat"},
        {"id": "roasted", "name": "Roasted", "description": "Foods cooked with dry heat in an oven"},
        {"id": "fried", "name": "Fried", "description": "Foods cooked in hot oil"},
        {"id": "steamed", "name": "Steamed", "description": "Foods cooked with hot vapor"},
        {"id": "stir_fried", "name": "Stir-Fried", "description": "Foods cooked quickly over high heat while stirring"},
        {"id": "slow_cooked", "name": "Slow-Cooked", "description": "Foods cooked at low temperature for extended periods"},
        {"id": "raw", "name": "Raw", "description": "Foods prepared without cooking"},
    ]
}



# ────────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────
def get_all_cuisines():
    """Get a flat list of all cuisines from all categories."""
    return [cuisine for category in CUISINES.values() for cuisine in category]

def get_all_tastes():
    """Get a flat list of all taste preferences from all categories."""
    return [taste for category in TASTES.values() for taste in category]

# For backward compatibility with code expecting the old format
CUISINE_TYPES = get_all_cuisines()
TASTE_PREFERENCES = get_all_tastes()
