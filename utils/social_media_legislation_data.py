# utils/social_media_legislation_data.py
# Source: UNICEF, 'Drawing a line in digital spaces' (March 2026)
# Data reflects status as of 13 March 2026.

LEGISLATION_DATA = [
    # Implemented
    {"country": "Australia",            "status": "Implemented"},
    # Passed
    {"country": "Indonesia",            "status": "Passed"},
    {"country": "Malaysia",             "status": "Passed"},
    # Proposed
    {"country": "France",               "status": "Proposed"},
    {"country": "Norway",               "status": "Proposed"},
    {"country": "Portugal",             "status": "Proposed"},
    {"country": "Turkey",               "status": "Proposed"},
    {"country": "United States",        "status": "Proposed"},
    {"country": "Brazil",               "status": "Proposed"},
    {"country": "Costa Rica",           "status": "Proposed"},
    {"country": "Ecuador",              "status": "Proposed"},
    {"country": "Philippines",          "status": "Proposed"},
    {"country": "South Korea",          "status": "Proposed"},
    # Under discussion
    {"country": "Austria",              "status": "Under discussion"},
    {"country": "Czech Republic",       "status": "Under discussion"},
    {"country": "Denmark",              "status": "Under discussion"},
    {"country": "Finland",              "status": "Under discussion"},
    {"country": "Germany",              "status": "Under discussion"},
    {"country": "Greece",               "status": "Under discussion"},
    {"country": "Netherlands",          "status": "Under discussion"},
    {"country": "Poland",               "status": "Under discussion"},
    {"country": "Slovenia",             "status": "Under discussion"},
    {"country": "Spain",                "status": "Under discussion"},
    {"country": "Sweden",               "status": "Under discussion"},
    {"country": "Switzerland",          "status": "Under discussion"},
    {"country": "United Kingdom",       "status": "Under discussion"},
    {"country": "Canada",               "status": "Under discussion"},
    {"country": "Egypt",                "status": "Under discussion"},
    {"country": "Nigeria",              "status": "Under discussion"},
    {"country": "South Africa",         "status": "Under discussion"},
    {"country": "Zimbabwe",             "status": "Under discussion"},
    {"country": "Fiji",                 "status": "Under discussion"},
    {"country": "India",                "status": "Under discussion"},
    {"country": "New Zealand",          "status": "Under discussion"},
    {"country": "Pakistan",             "status": "Under discussion"},
]

# Status levels in logical order (most to least advanced)
STATUS_ORDER = [
    "Implemented",
    "Passed",
    "Proposed",
    "Under discussion",
    "No data",
]

# Colour scale: single blue hue, dark to light, grey for No data
STATUS_COLORS = {
    "Implemented":      "#08519c",
    "Passed":           "#3182bd",
    "Proposed":         "#9ecae1",
    "Under discussion": "#deebf7",
    "No data":          "#d9d9d9",
}
