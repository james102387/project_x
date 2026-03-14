"""
Remulak — Synthetic Knowledge Graph Dataset

80 triplets describing a fictional alien planet. Designed so that:
1. No LLM can answer from training data (proves KG value)
2. Entities are interconnected (tests multi-hop potential)
3. Synonyms and alternate phrasings exist naturally (tests routing)

Triplet format: (subject, predicate, object)
Predicate aliases map common rephrasings to canonical predicates.
"""

TRIPLETS: list[tuple[str, str, str]] = [
    # ===================================================================
    # GEOGRAPHY — Planet
    # ===================================================================
    ("Remulak", "is a", "planet"),
    ("Remulak", "star system", "Veldra-7"),
    ("Remulak", "number of moons", "3"),
    ("Remulak", "moons", "Thyss, Orin, Pellux"),
    ("Remulak", "number of continents", "4"),
    ("Remulak", "continents", "Draveth, Sulari, Khotane, Yelvri"),
    ("Remulak", "largest continent", "Draveth"),
    ("Remulak", "smallest continent", "Yelvri"),
    ("Remulak", "capital", "Zelphos"),
    ("Remulak", "diameter", "14,200 km"),
    ("Remulak", "orbital period", "391 local days"),
    ("Remulak", "atmosphere", "nitrogen-argon with trace methane"),

    # ===================================================================
    # GEOGRAPHY — Continents & Cities
    # ===================================================================
    ("Draveth", "capital", "Zelphos"),
    ("Draveth", "population", "1.8 billion"),
    ("Draveth", "known for", "government and military academies"),
    ("Draveth", "major landmark", "The Spire of Vohn"),
    ("Draveth", "climate", "temperate with long dry seasons"),

    ("Sulari", "capital", "Mevrath"),
    ("Sulari", "population", "1.4 billion"),
    ("Sulari", "known for", "mining and heavy industry"),
    ("Sulari", "major landmark", "The Crucible Mines"),
    ("Sulari", "climate", "arid desert highlands"),

    ("Khotane", "capital", "Tessavri"),
    ("Khotane", "population", "900 million"),
    ("Khotane", "known for", "agriculture and bioengineering"),
    ("Khotane", "major landmark", "The Hanging Forests of Druen"),
    ("Khotane", "climate", "tropical monsoon"),

    ("Yelvri", "capital", "Pelkath"),
    ("Yelvri", "population", "200 million"),
    ("Yelvri", "known for", "scientific research and observatories"),
    ("Yelvri", "major landmark", "The Deep Array"),
    ("Yelvri", "climate", "polar tundra"),

    # ===================================================================
    # GOVERNMENT
    # ===================================================================
    ("Remulak", "government type", "technocratic council"),
    ("Remulak", "governing body", "The Quorum of Twelve"),
    ("Remulak", "leader", "Grand Vizier Korth"),
    ("Grand Vizier Korth", "title", "Grand Vizier"),
    ("Grand Vizier Korth", "real name", "Korth Vellan"),
    ("Grand Vizier Korth", "age", "142 standard years"),
    ("Grand Vizier Korth", "birthplace", "Zelphos"),
    ("Grand Vizier Korth", "predecessor", "Vizier Aamra Sel"),
    ("The Quorum of Twelve", "members", "12 elected technarchs"),
    ("The Quorum of Twelve", "term length", "25 standard years"),
    ("The Quorum of Twelve", "meeting place", "The Hall of Resonance in Zelphos"),

    # ===================================================================
    # DEMOGRAPHICS
    # ===================================================================
    ("Remulak", "population", "4.3 billion"),
    ("Remulak", "dominant species", "Remulaki"),
    ("Remulaki", "average lifespan", "210 standard years"),
    ("Remulaki", "distinguishing feature", "bioluminescent skin markings"),
    ("Remulaki", "number of languages", "3"),
    ("Remulaki", "languages", "Veldrasi, Khotani, Old Sulric"),
    ("Remulak", "official language", "Veldrasi"),
    ("Veldrasi", "writing system", "logographic with 4,200 base glyphs"),

    # ===================================================================
    # ECONOMY
    # ===================================================================
    ("Remulak", "currency", "the Vreth"),
    ("Remulak", "major exports", "resonance crystals, bioengineered grain, dark-ore"),
    ("Remulak", "major imports", "quantum processors, off-world genetics"),
    ("Remulak", "primary trade partner", "the Confederacy of Talshek"),
    ("resonance crystals", "found in", "Sulari"),
    ("resonance crystals", "used for", "energy storage and long-range communication"),
    ("dark-ore", "found in", "The Crucible Mines of Sulari"),
    ("dark-ore", "used for", "starship hull reinforcement"),
    ("the Vreth", "exchange rate", "1 Vreth = 340 Talshek credits"),

    # ===================================================================
    # MILITARY & TECHNOLOGY
    # ===================================================================
    ("Remulak", "military branch", "The Veldran Guard"),
    ("The Veldran Guard", "commander", "Marshal Draya Kess"),
    ("The Veldran Guard", "size", "2.1 million active personnel"),
    ("The Veldran Guard", "headquarters", "Fort Thalenn in Draveth"),
    ("Remulak", "technology level", "post-fusion, pre-singularity"),
    ("Remulak", "faster than light travel", "resonance-fold drives"),
    ("resonance-fold drives", "invented by", "Physicist Orath Yenn"),
    ("resonance-fold drives", "range", "approximately 400 light-years per fold"),

    # ===================================================================
    # CULTURE & HISTORY
    # ===================================================================
    ("Remulak", "founding year", "Year Zero of the Resonance Calendar"),
    ("Remulak", "major holiday", "The Festival of Vohn"),
    ("The Festival of Vohn", "celebrated on", "the summer solstice of Draveth"),
    ("The Festival of Vohn", "commemorates", "the first successful resonance-fold"),
    ("Remulak", "most popular sport", "skyracing"),
    ("skyracing", "played with", "single-pilot grav-sleds"),
    ("skyracing", "championship", "The Pellux Cup"),
    ("The Pellux Cup", "named after", "Remulak's smallest moon Pellux"),
    ("Remulak", "great historical conflict", "The Sulari Fracture War"),
    ("The Sulari Fracture War", "duration", "12 standard years"),
    ("The Sulari Fracture War", "cause", "dispute over Crucible Mine ownership"),
    ("The Sulari Fracture War", "ended by", "the Treaty of Mevrath"),
]


# ===================================================================
# Predicate aliases — maps alternate phrasings to canonical predicates
# used in the triplets above. Keys are lowercased at lookup time.
# ===================================================================
PREDICATE_ALIASES: dict[str, str] = {
    # geography
    "capital city": "capital",
    "main city": "capital",
    "star": "star system",
    "solar system": "star system",
    "how many moons": "number of moons",
    "moon count": "number of moons",
    "how many continents": "number of continents",
    "continent count": "number of continents",
    "biggest continent": "largest continent",
    "size": "diameter",
    "how big": "diameter",
    "year length": "orbital period",
    "orbit": "orbital period",
    "air": "atmosphere",
    "weather": "climate",

    # government
    "government": "government type",
    "type of government": "government type",
    "political system": "government type",
    "ruling body": "governing body",
    "head of state": "leader",
    "ruler": "leader",
    "who leads": "leader",
    "who rules": "leader",
    "real name": "real name",
    "birth name": "real name",
    "actual name": "real name",
    "born in": "birthplace",
    "where born": "birthplace",
    "previous leader": "predecessor",
    "how long is a term": "term length",
    "where do they meet": "meeting place",

    # demographics
    "how many people": "population",
    "population count": "population",
    "species": "dominant species",
    "race": "dominant species",
    "life expectancy": "average lifespan",
    "how long do they live": "average lifespan",
    "physical trait": "distinguishing feature",
    "appearance": "distinguishing feature",
    "how many languages": "number of languages",
    "language count": "number of languages",
    "main language": "official language",
    "primary language": "official language",
    "script": "writing system",
    "alphabet": "writing system",

    # economy
    "money": "currency",
    "what currency": "currency",
    "exports": "major exports",
    "what do they export": "major exports",
    "imports": "major imports",
    "what do they import": "major imports",
    "trade partner": "primary trade partner",
    "trading partner": "primary trade partner",
    "located in": "found in",
    "where found": "found in",
    "purpose": "used for",
    "what is it used for": "used for",
    "conversion rate": "exchange rate",

    # military & technology
    "military": "military branch",
    "army": "military branch",
    "armed forces": "military branch",
    "commanded by": "commander",
    "who commands": "commander",
    "how many soldiers": "size",
    "troop count": "size",
    "base": "headquarters",
    "hq": "headquarters",
    "tech level": "technology level",
    "ftl": "faster than light travel",
    "warp drive": "faster than light travel",
    "ftl travel": "faster than light travel",
    "who invented": "invented by",
    "creator": "invented by",
    "how far": "range",

    # culture & history
    "founded": "founding year",
    "when founded": "founding year",
    "holiday": "major holiday",
    "celebration": "major holiday",
    "when celebrated": "celebrated on",
    "what does it celebrate": "commemorates",
    "sport": "most popular sport",
    "popular sport": "most popular sport",
    "equipment": "played with",
    "trophy": "championship",
    "named for": "named after",
    "major war": "great historical conflict",
    "biggest war": "great historical conflict",
    "how long did it last": "duration",
    "length": "duration",
    "what caused": "cause",
    "reason": "cause",
    "how did it end": "ended by",
    "resolution": "ended by",
}
