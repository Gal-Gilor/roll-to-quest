from enum import StrEnum


class DndClass(StrEnum):
    """D&D 5E character classes."""

    BARBARIAN = "Barbarian"
    BARD = "Bard"
    CLERIC = "Cleric"
    DRUID = "Druid"
    FIGHTER = "Fighter"
    MONK = "Monk"
    PALADIN = "Paladin"
    RANGER = "Ranger"
    ROGUE = "Rogue"
    SORCERER = "Sorcerer"
    WARLOCK = "Warlock"
    WIZARD = "Wizard"


class DndRace(StrEnum):
    """D&D 5E character species (SRD v5.2)."""

    DRAGONBORN = "Dragonborn"
    DWARF = "Dwarf"
    ELF = "Elf"
    GNOME = "Gnome"
    GOLIATH = "Goliath"
    HALFLING = "Halfling"
    HUMAN = "Human"
    ORC = "Orc"
    TIEFLING = "Tiefling"


class Condition(StrEnum):
    """D&D 5E conditions."""

    BLINDED = "Blinded"
    CHARMED = "Charmed"
    DEAFENED = "Deafened"
    FRIGHTENED = "Frightened"
    GRAPPLED = "Grappled"
    INCAPACITATED = "Incapacitated"
    INVISIBLE = "Invisible"
    PARALYZED = "Paralyzed"
    PETRIFIED = "Petrified"
    POISONED = "Poisoned"
    PRONE = "Prone"
    RESTRAINED = "Restrained"
    STUNNED = "Stunned"
    UNCONSCIOUS = "Unconscious"
    EXHAUSTION = "Exhaustion"


class Alignment(StrEnum):
    """D&D 5E alignments."""

    LAWFUL_GOOD = "Lawful Good"
    NEUTRAL_GOOD = "Neutral Good"
    CHAOTIC_GOOD = "Chaotic Good"
    LAWFUL_NEUTRAL = "Lawful Neutral"
    NEUTRAL = "Neutral"
    CHAOTIC_NEUTRAL = "Chaotic Neutral"
    LAWFUL_EVIL = "Lawful Evil"
    NEUTRAL_EVIL = "Neutral Evil"
    CHAOTIC_EVIL = "Chaotic Evil"


class Size(StrEnum):
    """D&D 5E creature sizes."""

    TINY = "Tiny"
    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"
    HUGE = "Huge"
    GARGANTUAN = "Gargantuan"


class SchoolOfMagic(StrEnum):
    """D&D 5E schools of magic."""

    ABJURATION = "Abjuration"
    CONJURATION = "Conjuration"
    DIVINATION = "Divination"
    ENCHANTMENT = "Enchantment"
    EVOCATION = "Evocation"
    ILLUSION = "Illusion"
    NECROMANCY = "Necromancy"
    TRANSMUTATION = "Transmutation"


class DamageType(StrEnum):
    """D&D 5E damage types."""

    ACID = "Acid"
    BLUDGEONING = "Bludgeoning"
    COLD = "Cold"
    FIRE = "Fire"
    FORCE = "Force"
    LIGHTNING = "Lightning"
    NECROTIC = "Necrotic"
    PIERCING = "Piercing"
    POISON = "Poison"
    PSYCHIC = "Psychic"
    RADIANT = "Radiant"
    SLASHING = "Slashing"
    THUNDER = "Thunder"


class AbilityScore(StrEnum):
    """D&D 5E ability scores."""

    STRENGTH = "Strength"
    DEXTERITY = "Dexterity"
    CONSTITUTION = "Constitution"
    INTELLIGENCE = "Intelligence"
    WISDOM = "Wisdom"
    CHARISMA = "Charisma"


class AbilityScoreAbbreviation(StrEnum):
    """D&D 5E ability score abbreviations."""

    STR = "STR"
    DEX = "DEX"
    CON = "CON"
    INT = "INT"
    WIS = "WIS"
    CHA = "CHA"


class ArmorType(StrEnum):
    """D&D 5E armor types."""

    LIGHT = "Light"
    MEDIUM = "Medium"
    HEAVY = "Heavy"
    SHIELD = "Shield"


class CreatureType(StrEnum):
    """D&D 5E creature types."""

    ABERRATION = "Aberration"
    BEAST = "Beast"
    CELESTIAL = "Celestial"
    CONSTRUCT = "Construct"
    DRAGON = "Dragon"
    ELEMENTAL = "Elemental"
    FEY = "Fey"
    FIEND = "Fiend"
    GIANT = "Giant"
    HUMANOID = "Humanoid"
    MONSTROSITY = "Monstrosity"
    OOZE = "Ooze"
    PLANT = "Plant"
    UNDEAD = "Undead"


class Rarity(StrEnum):
    """D&D 5E magic item rarity."""

    COMMON = "Common"
    UNCOMMON = "Uncommon"
    RARE = "Rare"
    VERY_RARE = "Very Rare"
    LEGENDARY = "Legendary"
    ARTIFACT = "Artifact"
    VARIES = "Varies"  # Catalog items with many variants (e.g. Ioun Stone, Spell Scroll)


class WeaponCategory(StrEnum):
    """D&D 5E weapon categories."""

    SIMPLE_MELEE = "Simple Melee"
    SIMPLE_RANGED = "Simple Ranged"
    MARTIAL_MELEE = "Martial Melee"
    MARTIAL_RANGED = "Martial Ranged"


class SpellAttackType(StrEnum):
    """D&D 5E spell attack types."""

    MELEE_SPELL_ATTACK = "Melee Spell Attack"
    RANGED_SPELL_ATTACK = "Ranged Spell Attack"
    SAVING_THROW = "Saving Throw"
    NONE = "None"


class ActionType(StrEnum):
    """D&D 5E action types."""

    ACTION = "Action"
    BONUS_ACTION = "Bonus Action"
    REACTION = "Reaction"
    FREE_ACTION = "Free Action"


class DndSkill(StrEnum):
    """D&D 5E skills with their associated ability scores."""

    ACROBATICS = "Acrobatics"  # DEX
    ANIMAL_HANDLING = "Animal Handling"  # WIS
    ARCANA = "Arcana"  # INT
    ATHLETICS = "Athletics"  # STR
    DECEPTION = "Deception"  # CHA
    HISTORY = "History"  # INT
    INSIGHT = "Insight"  # WIS
    INTIMIDATION = "Intimidation"  # CHA
    INVESTIGATION = "Investigation"  # INT
    MEDICINE = "Medicine"  # WIS
    NATURE = "Nature"  # INT
    PERCEPTION = "Perception"  # WIS
    PERFORMANCE = "Performance"  # CHA
    PERSUASION = "Persuasion"  # CHA
    RELIGION = "Religion"  # INT
    SLEIGHT_OF_HAND = "Sleight of Hand"  # DEX
    STEALTH = "Stealth"  # DEX
    SURVIVAL = "Survival"  # WIS


# Skill to ability score mapping
SKILL_ABILITY_MAP = {
    DndSkill.ACROBATICS: AbilityScoreAbbreviation.DEX,
    DndSkill.ANIMAL_HANDLING: AbilityScoreAbbreviation.WIS,
    DndSkill.ARCANA: AbilityScoreAbbreviation.INT,
    DndSkill.ATHLETICS: AbilityScoreAbbreviation.STR,
    DndSkill.DECEPTION: AbilityScoreAbbreviation.CHA,
    DndSkill.HISTORY: AbilityScoreAbbreviation.INT,
    DndSkill.INSIGHT: AbilityScoreAbbreviation.WIS,
    DndSkill.INTIMIDATION: AbilityScoreAbbreviation.CHA,
    DndSkill.INVESTIGATION: AbilityScoreAbbreviation.INT,
    DndSkill.MEDICINE: AbilityScoreAbbreviation.WIS,
    DndSkill.NATURE: AbilityScoreAbbreviation.INT,
    DndSkill.PERCEPTION: AbilityScoreAbbreviation.WIS,
    DndSkill.PERFORMANCE: AbilityScoreAbbreviation.CHA,
    DndSkill.PERSUASION: AbilityScoreAbbreviation.CHA,
    DndSkill.RELIGION: AbilityScoreAbbreviation.INT,
    DndSkill.SLEIGHT_OF_HAND: AbilityScoreAbbreviation.DEX,
    DndSkill.STEALTH: AbilityScoreAbbreviation.DEX,
    DndSkill.SURVIVAL: AbilityScoreAbbreviation.WIS,
}
