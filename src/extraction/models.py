from pydantic import BaseModel
from pydantic import Field

from src.extraction.enums import AbilityScore as AbilityScoreEnum
from src.extraction.enums import AbilityScoreAbbreviation
from src.extraction.enums import ActionType
from src.extraction.enums import Alignment as AlignmentEnum
from src.extraction.enums import ArmorType
from src.extraction.enums import Condition as ConditionEnum
from src.extraction.enums import CreatureType as CreatureTypeEnum
from src.extraction.enums import DamageType as DamageTypeEnum
from src.extraction.enums import DndClass
from src.extraction.enums import DndRace
from src.extraction.enums import DndSkill
from src.extraction.enums import Rarity
from src.extraction.enums import SchoolOfMagic as SchoolOfMagicEnum
from src.extraction.enums import Size as SizeEnum
from src.extraction.enums import SpellAttackType
from src.extraction.enums import WeaponCategory


class Class(BaseModel):
    name: DndClass | None = Field(
        default=None, description="Class name from the DndClass enum"
    )
    hit_die: str | None = Field(
        default=None, description="Hit die notation (e.g., 'd8', 'd10', 'd12')"
    )
    primary_ability: AbilityScoreEnum | None = Field(
        default=None,
        description="Primary ability score for this class (e.g., Strength for Fighter)",
    )
    saving_throw_proficiencies: list[AbilityScoreAbbreviation] | None = Field(
        default=None,
        description="List of ability score abbreviations for saving throw proficiencies",
    )
    armor_proficiencies: list[str] | None = Field(
        default=None, description="Types of armor this class is proficient with"
    )
    weapon_proficiencies: list[str] | None = Field(
        default=None, description="Types of weapons this class is proficient with"
    )
    spellcasting_ability: AbilityScoreEnum | None = Field(
        default=None,
        description="Ability score used for spellcasting (if class has spellcasting)",
    )
    subclasses: list[str] | None = Field(
        default=None, description="Names of available subclasses for this class"
    )


class Subclass(BaseModel):
    name: str | None = Field(
        default=None,
        description=(
            "Subclass name (e.g., 'Life Domain', 'Champion', "
            "'Path of the Berserker')"
        ),
    )
    parent_class: DndClass | None = Field(
        default=None,
        description="Parent class for this subclass (e.g., Cleric, Fighter, Barbarian)",
    )
    description: str | None = Field(
        default=None,
        description=(
            "Complete subclass description including flavor text and all "
            "feature descriptions. Include feature names, level requirements, "
            "and mechanical effects."
        ),
    )
    level_obtained: int | None = Field(
        default=None,
        description="Level at which this subclass is selected (typically 3 for most classes)",
    )


class Race(BaseModel):
    name: DndRace | None = Field(
        default=None, description="Race/species name from the DndRace enum"
    )
    size: SizeEnum | None = Field(
        default=None, description="Creature size category (Tiny, Small, Medium, Large, etc.)"
    )
    speed: dict[str, int] | None = Field(
        default=None,
        description="Movement speeds in feet (e.g., {'walk': 30, 'fly': 60, 'swim': 30})",
    )
    ability_score_increase: dict[AbilityScoreAbbreviation, int] | None = Field(
        default=None,
        description="Ability score bonuses (e.g., {'STR': 2, 'CON': 1} for Dwarf)",
    )
    age: str | None = Field(
        default=None, description="Age description (e.g., 'reach adulthood at 20')"
    )
    lifespan: int | None = Field(default=None, description="Maximum lifespan in years")
    languages: list[str] | None = Field(
        default=None, description="Languages the race can speak"
    )
    darkvision: int | None = Field(
        default=None, description="Darkvision range in feet (typically 60)"
    )
    traits: list[str] | None = Field(
        default=None,
        description="Racial trait names (e.g., 'Dwarven Resilience', 'Stonecunning')",
    )


class Spell(BaseModel):
    name: str | None = Field(default=None, description="Spell name")
    level: int | None = Field(
        default=None, description="Spell level (0 for cantrips, 1-9 for leveled spells)"
    )
    school: SchoolOfMagicEnum | None = Field(
        default=None, description="School of magic (e.g., Evocation, Abjuration, Necromancy)"
    )
    casting_time: str | None = Field(
        default=None,
        description="Casting time (e.g., '1 action', '1 bonus action', '1 minute')",
    )
    range: str | None = Field(
        default=None,
        description="Spell range (e.g., 'Self', '30 feet', 'Touch', '1 mile')",
    )
    components: list[str] | None = Field(
        default=None,
        description="Spell components list (e.g., ['V', 'S'], ['V', 'S', 'M'])",
    )
    duration: str | None = Field(
        default=None,
        description=(
            "Spell duration (e.g., 'Instantaneous', '1 minute', '8 hours', 'Until dispelled')"
        ),
    )
    concentration: bool | None = Field(
        default=None, description="Whether the spell requires concentration"
    )
    ritual: bool | None = Field(
        default=None, description="Whether the spell can be cast as a ritual"
    )
    description: str | None = Field(
        default=None, description="Complete spell description including effects"
    )
    classes: list[DndClass] | None = Field(
        default=None, description="List of classes that can cast this spell"
    )
    damage_type: DamageTypeEnum | None = Field(
        default=None, description="Type of damage dealt (if applicable)"
    )
    attack_type: SpellAttackType | None = Field(
        default=None,
        description="Attack type (Melee/Ranged Spell Attack, Saving Throw, or None)",
    )
    save_type: AbilityScoreAbbreviation | None = Field(
        default=None, description="Saving throw type required (e.g., 'DEX', 'WIS')"
    )
    higher_levels: str | None = Field(
        default=None, description="Description of effects when cast at higher levels"
    )
    material_component_description: str | None = Field(
        default=None, description="Description of required material components (if any)"
    )


class Monster(BaseModel):
    name: str | None = Field(default=None, description="Monster or creature name")
    type: CreatureTypeEnum | None = Field(
        default=None,
        description="Creature type (e.g., Undead, Humanoid, Dragon, Beast, Aberration)",
    )
    size: SizeEnum | None = Field(
        default=None,
        description="Size category (Tiny, Small, Medium, Large, Huge, Gargantuan)",
    )
    alignment: AlignmentEnum | None = Field(
        default=None, description="Alignment (e.g., Lawful Good, Chaotic Evil, Neutral)"
    )
    description: str | None = Field(
        default=None,
        description="Physical description and appearance of the creature.",
    )
    armor_class: int | None = Field(default=None, description="Armor Class (AC) value")
    hit_points: int | None = Field(default=None, description="Average hit points")
    hit_dice: str | None = Field(
        default=None, description="Hit dice notation (e.g., '8d8+16', '12d10+36')"
    )
    speed: dict[str, int] | None = Field(
        default=None,
        description="Movement speeds in feet (e.g., {'walk': 30, 'fly': 60, 'swim': 30})",
    )
    strength: int | None = Field(default=None, description="Strength ability score")
    dexterity: int | None = Field(default=None, description="Dexterity ability score")
    constitution: int | None = Field(default=None, description="Constitution ability score")
    intelligence: int | None = Field(default=None, description="Intelligence ability score")
    wisdom: int | None = Field(default=None, description="Wisdom ability score")
    charisma: int | None = Field(default=None, description="Charisma ability score")
    challenge_rating: str | None = Field(
        default=None,
        description="Challenge Rating (CR) as a string (e.g., '1/8', '1/4', '1/2', '1', '13')",
    )
    experience_points: int | None = Field(
        default=None, description="Experience points awarded for defeating this creature"
    )
    saving_throws: dict[AbilityScoreAbbreviation, int] | None = Field(
        default=None,
        description="Saving throw bonuses (e.g., {'STR': 7, 'CON': 5})",
    )
    skills: dict[DndSkill, int] | None = Field(
        default=None,
        description="Skill bonuses (e.g., {'Stealth': 6, 'Perception': 4})",
    )
    damage_resistances: list[DamageTypeEnum] | None = Field(
        default=None, description="Damage types this creature resists"
    )
    damage_immunities: list[DamageTypeEnum] | None = Field(
        default=None, description="Damage types this creature is immune to"
    )
    damage_vulnerabilities: list[DamageTypeEnum] | None = Field(
        default=None, description="Damage types this creature is vulnerable to"
    )
    condition_immunities: list[ConditionEnum] | None = Field(
        default=None, description="Conditions this creature is immune to"
    )
    senses: dict[str, int] | None = Field(
        default=None,
        description=(
            "Special senses and ranges in feet (e.g., {'darkvision': 60, 'tremorsense': 30})"
        ),
    )
    languages: list[str] | None = Field(
        default=None, description="Languages the creature can speak or understand"
    )
    special_abilities: list[str] | None = Field(
        default=None, description="Names of special abilities or traits"
    )
    actions: list[str] | None = Field(
        default=None, description="Names of actions the creature can take"
    )
    legendary_actions: list[str] | None = Field(
        default=None, description="Names of legendary actions (if any)"
    )
    reactions: list[str] | None = Field(
        default=None, description="Names of reactions the creature can take"
    )


class Weapon(BaseModel):
    name: str | None = Field(default=None, description="Weapon name")
    category: WeaponCategory | None = Field(
        default=None, description="Weapon category (Simple Melee, Martial Ranged, etc.)"
    )
    cost: str | None = Field(default=None, description="Weapon cost (e.g., '10 gp', '50 gp')")
    damage_dice: str | None = Field(
        default=None, description="Damage dice notation (e.g., '1d8', '2d6')"
    )
    damage_type: DamageTypeEnum | None = Field(
        default=None, description="Type of damage dealt (Slashing, Piercing, Bludgeoning)"
    )
    weight: float | None = Field(default=None, description="Weapon weight in pounds")
    properties: list[str] | None = Field(
        default=None, description="Weapon properties (e.g., 'Finesse', 'Versatile', 'Heavy')"
    )
    range: str | None = Field(
        default=None, description="Weapon range (e.g., '30/120 ft.' for ranged weapons)"
    )


class Armor(BaseModel):
    name: str | None = Field(default=None, description="Armor name")
    category: str | None = Field(
        default=None, description="Armor category (e.g., 'Light Armor', 'Medium Armor')"
    )
    armor_class: str | None = Field(
        default=None, description="Base AC value or formula (e.g., '11', '12 + Dex modifier')"
    )
    strength_requirement: int | None = Field(
        default=None, description="Minimum Strength score required to wear without penalty"
    )
    stealth_disadvantage: bool | None = Field(
        default=None, description="Whether this armor imposes disadvantage on Stealth checks"
    )
    weight: float | None = Field(default=None, description="Armor weight in pounds")
    cost: str | None = Field(default=None, description="Armor cost (e.g., '5 gp', '200 gp')")
    armor_type: ArmorType | None = Field(
        default=None, description="Armor type (Light, Medium, Heavy, Shield)"
    )
    dex_bonus_max: int | None = Field(
        default=None, description="Maximum Dexterity bonus that can be added to AC"
    )


class MagicItem(BaseModel):
    name: str | None = Field(default=None, description="Magic item name")
    type: str | None = Field(
        default=None,
        description="Item type (e.g., 'Wondrous Item', 'Weapon', 'Armor', 'Potion')",
    )
    rarity: Rarity | None = Field(
        default=None,
        description="Item rarity (Common, Uncommon, Rare, Very Rare, Legendary, Artifact)",
    )
    requires_attunement: bool | None = Field(
        default=None, description="Whether the item requires attunement to use"
    )
    description: str | None = Field(
        default=None, description="Complete item description including properties and effects"
    )


class Feature(BaseModel):
    name: str | None = Field(
        default=None, description="Name of the feature, ability, or game mechanic"
    )
    description: str | None = Field(
        default=None, description="Complete description of what this feature does"
    )
    level_requirement: int | None = Field(
        default=None, description="Minimum character level required (if applicable)"
    )
    source_type: str | None = Field(
        default=None,
        description=(
            "What this feature belongs to. Examples: 'Class (Rogue)', 'Race (Elf)', "
            "'Magic Item (Deck of Many Things)', 'Monster (Dragon)', 'Game Mechanic'. "
            "Leave null if not clearly attributed to a specific class, race, item, "
            "or monster."
        ),
    )


class Condition(BaseModel):
    name: ConditionEnum | None = Field(
        default=None, description="Condition name (e.g., Blinded, Charmed, Paralyzed)"
    )
    description: str | None = Field(
        default=None, description="Description of what this condition does"
    )
    effects: list[str] | None = Field(
        default=None, description="List of specific mechanical effects of this condition"
    )


class Skill(BaseModel):
    name: DndSkill | None = Field(
        default=None, description="Skill name (e.g., Stealth, Perception, Athletics)"
    )
    description: str | None = Field(
        default=None, description="Description of what this skill represents"
    )
    ability: AbilityScoreEnum | None = Field(
        default=None, description="Associated ability score (e.g., Dexterity for Stealth)"
    )


class DamageType(BaseModel):
    name: DamageTypeEnum | None = Field(
        default=None, description="Damage type name (e.g., Fire, Cold, Piercing, Necrotic)"
    )
    description: str | None = Field(
        default=None, description="Description of this damage type"
    )


class Background(BaseModel):
    name: str | None = Field(
        default=None, description="Background name (e.g., Acolyte, Criminal, Sage)"
    )
    skill_proficiencies: list[DndSkill] | None = Field(
        default=None, description="Skills granted by this background"
    )
    tool_proficiencies: list[str] | None = Field(
        default=None, description="Tool proficiencies granted by this background"
    )
    languages: int | None = Field(
        default=None, description="Number of additional languages granted"
    )
    equipment: list[str] | None = Field(
        default=None, description="Starting equipment from this background"
    )
    feature: str | None = Field(default=None, description="Name of the background feature")
    description: str | None = Field(
        default=None, description="Complete background description"
    )


class Action(BaseModel):
    name: str | None = Field(
        default=None, description="Action name (e.g., Attack, Dash, Disengage, Help)"
    )
    action_type: ActionType | None = Field(
        default=None, description="Action type (Action, Bonus Action, Reaction, Free Action)"
    )
    description: str | None = Field(
        default=None, description="Description of what this action does"
    )


class Plane(BaseModel):
    name: str | None = Field(
        default=None, description="Plane name (e.g., Material Plane, Feywild, Nine Hells)"
    )
    description: str | None = Field(
        default=None, description="Description of this plane of existence"
    )
    traits: list[str] | None = Field(
        default=None, description="Planar traits and characteristics"
    )


class Deity(BaseModel):
    name: str | None = Field(default=None, description="Deity name")
    alignment: AlignmentEnum | None = Field(default=None, description="Deity's alignment")
    domains: list[str] | None = Field(
        default=None, description="Deity's domains (e.g., Life, War, Trickery)"
    )
    symbol: str | None = Field(default=None, description="Deity's holy symbol")
    description: str | None = Field(
        default=None, description="Description of this deity and their worship"
    )


class SchoolOfMagic(BaseModel):
    name: SchoolOfMagicEnum | None = Field(
        default=None,
        description="School of magic name (e.g., Evocation, Necromancy, Abjuration)",
    )
    description: str | None = Field(
        default=None, description="Description of this school of magic"
    )


class Language(BaseModel):
    name: str | None = Field(
        default=None, description="Language name (e.g., Common, Elvish, Draconic)"
    )
    script: str | None = Field(
        default=None, description="Script used for writing (e.g., Common, Elvish, Draconic)"
    )
    typical_speakers: list[str] | None = Field(
        default=None, description="Creatures that typically speak this language"
    )


class Environment(BaseModel):
    name: str | None = Field(
        default=None, description="Environment name (e.g., Forest, Desert, Underdark)"
    )
    description: str | None = Field(
        default=None, description="Description of this environment type"
    )


class Size(BaseModel):
    name: SizeEnum | None = Field(
        default=None,
        description="Size category (Tiny, Small, Medium, Large, Huge, Gargantuan)",
    )
    space: str | None = Field(
        default=None, description="Space controlled (e.g., '5 ft.', '10 ft.', '15 ft.')"
    )
    description: str | None = Field(
        default=None, description="Description of this size category"
    )


class Sense(BaseModel):
    name: str | None = Field(
        default=None, description="Sense name (e.g., Darkvision, Blindsight, Tremorsense)"
    )
    description: str | None = Field(
        default=None, description="Description of how this sense works"
    )
    range: int | None = Field(default=None, description="Range of this sense in feet")


class MovementType(BaseModel):
    name: str | None = Field(
        default=None, description="Movement type (e.g., Walk, Fly, Swim, Burrow, Climb)"
    )
    description: str | None = Field(
        default=None, description="Description of this movement type"
    )


class CreatureType(BaseModel):
    name: CreatureTypeEnum | None = Field(
        default=None, description="Creature type (e.g., Beast, Humanoid, Undead, Dragon)"
    )
    description: str | None = Field(
        default=None, description="Description of this creature type"
    )


class Feat(BaseModel):
    name: str | None = Field(
        default=None, description="Feat name (e.g., Alert, Lucky, War Caster)"
    )
    description: str | None = Field(default=None, description="Complete feat description")
    prerequisites: str | None = Field(
        default=None,
        description="Prerequisites to take this feat (e.g., 'Dexterity 13 or higher')",
    )
    benefits: list[str] | None = Field(
        default=None, description="List of benefits granted by this feat"
    )


class Tool(BaseModel):
    name: str | None = Field(
        default=None, description="Tool name (e.g., Thieves' Tools, Smith's Tools)"
    )
    category: str | None = Field(
        default=None, description="Tool category (e.g., Artisan's Tools, Gaming Set)"
    )
    cost: str | None = Field(default=None, description="Tool cost (e.g., '25 gp', '1 gp')")
    weight: float | None = Field(default=None, description="Tool weight in pounds")
    description: str | None = Field(
        default=None, description="Description of what this tool is used for"
    )


class Item(BaseModel):
    name: str | None = Field(
        default=None,
        description="Item name (e.g., 'Backpack', 'Ball Bearings', 'Rope', 'Torch')",
    )
    description: str | None = Field(
        default=None,
        description=(
            "Complete item description including mechanical effects, capacity, "
            "and usage rules. For consumables, include action economy "
            "(e.g., 'As a Utilize action...')."
        ),
    )
    cost: str | None = Field(
        default=None, description="Item cost (e.g., '2 GP', '5 SP', '1 CP')"
    )
    weight: float | None = Field(default=None, description="Item weight in pounds")
    category: str | None = Field(
        default=None,
        description=(
            "Item category for classification (e.g., 'Container', 'Light Source', "
            "'Consumable', 'Utility', 'Exploration Gear'). Keep concise."
        ),
    )


class Vehicle(BaseModel):
    name: str | None = Field(
        default=None, description="Vehicle name (e.g., Rowboat, Airship, Wagon)"
    )
    type: str | None = Field(
        default=None, description="Vehicle type (e.g., Water, Air, Land)"
    )
    cost: str | None = Field(
        default=None, description="Vehicle cost (e.g., '50 gp', '20,000 gp')"
    )
    speed: str | None = Field(
        default=None, description="Vehicle speed (e.g., '1.5 mph', '8 mph')"
    )
    capacity: str | None = Field(
        default=None, description="Carrying capacity or passenger count"
    )
    description: str | None = Field(default=None, description="Complete vehicle description")


class ExtractedEntities(BaseModel):
    """Container for all entities extracted from a text chunk.

    DEPRECATED: This model is too complex for Gemini's structured output API.
    Schema size: ~26,871 characters exceeds Gemini's complexity limits.

    Use the focused extraction models below instead:
    - GameplayEntities (42.1% of full schema)
    - EquipmentEntities (20.1% of full schema)
    - CharacterEntities (12.4% of full schema)
    - DescriptorEntities (14-17% of full schema)
    - WorldEntities (12.1% of full schema)
    """

    classes: list[Class] = Field(default_factory=list)
    subclasses: list[Subclass] = Field(default_factory=list)
    races: list[Race] = Field(default_factory=list)
    spells: list[Spell] = Field(default_factory=list)
    monsters: list[Monster] = Field(default_factory=list)
    weapons: list[Weapon] = Field(default_factory=list)
    armor: list[Armor] = Field(default_factory=list)
    magic_items: list[MagicItem] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    damage_types: list[DamageType] = Field(default_factory=list)
    backgrounds: list[Background] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    planes: list[Plane] = Field(default_factory=list)
    deities: list[Deity] = Field(default_factory=list)
    schools_of_magic: list[SchoolOfMagic] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    environments: list[Environment] = Field(default_factory=list)
    sizes: list[Size] = Field(default_factory=list)
    senses: list[Sense] = Field(default_factory=list)
    movement_types: list[MovementType] = Field(default_factory=list)
    creature_types: list[CreatureType] = Field(default_factory=list)
    feats: list[Feat] = Field(default_factory=list)
    tools: list[Tool] = Field(default_factory=list)
    vehicles: list[Vehicle] = Field(default_factory=list)


# ==============================================================================
# FOCUSED EXTRACTION MODELS
# ==============================================================================
# These models break down the full ExtractedEntities into smaller, focused
# schemas that work with Gemini's structured output API (complexity limits).
# All 30 entities are covered across these 5 models.
# ==============================================================================


class GameplayEntities(BaseModel):
    """Extract core gameplay entities: spells, monsters, classes, races, subclasses.

    Use this model when processing SRD sections containing:
    - Spell descriptions and spellcasting rules
    - Monster stat blocks and bestiary entries
    - Class features and progression tables
    - Race traits and ability score increases
    - Subclass descriptions and features

    Schema size: ~11,300 chars (42.1% of full ExtractedEntities)
    """

    spells: list[Spell] = Field(default_factory=list)
    monsters: list[Monster] = Field(default_factory=list)
    classes: list[Class] = Field(default_factory=list)
    races: list[Race] = Field(default_factory=list)
    subclasses: list[Subclass] = Field(default_factory=list)


class EquipmentEntities(BaseModel):
    """Extract equipment and items: weapons, armor, magic items, tools, vehicles,
    adventuring gear.

    Use this model when processing SRD sections containing:
    - Equipment lists and pricing
    - Weapon and armor properties
    - Magic item descriptions
    - Tool proficiencies
    - Vehicle stats and capacity
    - Adventuring gear (Backpack, Rope, Torches, etc.)

    Schema size: ~5,400 chars (20.1% of full ExtractedEntities)
    """

    weapons: list[Weapon] = Field(default_factory=list)
    armor: list[Armor] = Field(default_factory=list)
    magic_items: list[MagicItem] = Field(default_factory=list)
    tools: list[Tool] = Field(default_factory=list)
    vehicles: list[Vehicle] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)


class CharacterEntities(BaseModel):
    """Extract character-related entities: backgrounds, feats, features, actions.

    Use this model when processing SRD sections containing:
    - Character background options
    - Feat descriptions and prerequisites
    - Class/race features
    - Combat action descriptions

    Schema size: ~3,332 chars (12.4% of full ExtractedEntities)
    """

    backgrounds: list[Background] = Field(default_factory=list)
    feats: list[Feat] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)


class DescriptorEntities(BaseModel):
    """Extract descriptor entities: conditions, skills, damage types, sizes, schools of magic.

    Use this model when processing SRD sections containing:
    - Condition definitions (Blinded, Charmed, etc.)
    - Skill descriptions with ability associations
    - Damage type explanations
    - Creature size mechanics
    - Creature type descriptions
    - School of magic descriptions

    These entities become nodes in the knowledge graph for player queries.
    """

    conditions: list[Condition] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    damage_types: list[DamageType] = Field(default_factory=list)
    schools_of_magic: list[SchoolOfMagic] = Field(default_factory=list)
    sizes: list[Size] = Field(default_factory=list)
    creature_types: list[CreatureType] = Field(default_factory=list)


class WorldEntities(BaseModel):
    """Extract world-building entities: planes, deities, environments, languages, etc.

    Use this model when processing SRD sections containing:
    - Cosmology and plane descriptions
    - Deity pantheons and domains
    - Language families
    - Environment/terrain types
    - Creature senses
    - Movement types

    Schema size: ~3,251 chars (12.1% of full ExtractedEntities)
    """

    planes: list[Plane] = Field(default_factory=list)
    deities: list[Deity] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    environments: list[Environment] = Field(default_factory=list)
    senses: list[Sense] = Field(default_factory=list)
    movement_types: list[MovementType] = Field(default_factory=list)
