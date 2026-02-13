# D&D 5E Entity Extraction

Extract structured D&D 5th Edition game data from text. Only extract entity types that exist in the provided schema. If the text describes an entity type not in your schema, return empty lists. Do not force-fit entities into wrong types.

## Entity Types

### Monsters
Creatures with stat blocks, NPCs with combat statistics, or summoned creatures.

Indicators: ability scores, AC, HP, CR, actions, "stat block", "appears"

When a chunk contains a stat block, populate every available field:
- All 6 ability scores: STR, DEX, CON, INT, WIS, CHA
- armor_class, hit_points, hit_dice
- speed dict with keys: walk, fly, swim, burrow, climb
- damage_immunities, damage_resistances, damage_vulnerabilities
- condition_immunities — parse from the "Immunities" line after the semicolon
- senses dict with keys: darkvision, blindsight, tremorsense, truesight
- languages, challenge_rating (fraction string, e.g. "1/8"), experience_points, alignment
- saving_throws, skills, special_abilities, actions, legendary_actions, reactions

### Features
Class or race abilities, game mechanics, background features, feats, and character options.

Do not extract as Feature:
- Creatures with stats — use Monster
- Items with rarity ratings — use MagicItem
- Spells with casting time — use Spell
- Weapons or armor — use Weapon or Armor
- Text starting with "Wondrous Item" or listing rarity — use MagicItem

### Magic Items
Wondrous items, artifacts, items with magical properties, items requiring attunement, or objects with rarity ratings.

Indicators: "Wondrous Item", "Weapon (+1)", "Armor", rarity rating

If text describes a magic item, extract only as MagicItem, not as Feature.

### Spells
Spells with casting time, range, and components. Includes cantrips and leveled spells.

Indicators: spell level, school of magic, casting time, components (V, S, M)

### Equipment
Weapons with damage dice and properties, armor with AC values, mundane equipment.

Indicators: damage dice, AC values, weight, cost

## Field Guidelines

### source_type (Feature only)
- Specific attribution: "Magic Item (Deck of Many Things)", "Class (Rogue)"
- General rules: "Game Mechanic"
- No clear source: null
- Keep under 50 characters. No instructions or explanations.

### Monster vs Feature overlap
- Text mentions a creature with combat abilities: extract as Monster
- Text describes an effect that summons a creature: extract both Monster and Feature
- The Feature describes the event. The Monster describes the creature.

## Rules

1. Extract all entities, even with partial data.
2. Each entity appears in one category only. No duplicates.
3. Prefer specific types over Feature: Monster > Feature, MagicItem > Feature, Spell > Feature.
4. Include full text in description fields. Do not truncate.
5. Match enum values exactly. Use "Undead" not "undead".
6. Use null for unknown values. Do not guess or leave blank strings.
7. Do not extract core game actions (Attack, Dash, Utilize, Help) as entities.
8. Only extract entities being defined in the text, not just mentioned.
   - "scattered in the Astral Plane" — do not extract Astral Plane
   - A dedicated section defining the Astral Plane — extract it
9. Use only information from the source text. Do not add definitions, lore, or descriptions from external knowledge. If a description cannot be filled from the source text, use null.

## Examples

### Monster with Event Context

Input:
```
## Skull
An Avatar of Death appears in an unoccupied space as close to you as possible.
The avatar targets only you with its attacks, appearing as a ghostly skeleton
clad in a tattered black robe carrying a spectral scythe. The avatar disappears
when it drops to 0 Hit Points or you die.
```

Output:
```json
{
  "monsters": [
    {
      "name": "Avatar of Death",
      "type": "Undead",
      "description": "A ghostly skeleton clad in a tattered black robe carrying a spectral scythe"
    }
  ],
  "features": [
    {
      "name": "Skull",
      "description": "An Avatar of Death (see the accompanying stat block) appears in an unoccupied space as close to you as possible. The avatar targets only you with its attacks, appearing as a ghostly skeleton clad in a tattered black robe carrying a spectral scythe. The avatar disappears when it drops to 0 Hit Points or you die. If an ally of yours deals damage to the avatar, that ally summons another Avatar of Death. The new avatar appears in an unoccupied space as close to that ally as possible and targets only that ally with its attacks. You and your allies can each summon only one avatar as a consequence of this draw. A creature slain by an avatar can't be restored to life.",
      "source_type": "Magic Item (Deck of Many Things)"
    }
  ]
}
```

### Spell

Input:
```
## Fireball
3 Evocation

Casting Time: 1 action
Range: 150 feet
Target: A point you choose within range
Components: V S M (A tiny ball of bat guano and sulfur)
Duration: Instantaneous
Classes: Sorcerer, Wizard

A bright streak flashes from your pointing finger to a point you choose within range and then blossoms with a low roar into an explosion of flame. Each creature in a 20-foot-radius sphere centered on that point must make a Dexterity saving throw. A target takes 8d6 fire damage on a failed save, or half as much damage on a successful one. The fire spreads around corners. It ignites flammable objects in the area that aren't being worn or carried.

At Higher Levels: When you cast this spell using a spell slot of 4th level or higher, the damage increases by 1d6 for each slot level above 3rd.
```

Output:
```json
{
  "spells": [
    {
      "name": "Fireball",
      "level": 3,
      "school": "Evocation",
      "casting_time": "1 action",
      "range": "150 feet",
      "components": ["V", "S", "M"],
      "duration": "Instantaneous",
      "classes": ["Sorcerer", "Wizard"],
      "damage_type": "Fire",
      "save_type": "DEX",
      "material_component_description": "A tiny ball of bat guano and sulfur",
      "description": "A bright streak flashes from your pointing finger to a point you choose within range and then blossoms with a low roar into an explosion of flame. Each creature in a 20-foot-radius sphere centered on that point must make a Dexterity saving throw. A target takes 8d6 fire damage on a failed save, or half as much damage on a successful one. The fire spreads around corners. It ignites flammable objects in the area that aren't being worn or carried.",
      "higher_levels": "When you cast this spell using a spell slot of 4th level or higher, the damage increases by 1d6 for each slot level above 3rd."
    }
  ]
}
```

### Class Feature

Input:
```
## Rage
In battle, you fight with primal ferocity. On your turn, you can enter a rage as a bonus action.

While raging, you gain the following benefits if you aren't wearing heavy armor:

• You have advantage on Strength checks and Strength saving throws.
• When you make a melee weapon attack using Strength, you gain a +2 bonus to the damage roll. This bonus increases as you level.
• You have resistance to bludgeoning, piercing, and slashing damage.

If you are able to cast spells, you can't cast them or concentrate on them while raging.

Your rage lasts for 1 minute. It ends early if you are knocked unconscious or if your turn ends and you haven't attacked a hostile creature since your last turn or taken damage since then. You can also end your rage on your turn as a bonus action.

Once you have raged the maximum number of times for your barbarian level, you must finish a long rest before you can rage again. You may rage 2 times at 1st level, 3 at 3rd, 4 at 6th, 5 at 12th, and 6 at 17th.

Attributes
Subtype: Barbarian
Type: Class Feature
```

Output:
```json
{
  "features": [
    {
      "name": "Rage",
      "description": "In battle, you fight with primal ferocity. On your turn, you can enter a rage as a bonus action. While raging, you gain the following benefits if you aren't wearing heavy armor: You have advantage on Strength checks and Strength saving throws. When you make a melee weapon attack using Strength, you gain a +2 bonus to the damage roll. This bonus increases as you level. You have resistance to bludgeoning, piercing, and slashing damage. If you are able to cast spells, you can't cast them or concentrate on them while raging. Your rage lasts for 1 minute. It ends early if you are knocked unconscious or if your turn ends and you haven't attacked a hostile creature since your last turn or taken damage since then. You can also end your rage on your turn as a bonus action. Once you have raged the maximum number of times for your barbarian level, you must finish a long rest before you can rage again. You may rage 2 times at 1st level, 3 at 3rd, 4 at 6th, 5 at 12th, and 6 at 17th.",
      "source_type": "Class (Barbarian)",
      "level_requirement": 1
    }
  ]
}
```

### Magic Item

Input:
```
## Bag of Holding
Wondrous Item, Uncommon

This bag has an interior space considerably larger than its outside dimensions—roughly 2 feet square and 4 feet deep on the inside. The bag can hold up to 500 pounds, not exceeding a volume of 64 cubic feet. The bag weighs 5 pounds, regardless of its contents. Retrieving an item from the bag requires a Utilize action.

If the bag is overloaded, pierced, or torn, it is destroyed, and its contents are scattered in the Astral Plane.
```

Output:
```json
{
  "magic_items": [
    {
      "name": "Bag of Holding",
      "type": "Wondrous Item",
      "rarity": "Uncommon",
      "requires_attunement": false,
      "description": "This bag has an interior space considerably larger than its outside dimensions—roughly 2 feet square and 4 feet deep on the inside. The bag can hold up to 500 pounds, not exceeding a volume of 64 cubic feet. The bag weighs 5 pounds, regardless of its contents. Retrieving an item from the bag requires a Utilize action. If the bag is overloaded, pierced, or torn, it is destroyed, and its contents are scattered in the Astral Plane."
    }
  ]
}
```

### Monster Stat Block

Input:
```
## Skeleton Archer
*Medium Undead, Lawful Evil*

**AC** 13 (armor scraps)
**HP** 13 (2d8+4)
**Speed** 30 ft.

| Stat | Value | Mod | Save |
| STR  | 10    | +0  | +0   |
| DEX  | 14    | +2  | +2   |
| CON  | 15    | +2  | +2   |
| INT  | 6     | -2  | -2   |
| WIS  | 8     | -1  | -1   |
| CHA  | 5     | -3  | -3   |

**Immunities** Poison; Exhaustion, Poisoned
**Senses** Darkvision 60 ft., Passive Perception 9
**Languages** Understands Common but can't speak
**CR** 1/4 (XP 50; PB +2)
```

Output:
```json
{
  "monsters": [
    {
      "name": "Skeleton Archer",
      "type": "Undead",
      "size": "Medium",
      "alignment": "Lawful Evil",
      "armor_class": 13,
      "hit_points": 13,
      "hit_dice": "2d8+4",
      "speed": {"walk": 30},
      "strength": 10,
      "dexterity": 14,
      "constitution": 15,
      "intelligence": 6,
      "wisdom": 8,
      "charisma": 5,
      "damage_immunities": ["Poison"],
      "condition_immunities": ["Exhaustion", "Poisoned"],
      "senses": {"darkvision": 60},
      "languages": ["Understands Common but can't speak"],
      "challenge_rating": "1/4",
      "experience_points": 50
    }
  ]
}
```

The "Immunities" line uses a semicolon separator. Damage immunities appear before the semicolon. Condition immunities appear after it.

---

## Task

Extract all entities from the following text according to the provided schema.

{{ text }}
