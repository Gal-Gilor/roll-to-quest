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

### Spells
Spells with casting time, range, and components. Includes cantrips and leveled spells.

Indicators: spell level, school of magic, casting time, components (V, S, M)

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
