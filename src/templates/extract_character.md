## Entity Types

### Features
Named class or race abilities, background features, feats, and specific character options.

Do not extract as Feature:
- Core game rules or procedures (ability checks, advantage/disadvantage, proficiency, rolling dice) — these are rules, not entities
- Creatures with stats — use Monster
- Items with rarity ratings — use MagicItem
- Spells with casting time — use Spell
- Weapons or armor — use Weapon or Armor
- Text starting with "Wondrous Item" or listing rarity — use MagicItem

## Examples

### Monster with Event Context (Feature part)

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
  "features": [
    {
      "name": "Skull",
      "description": "An Avatar of Death (see the accompanying stat block) appears in an unoccupied space as close to you as possible. The avatar targets only you with its attacks, appearing as a ghostly skeleton clad in a tattered black robe carrying a spectral scythe. The avatar disappears when it drops to 0 Hit Points or you die. If an ally of yours deals damage to the avatar, that ally summons another Avatar of Death. The new avatar appears in an unoccupied space as close to that ally as possible and targets only that ally with its attacks. You and your allies can each summon only one avatar as a consequence of this draw. A creature slain by an avatar can't be restored to life.",
      "source_type": "Magic Item (Deck of Many Things)"
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

- You have advantage on Strength checks and Strength saving throws.
- When you make a melee weapon attack using Strength, you gain a +2 bonus to the damage roll. This bonus increases as you level.
- You have resistance to bludgeoning, piercing, and slashing damage.

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

---

## Task

Extract all entities from the following text according to the provided schema.

{{ text }}
