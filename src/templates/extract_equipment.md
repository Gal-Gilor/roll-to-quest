## Entity Types

### Magic Items
Wondrous items, artifacts, items with magical properties, items requiring attunement, or objects with rarity ratings.

Indicators: "Wondrous Item", "Weapon (+1)", "Armor", rarity rating

If text describes a magic item, extract only as MagicItem, not as Feature.

### Equipment
Weapons with damage dice and properties, armor with AC values, mundane equipment.

Indicators: damage dice, AC values, weight, cost

## Examples

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

---

## Task

Extract all entities from the following text according to the provided schema.

{{ text }}
