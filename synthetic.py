import csv
import random
from collections import defaultdict

REAL_DATA_FILE = "data/training_features.csv"
OUTPUT_FILE    = "data/synthetic_features.csv"
N_ROWS         = 4000
SEED           = 44

random.seed(SEED)

BLEACHING_VALUES     = [1, 2, 3, 4]
DURATION_VALUES      = [round(v * 0.5, 1) for v in range(2, 15)]   # 1.0 … 7.0
CONCENTRATION_VALUES = list(range(1000, 7001, 250))

TYPE_PROPS_COLS = [
    "coloring_type", "yarn_number", "frequency", "knitting",
    "fabric_elasticity", "cielab_l_raw", "cielab_a_raw", "cielab_b_raw",
    "initial_filename",
]

type_rows = defaultdict(list)
with open(REAL_DATA_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        type_rows[int(row["type"])].append(row)

# Take the first occurrence for each type as the canonical properties
TYPE_PROPS = {
    t: {col: rows[0][col] for col in TYPE_PROPS_COLS}
    for t, rows in type_rows.items()
}

recipe_counter = defaultdict(int)

FIELDNAMES = [
    "type", "recipe", "bleaching", "duration", "concentration",
    "coloring_type", "yarn_number", "frequency", "knitting",
    "fabric_elasticity", "cielab_l_raw", "cielab_a_raw", "cielab_b_raw",
    "initial_filename", "target_filename",
]

rows = []
for _ in range(N_ROWS):
    type_id = random.randint(1, 50)
    props   = TYPE_PROPS[type_id]

    bleaching     = random.choice(BLEACHING_VALUES)
    duration      = random.choice(DURATION_VALUES)
    concentration = random.choice(CONCENTRATION_VALUES)

    recipe_counter[type_id] += 1
    recipe = recipe_counter[type_id]
    repl   = ((recipe - 1) % 3) + 1

    rows.append({
        "type"             : type_id,
        "recipe"           : recipe,
        "bleaching"        : bleaching,
        "duration"         : duration,
        "concentration"    : concentration,
        "coloring_type"    : props["coloring_type"],
        "yarn_number"      : props["yarn_number"],
        "frequency"        : props["frequency"],
        "knitting"         : props["knitting"],
        "fabric_elasticity": props["fabric_elasticity"],
        "cielab_l_raw"     : props["cielab_l_raw"],
        "cielab_a_raw"     : props["cielab_a_raw"],
        "cielab_b_raw"     : props["cielab_b_raw"],
        "initial_filename" : props["initial_filename"],
        "target_filename"  : f"tip{type_id}-recete{recipe}-repl{repl}.jpg",
    })

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"✓  {N_ROWS} rows written to '{OUTPUT_FILE}'")
print(f"   Types covered    : {len(set(r['type'] for r in rows))}/50")
print(f"   Bleaching dist   : { {v: sum(1 for r in rows if r['bleaching']==v) for v in BLEACHING_VALUES} }")
print(f"   Duration  range  : [{min(r['duration'] for r in rows)}, {max(r['duration'] for r in rows)}]")
print(f"   Conc.     range  : [{min(r['concentration'] for r in rows)}, {max(r['concentration'] for r in rows)}]")