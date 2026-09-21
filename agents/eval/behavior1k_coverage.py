"""BEHAVIOR-1K Tier-0 coverage analysis: for each of the 1018 BDDL activities,
how much of its object/room requirement is already satisfiable by SimAny
(vocabulary match) or already concretely present in our 50 processed
ScanNet++ val scenes (actually-discovered-label match)?

No simulator, no GPU - pure text parsing + set overlap against data we
already have on disk. Run: python3 behavior1k_coverage.py
"""
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BDDL = REPO / "third_party" / "bddl_data"

# -------------------------------------------------------------- synset db --
synset_type = {}
with open(BDDL / "generated_data" / "synsets.csv") as f:
    for row in csv.DictReader(f):
        synset_type[row["synset"]] = row["objectType"]

RIGID_LIKE = {"rigidBody", "softBody", "cloth", "rope"}  # our export can at
# least approximate these as rigid bodies (paper's own stated limitation for
# softBody/cloth/rope); liquid/*Substance/visualSubstance need fluid/particle
# sim we don't have at all.

PLAUSIBLE_ROOMS = {
    "bathroom", "bedroom", "corridor", "dining_room", "living_room",
    "private_office", "shared_office", "lobby", "entryway", "closet",
    "pantry_room", "staircase", "storage_room", "utility_room",
    "television_room", "childs_room", "playroom", "exercise_room",
    "kitchen", "empty_room", "break_room",
}
IMPLAUSIBLE_ROOMS = {
    "bar", "biology_lab", "chemistry_lab", "classroom", "computer_lab",
    "conference_hall", "copy_room", "grocery_store", "gym", "hammam",
    "infirmary", "locker_room", "meeting_room", "phone_room", "sauna",
    "spa", "garage", "garden",
}

# ------------------------------------------------------ our vocabulary(s) --
import sys
sys.path.insert(0, str(REPO))
from agents.core import common as C  # noqa: E402

our_vocab = set(C.VOCAB_SMALL) | set(C.VOCAB_FURNITURE)

# Built-in fixtures: plumbed/wired into the room, part of the base scan mesh
# in any indoor capture (like STRUCTURAL_EXCLUDE, but plumbing/cabinetry
# fixtures rather than the building shell itself) - not something SimAny
# needs to discover-and-generate as a discrete movable asset. This list is
# specific to this coverage analysis, NOT a change to the pipeline's own
# common.STRUCTURAL_EXCLUDE.
FIXTURE_EXCLUDE = {
    "sink", "faucet", "toilet", "bathtub", "tub", "shower",
    "shower stall", "stove", "cooktop", "range hood", "exhaust hood",
    "countertop", "counter top", "counter", "bathroom counter",
    "kitchen counter", "electric refrigerator", "refrigerator", "fridge",
    "dishwasher", "washer", "clothes dryer", "dryer", "fireplace",
    "hvac", "thermostat", "wall socket", "wall outlet",
}


def norm(word):
    return re.sub(r"[_.]", " ", word).strip().lower()


our_vocab_norm = {norm(w) for w in our_vocab}

discovered_labels = set()
for p in (REPO / "outputs").glob("*_factory/objects/objects.json"):
    try:
        for m in json.loads(p.read_text()):
            discovered_labels.add(m["label"].strip().lower())
    except Exception:
        pass


def synset_to_words(synset):
    """acetone__atomizer.n.01 -> {'acetone atomizer', 'atomizer'} etc."""
    base = re.sub(r"\.n\.\d+$", "", synset)
    words = [norm(base)]
    parts = base.split("__")
    if len(parts) > 1:
        words.append(norm(parts[-1]))  # the head noun, e.g. 'atomizer'
    return set(w for w in words if w)


def matches(words, vocab):
    for w in words:
        if w in vocab:
            return True
        # loose containment (e.g. 'plastic bottle' vs 'bottle')
        if any(w == v or w.endswith(" " + v) or v.endswith(" " + w)
               for v in vocab):
            return True
    return False


def synset_matches(synset, vocab):
    # substance-prefixed compounds (e.g. detergent__bottle.n.01): the bare
    # generic container word alone must NOT count - the substance/content
    # part before "__" has to be covered too, else no match.
    parts = re.sub(r"\.n\.\d+$", "", synset).split("__")
    if len(parts) > 1 and not all(matches({norm(p)}, vocab) for p in parts[:-1]):
        return False
    return matches(synset_to_words(synset), vocab)


# ------------------------------------------------------- parse activities --
def parse_bddl(path):
    txt = path.read_text()
    obj_block = re.search(r"\(:objects(.*?)\)\s*\n\s*\(:init", txt, re.S)
    objects = []
    if obj_block:
        for line in obj_block.group(1).splitlines():
            line = line.strip()
            m = re.match(r"^(.+?)\s*-\s*(\S+)$", line)
            if m:
                objects.append(m.group(2))
    rooms = set(re.findall(r"\(inroom\s+\S+\s+(\w+)\)", txt))
    return objects, rooms


results = []
for act_dir in sorted((BDDL / "activity_definitions").iterdir()):
    probs = sorted(act_dir.glob("problem*.bddl"))
    if not probs:
        continue
    objects, rooms = parse_bddl(probs[0])
    def is_fixture_or_structural(s):
        words = synset_to_words(s)
        return any(w in FIXTURE_EXCLUDE or norm(w).rstrip("s") in C.STRUCTURAL_EXCLUDE
                   for w in words)

    rigid_needed = sorted({s for s in objects
                           if s != "agent.n.01"
                           and synset_type.get(s, "rigidBody") in RIGID_LIKE
                           and not is_fixture_or_structural(s)})
    if not rigid_needed:
        continue  # pure-substance/ability tasks, not in our object domain at all

    vocab_hits = sum(1 for s in rigid_needed
                     if synset_matches(s, our_vocab_norm))
    disc_hits = sum(1 for s in rigid_needed
                    if synset_matches(s, discovered_labels))
    rooms_ok = bool(rooms) and rooms <= PLAUSIBLE_ROOMS
    rooms_bad = bool(rooms & IMPLAUSIBLE_ROOMS)

    results.append({
        "activity": act_dir.name,
        "n_rigid_objects": len(rigid_needed),
        "vocab_coverage": vocab_hits / len(rigid_needed),
        "discovered_coverage": disc_hits / len(rigid_needed),
        "rooms": sorted(rooms),
        "rooms_plausible": rooms_ok and not rooms_bad,
    })

n = len(results)
full_vocab = sum(1 for r in results if r["vocab_coverage"] >= 0.999)
full_disc = sum(1 for r in results if r["discovered_coverage"] >= 0.999)
full_both = sum(1 for r in results
                if r["vocab_coverage"] >= 0.999 and r["rooms_plausible"])
full_disc_room = sum(1 for r in results
                     if r["discovered_coverage"] >= 0.999 and r["rooms_plausible"])
room_ok = sum(1 for r in results if r["rooms_plausible"])

print(f"activities with >=1 rigid-like object requirement: {n} / 1018 total")
print(f"  room-type plausible for a ScanNet++-style indoor scene: {room_ok} "
      f"({room_ok/n:.1%})")
print(f"  100% object-category vocab coverage (any room):        {full_vocab} "
      f"({full_vocab/n:.1%})")
print(f"  100% ALREADY-DISCOVERED-LABEL coverage (any room):      {full_disc} "
      f"({full_disc/n:.1%})")
print(f"  100% vocab coverage AND plausible room:                 {full_both} "
      f"({full_both/n:.1%})")
print(f"  100% already-discovered coverage AND plausible room:    {full_disc_room} "
      f"({full_disc_room/n:.1%})")
# headline "fully groundable today": room-gated filter for BOTH the count and
# the percentage (never mix the ungated pct with the gated count)
print(f"  => HEADLINE fully groundable: {full_disc_room}/{n} = {full_disc_room/n:.1%}")

import statistics
print(f"\nmean vocab_coverage across all {n}: "
      f"{statistics.mean(r['vocab_coverage'] for r in results):.3f}")
print(f"mean discovered_coverage across all {n}: "
      f"{statistics.mean(r['discovered_coverage'] for r in results):.3f}")

print("\nsample of fully-groundable-today activities "
      "(100% discovered-label coverage + plausible room):")
shown = 0
for r in results:
    if r["discovered_coverage"] >= 0.999 and r["rooms_plausible"] and shown < 15:
        print(f"  {r['activity']:40s} rooms={r['rooms']}")
        shown += 1

out = {
    "n_object_activities": n,
    "n_total_activities": 1018,
    "room_plausible": room_ok,
    "full_vocab_coverage_ungated": full_vocab,
    "full_discovered_coverage_ungated": full_disc,
    "full_vocab_and_room": full_both,
    "full_discovered_and_room": full_disc_room,
    # headline fully-groundable figure: room-gated count AND room-gated pct
    "fully_groundable_count": full_disc_room,
    "fully_groundable_pct": full_disc_room / n,
    "mean_vocab_coverage": statistics.mean(r["vocab_coverage"] for r in results),
    "mean_discovered_coverage": statistics.mean(r["discovered_coverage"] for r in results),
    "per_activity": results,
}
(C.ROOT / "outputs" / "behavior1k_coverage.json").write_text(
    json.dumps(out, indent=1))
print("\nwrote outputs/behavior1k_coverage.json")
