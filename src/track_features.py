import math

BPM_MIN = 100.0
BPM_RANGE = 100.0  # covers 100–200 BPM
CAMELOT_SIZE = 24  # 1A-12A = 1-12, 1B-12B = 13-24, 0 = unknown

# Standard pitch notation to Camelot string.
# Covers both sharp and flat spellings for each position.
_STANDARD_TO_CAMELOT = {
    # Major keys → B side
    "B": "1B",  "Cb": "1B",
    "Gb": "2B", "F#": "2B",
    "Db": "3B", "C#": "3B",
    "Ab": "4B", "G#": "4B",
    "Eb": "5B", "D#": "5B",
    "Bb": "6B", "A#": "6B",
    "F":  "7B",
    "C":  "8B",
    "G":  "9B",
    "D":  "10B",
    "A":  "11B",
    "E":  "12B",
    # Minor keys → A side
    "G#m": "1A",  "Abm": "1A",
    "D#m": "2A",  "Ebm": "2A",
    "A#m": "3A",  "Bbm": "3A",
    "Fm":  "4A",
    "Cm":  "5A",
    "Gm":  "6A",
    "Dm":  "7A",
    "Am":  "8A",
    "Em":  "9A",
    "Bm":  "10A",
    "F#m": "11A", "Gbm": "11A",
    "C#m": "12A", "Dbm": "12A",
}


# Converts a Camelot string (e.g. "7A") to its integer index.
# In:
# - camelot: string like "7A" or "12B".
# Out:
# - integer 1..24, or 0 if unparseable.
def _camelot_str_to_index(camelot: str) -> int:
    camelot = camelot.strip().upper()
    if len(camelot) < 2:
        return 0
    letter = camelot[-1]
    if letter not in ("A", "B"):
        return 0
    try:
        num = int(camelot[:-1])
    except ValueError:
        return 0
    if num < 1 or num > 12:
        return 0
    return num if letter == "A" else num + 12


# Encodes a key string to a Camelot integer index.
# Accepts Camelot notation ("7A", "12B") or standard pitch notation ("Am", "C#").
# In:
# - key: key string or None.
# Out:
# - integer in range 0..24; 0 = unknown/OOV.
def encode_camelot_key(key: str | None) -> int:
    if not key:
        return 0
    key = key.strip()

    # Try Camelot notation first (ends in A or B after a number)
    upper = key.upper()
    if upper[-1] in ("A", "B") and upper[:-1].isdigit():
        return _camelot_str_to_index(upper)

    # Try standard pitch notation via lookup
    camelot = _STANDARD_TO_CAMELOT.get(key) or _STANDARD_TO_CAMELOT.get(key.capitalize())
    if camelot:
        return _camelot_str_to_index(camelot)

    return 0


# Encodes BPM as a 2-float vector [normalized_bpm, bpm_known].
# Normalizes against the 100–200 BPM range typical of this genre.
# In:
# - bpm: BPM float or None.
# Out:
# - [normalized, known]: normalized in [0, 1], known is 1.0 if bpm is present else 0.0.
def normalize_bpm(bpm: float | None) -> list[float]:
    if bpm is None:
        return [0.0, 0.0]
    normalized = max(0.0, min((float(bpm) - BPM_MIN) / BPM_RANGE, 1.0))
    return [normalized, 1.0]


# Encodes a musical key as a 4-float circular vector [sin, cos, mode, known].
# The Camelot number (1–12) is mapped onto a unit circle so that adjacent positions
# are geometrically close, including the wrap-around from 12 back to 1.
# In:
# - key: key string (Camelot or standard pitch notation) or None.
# Out:
# - [sin, cos, mode, known]: mode is 0.0 for minor (A), 1.0 for major (B);
#   known is 1.0 if the key was parsed successfully, 0.0 otherwise.
def encode_key_circular(key: str | None) -> list[float]:
    idx = encode_camelot_key(key)
    if idx == 0:
        return [0.0, 0.0, 0.0, 0.0]
    num = idx if idx <= 12 else idx - 12   # 1..12 (Camelot position)
    mode = 0.0 if idx <= 12 else 1.0       # 0 = minor/A, 1 = major/B
    angle = 2 * math.pi * (num - 1) / 12
    return [math.sin(angle), math.cos(angle), mode, 1.0]
