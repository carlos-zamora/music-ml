import re
import torch


# Extracts featured artists and remixers from a track title.
# In:
# - title: track title string.
# Out:
# - list of artist name strings found in the title.
def parse_featured_artists(title: str) -> list:
    results = []

    # feat/ft/featuring: "(feat. X & Y)", "feat X", "ft. X", "featuring X"
    feat_pattern = re.compile(
        r'[\(\[]?(?:feat(?:uring)?\.?|ft\.)\s+([^\)\]\-–]+?)(?=[\)\]\-–]|$)',
        re.IGNORECASE,
    )
    for match in feat_pattern.finditer(title):
        for name in re.split(r'\s*[&,]\s*', match.group(1)):
            name = name.strip()
            if name:
                results.append(name)

    # remix/edit/rework: "(Name Remix)", "[Name Edit]"
    rework_pattern = re.compile(
        r'[\(\[]\s*(.+?)\s+(?:remix|edit|rework|flip)\s*[\)\]]',
        re.IGNORECASE,
    )
    for match in rework_pattern.finditer(title):
        for name in re.split(r'\s*[&,]\s*', match.group(1)):
            name = name.strip()
            if name:
                results.append(name)

    return results


class ArtistVocab:
    # Builds a vocabulary from a list of Track ORM objects.
    # Index 0 is reserved for OOV/unknown; real artists start at 1.
    def __init__(self, tracks):
        names = sorted(set(t.artist.strip().lower() for t in tracks if t.artist))
        self._vocab = {name: i + 1 for i, name in enumerate(names)}

    # Encodes a single artist name to an index (0 = OOV).
    def encode(self, name: str) -> int:
        if not name:
            return 0
        return self._vocab.get(name.strip().lower(), 0)

    # Returns vocab size including the OOV slot (index 0).
    def size(self) -> int:
        return len(self._vocab) + 1

    # Encodes primary artist plus any featured/remix artists from title.
    # Returns a deduplicated list of indices (at least [0] if nothing encodes).
    def encode_all(self, primary: str, title: str) -> list:
        indices = []
        seen = set()
        for name in [primary] + parse_featured_artists(title):
            idx = self.encode(name)
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        return indices if indices else [0]

    # Returns a DataLoader collate_fn that pads variable-length artist index lists.
    # Module-level function (not a closure) so it is picklable for DataLoader workers on Windows (spawn).
    def make_collate_fn(self):
        return pad_collate


# DataLoader collate_fn that pads variable-length artist index lists.
# Batch items: (stacked_mels, artist_indices_list, bpm, key_feat, panns_feat, label).
# Output batch: (mels, padded_artist_idx, artist_mask, bpms, key_feats, panns_feats, labels).
def pad_collate(batch):
    mels, artist_lists, bpms, key_feats, panns_feats, labels = zip(*batch)
    max_len = max(len(a) for a in artist_lists)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, a in enumerate(artist_lists):
        padded[i, :len(a)] = torch.tensor(a, dtype=torch.long)
        mask[i, :len(a)] = True
    return (
        torch.stack(mels),
        padded,
        mask,
        torch.stack(bpms),
        torch.stack(key_feats),
        torch.stack(panns_feats),
        torch.stack(labels),
    )
