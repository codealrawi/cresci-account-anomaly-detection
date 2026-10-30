"""Stage 35 reading adapter (see 02_data_contract.md, addendum_01.md).

Streams tweet records from a binary CSV stream and yields pandas blocks with exactly the five columns used by the frozen feature extraction:
user_id, text, created_at, num_urls, retweeted_status_id   (all str, missing = NaN).

kind = "escaped": official genuine tweets.csv - no header, 26 fields, backslash escapes, \\N for NULL, fixed positions (1, 3, 22, 20, 8).
kind = "std"    : header-based, standard CSV (module csv) - used to validate that csv-module reading reproduces the frozen pandas reading.
Only aggregate statistics are kept (no text, no ids): stats = {"records", "malformed"}.
"""
import io, csv, re
import numpy as np
import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES      # the default NA tokens that pandas.read_csv applied in the frozen extraction

csv.field_size_limit(1 << 28)
COLS = ["user_id", "text", "created_at", "num_urls", "retweeted_status_id"]
OFFICIAL_NFIELDS = 26
OFFICIAL_POS = {"text": 1, "user_id": 3, "retweeted_status_id": 8, "num_urls": 20, "created_at": 22}   # data contract §3; positions 4, 9, 10, 12, 16-18, 24-25 are never used
SENT = "\x01NULL\x01"                                 # replaces an UNQUOTED \N before csv parsing (csv.reader with escapechar would turn \N into the letter "N")
NULL_RE = re.compile(r"(?<![^,])\\N(?=,|[\r\n]|$)")  # \N alone in a field: preceded by line start/comma, followed by comma/EOL.  A literal backslash inside quoted text is written \\ so it cannot match
NA_TOKENS = set(STR_NA_VALUES) | {"NULL", SENT}


def _cell(v):
    return np.nan if v in NA_TOKENS else v


def iter_chunks(binary_stream, kind, chunk_size=400_000, stats=None):
    stats = stats if stats is not None else {}
    stats.setdefault("records", 0); stats.setdefault("malformed", 0)
    text = io.TextIOWrapper(binary_stream, encoding="utf-8", errors="replace", newline="")
    # kind "escaped_nonl" = "escaped" + removal of CR/LF characters from the tweet text.  Reason (stage 35 diagnostics): the official genuine file keeps
    # embedded line breaks, while every other file of the archive (positives, and the mirror's genuine file) has none; removal harmonises the two classes.
    strip_nl = kind == "escaped_nonl"
    if strip_nl:
        kind = "escaped"
    if kind == "escaped":
        rd = csv.reader((NULL_RE.sub(SENT, ln) for ln in text), quotechar='"', escapechar="\\", doublequote=False)
        nfields, pos = OFFICIAL_NFIELDS, OFFICIAL_POS
    elif kind == "std":
        rd = csv.reader(text)
        header = next(rd)
        nfields = len(header)
        pos = {c: header.index(c) for c in COLS}
    else:
        raise ValueError(kind)
    idx = [pos[c] for c in COLS]
    buf = []
    for row in rd:
        stats["records"] += 1
        if len(row) != nfields:                       # wrong number of fields: counted, never used
            stats["malformed"] += 1
            continue
        if kind == "escaped" and not row[0].isdigit():  # id must be an integer
            stats["malformed"] += 1
            continue
        cells = [_cell(row[i]) for i in idx]
        if strip_nl and isinstance(cells[1], str):                # cells[1] = text; NA tokens are decided BEFORE removal (as in the original parsing)
            cells[1] = cells[1].replace("\r", "").replace("\n", "")
        buf.append(cells)
        if len(buf) >= chunk_size:
            yield pd.DataFrame(buf, columns=COLS, dtype=object); buf = []
    if buf:
        yield pd.DataFrame(buf, columns=COLS, dtype=object)
