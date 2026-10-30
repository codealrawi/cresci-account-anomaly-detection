"""Tests of the tweet reader and feature extraction (synthetic records only; no real tweet text).
Run from the repository root: python -I -m pytest tests -q -p no:cacheprovider
"""
import io, os, sys, zipfile, re, tempfile
import numpy as np, pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import official_reader as orr
import extract_features_v2 as v2
import extract_features as orig          # frozen original (read-only import)

NF = 26
UNKNOWN = (4, 9, 10, 12, 16, 17, 18, 24, 25)


def esc(v):
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\\n")


def rec(uid="1001", text="hello, world", created="Fri May 01 00:18:11 +0000 2015", nu="0", rt=None, tid="500", extra=None):
    f = ['"%s"' % tid, '"%s"' % esc(text), '"src"', uid, '""', "0", "0", '""', ("\\N" if rt is None else rt), "\\N", "\\N", '""', '""', "0", "0", "0", '""', '""',
         "\\N", "0", nu, "0", '"%s"' % created, '"2015-05-01 02:18:11"', '"2015-05-01 12:57:19"', '"2015-05-01 12:57:19"']
    assert len(f) == NF
    for k, v in (extra or {}).items(): f[k] = v
    return ",".join(f) + "\n"


def read(data, kind="escaped", chunk=400_000):
    st = {}
    frames = list(orr.iter_chunks(io.BytesIO(data.encode("utf-8")), kind, chunk_size=chunk, stats=st))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=orr.COLS)
    return df, st


def test_typical_record_positions():
    df, st = read(rec(uid="777", text="plain", nu="2", rt="9001", created="Sat May 02 10:00:00 +0000 2015"))
    assert st == {"records": 1, "malformed": 0}
    r = df.iloc[0]
    assert (r.user_id, r.text, r.num_urls, r.retweeted_status_id, r.created_at) == ("777", "plain", "2", "9001", "Sat May 02 10:00:00 +0000 2015")


def test_commas_quotes_newlines_backslashes():
    t = 'a, b "quoted" \\ back\nsecond line, end'
    df, st = read(rec(text=t) + rec(uid="2", text="x"))
    assert st["records"] == 2 and st["malformed"] == 0 and df.text.iloc[0] == t and df.user_id.iloc[1] == "2"


def test_escaped_nonl_removes_only_line_breaks():
    t = "line1\nline2\r\nline3, \"q\" \\ end"
    df, st = read(rec(text=t) + rec(text="NA"), "escaped_nonl")
    assert df.text.iloc[0] == "line1line2line3, \"q\" \\ end" and pd.isna(df.text.iloc[1]) and st == {"records": 2, "malformed": 0}
    keep, _ = read(rec(text=t), "escaped")
    assert keep.text.iloc[0] == t                                         # plain "escaped" keeps line breaks (faithful official text)
    other = ["user_id", "created_at", "num_urls", "retweeted_status_id"]
    pd.testing.assert_frame_equal(df.iloc[:1][other].reset_index(drop=True), keep[other].reset_index(drop=True))


def test_wrong_field_counts_are_counted_not_used():
    ok = rec(); short = ",".join(ok.strip().split(",")[:25]) + "\n"; long = ok.strip() + ',"x"\n'
    df, st = read(ok + short + long + ok)
    assert st == {"records": 4, "malformed": 2} and len(df) == 2


def test_missing_values():
    df, st = read(rec(text="", rt=None, nu="0"))
    assert pd.isna(df.text.iloc[0]) and pd.isna(df.retweeted_status_id.iloc[0])           # "" and \N -> NaN
    df, _ = read(rec(text="NA") + rec(text="None") + rec(text="NULL"))                      # pandas default NA tokens are emulated
    assert df.text.isna().all()
    df, _ = read(rec(text="N/A ok"))
    assert df.text.iloc[0] == "N/A ok"


def test_null_marker_is_not_confused_with_the_letter_N():
    # unquoted \N is NULL; a quoted text "N" is the letter N; a literal "\N" inside text (written \\N) stays text
    df, _ = read(rec(text="N", rt=None) + rec(text="a,\\N,b", rt="55") + rec(text="\\N", rt="0"))
    assert df.text.iloc[0] == "N" and pd.isna(df.retweeted_status_id.iloc[0])
    assert df.text.iloc[1] == "a,\\N,b" and df.retweeted_status_id.iloc[1] == "55"
    assert df.text.iloc[2] == "\\N" and df.retweeted_status_id.iloc[2] == "0"
    df, _ = read(rec(rt="\\N", nu="\\N"))                                   # NULL in other used fields -> NaN, not the letter N
    assert pd.isna(df.retweeted_status_id.iloc[0]) and pd.isna(df.num_urls.iloc[0])


def test_end_of_file_cases():
    df, st = read(rec().rstrip("\n"))                                    # no trailing newline
    assert st == {"records": 1, "malformed": 0}
    df, st = read(rec() + '"123","partial text')                         # cut inside a record
    assert st["malformed"] == 1 and len(df) == 1
    df, st = read(rec() + "Fatal error: Maximum execution time of 300 seconds exceeded in x on line 1\n")
    assert st == {"records": 2, "malformed": 1} and len(df) == 1


def test_unknown_fields_are_not_used():
    base, _ = read(rec())
    changed, _ = read(rec(extra={k: '"zzz"' for k in UNKNOWN}))
    pd.testing.assert_frame_equal(base, changed)


def test_control_counts_synthetic():
    n = 1234; data = "".join(rec(uid=str(i % 50), tid=str(i)) for i in range(n)) + ",".join(["a"] * 7) + "\n"
    df, st = read(data)
    assert st == {"records": n + 1, "malformed": 1} and len(df) == n and df.user_id.nunique() == 50


def test_deterministic_and_chunk_independent():
    data = "".join(rec(uid=str(i % 5), text="t%d, \"q\"" % i, tid=str(i)) for i in range(101))
    a, _ = read(data); b, _ = read(data); c, _ = read(data, chunk=7)
    pd.testing.assert_frame_equal(a, b); pd.testing.assert_frame_equal(a, c)


STD = ('"id","text","user_id","num_urls","retweeted_status_id","created_at","other"\n'
       '1,"he said ""hi"", ok",10,0,,"Fri May 01 00:18:11 +0000 2015","x"\n'
       '2,"multi\nline",10,1,NULL,"Fri May 01 01:18:11 +0000 2015","x"\n'
       '3,NA,11,0,77,"Fri May 01 02:18:11 +0000 2015","x"\n'
       '4,"",11,0,0,"Fri May 01 03:18:11 +0000 2015","x"\n')


def test_std_provider_equals_pandas():
    got, st = read(STD, "std")
    exp = pd.read_csv(io.StringIO(STD), usecols=orr.COLS, dtype=str, na_values=["NULL"])[orr.COLS]
    pd.testing.assert_frame_equal(got.reset_index(drop=True), exp.reset_index(drop=True), check_dtype=False)
    assert st == {"records": 4, "malformed": 0}


def _mini_archive(tmp, escaped):
    rows = []
    for i in range(30):
        u = "10" if i < 12 else ("11" if i < 25 else "12")                 # user 12 has 5 tweets, all three have >= 2
        rows.append((str(1000 + i), "Hello World %d, ok. HELLO WORLD %d http://t.co/x" % (i % 4, i % 4), u,
                     "Sat May %02d %02d:%02d:00 +0000 2015" % (1 + i % 5, (i * 7) % 24, (i * 13) % 60), str(i % 2), ("900%d" % i) if i % 3 == 0 else None))
    if escaped:
        tw = "".join(rec(uid=u, text=t, created=c, nu=nu, rt=rt, tid=i) for (i, t, u, c, nu, rt) in rows)
    else:
        tw = '"id","text","user_id","num_urls","retweeted_status_id","created_at"\n' + "".join(
            '%s,"%s",%s,%s,%s,"%s"\n' % (i, t.replace('"', '""'), u, nu, ("" if rt is None else rt), c) for (i, t, u, c, nu, rt) in rows)
    users = '"id","favourites_count","created_at","crawled_at","statuses_count"\n10,5,"Tue Jun 11 11:20:35 +0000 2013","2015-05-06 06:41:46",100\n11,0,"Tue Jun 11 11:20:35 +0000 2013","2015-05-06 06:41:46",50\n12,9,"Tue Jun 11 11:20:35 +0000 2013","2015-05-06 06:41:46",5\n13,9,"Tue Jun 11 11:20:35 +0000 2013","2015-05-06 06:41:46",5\n'
    name = "genuine_accounts"
    with zipfile.ZipFile(os.path.join(tmp, name + ".csv.zip"), "w") as z:
        z.writestr(name + ".csv/users.csv", users); z.writestr(name + ".csv/tweets.csv", tw)
    return name


def test_extraction_equivalence_across_providers():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        n = _mini_archive(a, escaped=False); _mini_archive(b, escaped=True)
        ref = orig.process_group(a, n, 0)
        p = v2.process_group(a, n, 0, "pandas"); s = v2.process_group(a, n, 0, "std"); e = v2.process_group(b, n, 0, "escaped")
        for other in (p, s, e):
            pd.testing.assert_frame_equal(ref.reset_index(drop=True), other.reset_index(drop=True))
        assert list(ref.user_id) == ["10", "11", "12"]                 # user 13 has no tweets -> excluded by the frozen >= 2 rule
        again = v2.process_group(b, n, 0, "escaped")                   # determinism
        pd.testing.assert_frame_equal(e, again)


def test_feature_logic_is_verbatim_copy_of_frozen_extraction():
    SRC = os.path.join(HERE, "..", "src")
    o = open(os.path.join(SRC, "extract_features.py"), encoding="utf-8").read().splitlines()
    v = [l.strip() for l in open(os.path.join(SRC, "extract_features_v2.py"), encoding="utf-8").read().splitlines()]
    a0 = next(i for i, l in enumerate(o) if l.strip().startswith("ts = pd.to_datetime(ch"))
    a1 = next(i for i, l in enumerate(o) if l.strip().startswith("rows = []")) + 0
    b0 = next(i for i, l in enumerate(o) if l.strip().startswith("for _, u in users.iterrows()"))
    b1 = next(i for i, l in enumerate(o) if l.strip().startswith("print(name"))
    for seg in (o[a0:a1], o[b0:b1]):
        pos = 0
        for line in seg:
            l = line.strip()
            if not l: continue
            assert l in v[pos:], "line missing/out of order in v2: " + l[:60]
            pos = v.index(l, pos) + 1
