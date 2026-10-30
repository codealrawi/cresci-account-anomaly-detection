"""Stage 35: copy of the FROZEN feature logic of pipeline/extract_features.py (SHA-256 09B06980...D85BBDE) with a pluggable tweet-block provider.
The accumulation and the p1-p9 formulas are verbatim; only the source of the tweet blocks differs.  Do not edit the formulas.
providers: pandas (as in the original), std / escaped (official_reader.py).
Usage: python -I extract_features_v2.py <datasets_full.csv dir> <out.csv> <genuine_provider> <other_provider> [stats.json]
Aggregates only are printed/stored.
"""
import sys, os, zipfile, re, collections, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import official_reader as orr

TEXT_CAP = 100
GROUPS = [("genuine_accounts", 0), ("social_spambots_1", 1), ("social_spambots_2", 1),
          ("social_spambots_3", 1), ("fake_followers", 1)]
WORD = re.compile(r"\w+", re.UNICODE)


def blocks(z, name, provider, stats):
    member = f"{name}.csv/tweets.csv"
    if provider == "pandas":
        cols = ["user_id", "text", "created_at", "num_urls", "retweeted_status_id"]
        with z.open(member) as f:
            yield from pd.read_csv(f, usecols=cols, dtype=str, chunksize=400_000, na_values=["NULL"], encoding_errors="replace")
    else:
        with z.open(member) as f:
            yield from orr.iter_chunks(f, provider, stats=stats)


def process_group(base, name, label, provider="pandas", stats=None):
    stats = stats if stats is not None else {}
    z = zipfile.ZipFile(f"{base}/{name}.csv.zip")
    with z.open(f"{name}.csv/users.csv") as f:
        users = pd.read_csv(f, usecols=lambda c: c in {"id", "favourites_count", "created_at", "crawled_at", "statuses_count"},
                            dtype=str, na_values=["NULL"], encoding_errors="replace")
    users["id"] = users["id"].astype(str)
    if "crawled_at" not in users.columns: users["crawled_at"] = np.nan  # filled with last tweet time of the group below
    st = {}  # user_id -> state
    gmax = 0
    for ch in blocks(z, name, provider, stats):
        ts = pd.to_datetime(ch["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
        sec = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().fillna(-1).astype("int64")
        ch = ch.assign(_ts=sec)
        for uid, text, t, nu, rt, tsec in zip(ch["user_id"], ch["text"].fillna(""), ch["created_at"],
                                              ch["num_urls"], ch["retweeted_status_id"], ch["_ts"]):
            s = st.get(uid)
            if s is None:
                s = st[uid] = dict(n=0, url=0, rt=0, hours=np.zeros(24), ts=[], seen=0, len=0, up=0, let=0,
                                   ng=collections.Counter(), ngn=0)
            s["n"] += 1
            try:
                if isinstance(nu, str) and float(nu) > 0: s["url"] += 1
            except ValueError:
                pass
            if isinstance(rt, str) and rt not in ("0", ""): s["rt"] += 1
            if tsec > gmax: gmax = tsec
            if tsec >= 0:
                s["ts"].append(tsec); s["hours"][(tsec // 3600) % 24] += 1
            if s["seen"] < TEXT_CAP:
                s["seen"] += 1; s["len"] += len(text)
                s["up"] += sum(c.isupper() for c in text); s["let"] += sum(c.isalpha() for c in text)
                w = [x.lower() for x in WORD.findall(text)]
                for k in (2, 3):
                    for i in range(len(w) - k + 1):
                        s["ng"][tuple(w[i:i + k])] += 1; s["ngn"] += 1
    rows = []
    for _, u in users.iterrows():
        s = st.get(u["id"])
        if s is None or s["n"] < 2:
            continue
        ts = np.sort(np.array(s["ts"], dtype=np.int64))
        span = max(1.0, (ts[-1] - ts[0]) / 86400.0) if len(ts) > 1 else 1.0
        dt = np.diff(ts) / 3600.0
        h = s["hours"] / max(1.0, s["hours"].sum())
        ent = float(-(h[h > 0] * np.log2(h[h > 0])).sum())
        try:
            end = pd.to_datetime(u["crawled_at"], utc=True) if pd.notna(u["crawled_at"]) else pd.to_datetime(gmax, unit="s", utc=True)
            age = max(1.0, (end - pd.to_datetime(u["created_at"], format="%a %b %d %H:%M:%S %z %Y", utc=True)).total_seconds() / 86400.0)
        except Exception:
            age = np.nan
        fav = float(u["favourites_count"]) if pd.notna(u["favourites_count"]) else np.nan
        rep = sum(c - 1 for c in s["ng"].values() if c > 1) / s["ngn"] if s["ngn"] else 0.0
        rows.append(dict(user_id=u["id"], group=name, label=label, n_tweets=s["n"],
                         p1=s["n"] / span, p2=fav / age if age == age else np.nan, p3=s["rt"] / s["n"],
                         p4=s["url"] / s["n"], p5=ent, p6=s["len"] / s["seen"],
                         p7=s["up"] / s["let"] if s["let"] else 0.0,
                         p8=float(np.var(dt)) if len(dt) else 0.0, p9=rep))
    stats.update(group=name, users=len(users), accounts_ge2=len(rows), accounts_with_tweets=len(st))
    print(name, "users", len(users), "with>=2 tweets", len(rows), "reader_stats", {k: v for k, v in stats.items() if k in ("records", "malformed")}, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    base, out, pg, po = sys.argv[1:5]
    sj = sys.argv[5] if len(sys.argv) > 5 else None
    allstats = {}; frames = []
    for n, l in GROUPS:
        st_ = {}; frames.append(process_group(base, n, l, pg if n == "genuine_accounts" else po, st_)); allstats[n] = st_
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)
    if sj: json.dump(allstats, open(sj, "w"), indent=1)
    print(df.groupby("group").size())
