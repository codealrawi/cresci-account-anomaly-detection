"""Per-account behavioural features from Cresci-2017 (streamed from the zip files).

Usage: python -I extract_features.py <datasets_full.csv dir> <out.csv>

Features (our own definitions; the article's original feature code/matrix is not available):
  p1 posts_per_day        tweets in corpus / observed span (days, >=1)
  p2 likes_per_day        profile favourites_count / account age at crawl (days, >=1)
  p3 retweet_share        share of tweets that are retweets (replaces article's medical-lexicon share,
                          which cannot be computed: no dictionary, multilingual tweets)
  p4 url_share            share of tweets with >=1 URL
  p5 hour_entropy_bits    Shannon entropy (bits) of the posting-hour histogram (UTC)
  p6 mean_len_chars       mean tweet length (first TEXT_CAP tweets seen per account)
  p7 upper_share          uppercase share among letters (first TEXT_CAP tweets)
  p8 interval_var_h2      variance of inter-post intervals in hours^2 (all tweets)
  p9 repeat_ngram_share   share of repeated word bi/trigram occurrences (first TEXT_CAP tweets)
Label: 0 genuine; 1 social_spambots_1-3 + fake_followers (as in the article). Traditional spambots are
excluded (as in the article; tweets of #2-#4 are not provided anyway).
"""
import sys, zipfile, io, re, math, collections
import numpy as np
import pandas as pd

TEXT_CAP = 100
GROUPS = [("genuine_accounts", 0), ("social_spambots_1", 1), ("social_spambots_2", 1),
          ("social_spambots_3", 1), ("fake_followers", 1)]
WORD = re.compile(r"\w+", re.UNICODE)


def process_group(base, name, label):
    z = zipfile.ZipFile(f"{base}/{name}.csv.zip")
    with z.open(f"{name}.csv/users.csv") as f:
        users = pd.read_csv(f, usecols=lambda c: c in {"id", "favourites_count", "created_at", "crawled_at", "statuses_count"},
                            dtype=str, na_values=["NULL"], encoding_errors="replace")
    users["id"] = users["id"].astype(str)
    if "crawled_at" not in users.columns: users["crawled_at"] = np.nan  # filled with last tweet time of the group below
    st = {}  # user_id -> state
    gmax = 0
    cols = ["user_id", "text", "created_at", "num_urls", "retweeted_status_id"]
    with z.open(f"{name}.csv/tweets.csv") as f:
        for ch in pd.read_csv(f, usecols=cols, dtype=str, chunksize=400_000, na_values=["NULL"], encoding_errors="replace"):
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
    print(name, "users", len(users), "with>=2 tweets", len(rows), flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    base, out = sys.argv[1], sys.argv[2]
    df = pd.concat([process_group(base, n, l) for n, l in GROUPS], ignore_index=True)
    df.to_csv(out, index=False)
    print(df.groupby("group").size())
    print("NaN per column:\n", df.isna().sum())
