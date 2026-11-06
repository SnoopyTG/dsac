#!/usr/bin/env python3
"""Print one comparison table from every runs/*/metrics.json (single-split or CV)."""
import glob, json, os, sys
runs = sys.argv[1] if len(sys.argv) > 1 else "runs"
rows = []
for f in sorted(glob.glob(os.path.join(runs, "*", "metrics.json"))):
    d = json.load(open(f)); name = os.path.basename(os.path.dirname(f))
    if d["mode"] == "cv":
        s = d["summary"]; g = lambda k: f"{s[k][0]:.3f}+/-{s[k][1]:.3f}" if k in s else "  -  "
        rows.append((name, "CV", g("auc"), g("f1"), g("false_alarms_per_hour"), g("pearson_p_vs_kss")))
    else:
        m = d["metrics"]; g = lambda k: f"{m[k]:.3f}" if k in m else "  -  "
        rows.append((name, "split", g("auc"), g("f1"), g("false_alarms_per_hour"), g("pearson_p_vs_kss")))
hdr = ("run", "eval", "AUC", "F1", "FalseAlarms/h", "corr(P,KSS)")
w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(6)]
for r in [hdr] + rows:
    print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
