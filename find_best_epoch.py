"""Find the best epoch across custom_* logs (max Fmax in 'Averaged stats(Valid)').

Prints the best epoch's Overall Fmax/AUPR and the 4 ALF frequency-bucket metrics.
"""
import glob
import os
import re

LOG_DIR = os.path.expanduser("~/project/log")

averaged_re = re.compile(r"Averaged stats\(Valid\):.*?Fmax:\s*([\d.]+),\s*micro AUPRC:\s*([\d.]+)")
epoch_re = re.compile(r"Valid Epoch:\s*\[(\d+)\]")
alf_re = re.compile(r"ALF\s*(\[[^\]]+\]|\([^)]+\]):\s*n=(\d+),\s*Fmax=([\d.]+),\s*micro_AUPRC=([\d.]+)")
overall_re = re.compile(r"Overall:\s*Fmax=([\d.]+),\s*micro_AUPRC=([\d.]+)")


def parse_log(path):
    """Return list of dicts per epoch: {epoch, fmax, aupr, alf: [...], overall: (...)}."""
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    records = []
    current_epoch = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = epoch_re.search(line)
        if m:
            current_epoch = int(m.group(1))
        m = averaged_re.search(line)
        if m and current_epoch is not None:
            rec = {
                "epoch": current_epoch,
                "fmax": float(m.group(1)),
                "aupr": float(m.group(2)),
                "alf": [],
                "overall": None,
            }
            # Scan forward for the ALF evaluation block belonging to this epoch
            j = i + 1
            while j < len(lines):
                l2 = lines[j]
                if "Averaged stats(Valid)" in l2 or epoch_re.search(l2):
                    break
                am = alf_re.search(l2)
                if am:
                    rec["alf"].append({
                        "bucket": am.group(1),
                        "n": int(am.group(2)),
                        "fmax": float(am.group(3)),
                        "aupr": float(am.group(4)),
                    })
                om = overall_re.search(l2)
                if om:
                    rec["overall"] = (float(om.group(1)), float(om.group(2)))
                j += 1
            records.append(rec)
        i += 1
    return records


def main():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "custom*.log")))
    if not files:
        print(f"No custom*.log files found in {LOG_DIR}")
        return

    best = None  # (fmax, file, record)
    for path in files:
        records = parse_log(path)
        fname = os.path.basename(path)
        if not records:
            print(f"{fname}: (no valid epochs found)")
            continue
        rec = max(records, key=lambda r: r["fmax"])
        if best is None or rec["fmax"] > best[0]:
            best = (rec["fmax"], fname, rec)

        print("=" * 70)
        print(f"{fname}  |  best epoch = {rec['epoch']}")
        if rec["overall"]:
            print(f"  Overall: Fmax={rec['overall'][0]:.4f}, AUPR(micro_AUPRC)={rec['overall'][1]:.4f}")
        else:
            print("  Overall: (not found in log)")
        print("  ALF frequency buckets:")
        for b in rec["alf"]:
            print(f"    ALF {b['bucket']:<12} n={b['n']:<6} Fmax={b['fmax']:.4f}, AUPR(micro_AUPRC)={b['aupr']:.4f}")

    if best is None:
        print("No 'Averaged stats(Valid)' entries found.")
        return

    print("=" * 70)
    print(f"GLOBAL BEST: file={best[1]}, epoch={best[2]['epoch']}, Valid Fmax={best[0]:.4f}")


if __name__ == "__main__":
    main()
