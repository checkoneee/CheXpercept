"""Auto-fill the doctor `optimal?` column with 'y' for sample_test.sh.

Production runs require real annotations; this exists only so the
sample test can run unattended.
"""
import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--value", default="y")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.csv_path)))
    if not rows:
        print(f"empty: {args.csv_path}")
        return

    optimal_col = "optimal?" if "optimal?" in rows[0] else "optimal"
    for row in rows:
        row[optimal_col] = args.value

    with open(args.csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"auto-filled {len(rows)} rows in {args.csv_path}")


if __name__ == "__main__":
    main()
