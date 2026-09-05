"""
CHECK FOR ORPHANED TABLE HEADERS IN THE PDF

When a longtable starts too low on a page, LaTeX prints the header, breaks,
and prints it again on the next page -- so it appears once above the caption
and once inside the table. By eye you read straight past it.

The check: for every "Table N:", if the words of the line IMMEDIATELY AFTER
the caption (the header) also appear BEFORE the caption on the same page,
there is an orphaned header.

Usage:  python3 code/check_tables.py paper/georgitzikis_2026_cross_trial_bell_bound.pdf
"""
import re, subprocess, sys, os


def check(pdf):
    txt = subprocess.run(["pdftotext", "-layout", pdf, "-"],
                         capture_output=True, text=True,
                         errors="replace").stdout
    bad = []
    n_tab = 0
    for pno, page in enumerate(txt.split("\f"), 1):
        lines = [l.rstrip() for l in page.split("\n")]
        for i, l in enumerate(lines):
            m = re.match(r"\s*Table (\d+):", l)
            if not m:
                continue
            n_tab += 1
            # the header: the first non-empty line after the caption (the
            # caption may span two lines)
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or
                                      len(lines[j].split()) < 2 or
                                      lines[j].strip().endswith(".")):
                j += 1
            if j >= len(lines):
                continue
            head = set(lines[j].split())
            if len(head) < 2:
                continue
            before = " ".join(lines[max(0, i - 6):i])
            overlap = head & set(before.split())
            if len(overlap) >= max(2, len(head) - 1):
                bad.append((pno, m.group(1), lines[j].strip()[:60]))
    print(f"--- {os.path.basename(pdf)}: {n_tab} tables ---")
    if bad:
        for pno, n, h in bad:
            print(f"  ORPHANED HEADER: page {pno}, Table {n}: \"{h}\"")
    else:
        print("  NO ORPHANED HEADERS")
    return len(bad)


if __name__ == "__main__":
    paths = sys.argv[1:] or ["paper/georgitzikis_2026_cross_trial_bell_bound.pdf"]
    sys.exit(min(1, sum(check(p) for p in paths)))
