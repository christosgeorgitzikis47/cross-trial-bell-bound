"""
ΕΛΕΓΧΟΣ ΕΣΩΤΕΡΙΚΩΝ ΠΑΡΑΠΟΜΠΩΝ ΤΟΥ PAPER

Κάθε «§N», «§N.M», «Figure N», «Table N», «Appendix X», «Limitation N» και
«[n]» πρέπει να δείχνει σε κάτι που υπάρχει. Ο έλεγχος γίνεται με script και
όχι με το μάτι, επειδή οι αναριθμήσεις (6.4 -> 6.5, §7.1 -> Limitation 1)
είναι ακριβώς εκεί που ξεφεύγουν οι σπασμένες παραπομπές.

Χρήση:  python3 code/check_refs.py [paper/paper_en.md ...]
"""
import re, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = [os.path.join(os.path.dirname(HERE), "paper", "paper_en.md")]


def check(path):
    s = open(path, encoding="utf-8").read()
    body_end = s.index("## References") if "## References" in s else \
        s.index("## Αναφορές")
    body = s[:body_end]
    bad = []

    # --- τι υπάρχει ---
    secs = set(re.findall(r"^## (\d+)\. ", body, re.M))
    subs = set(re.findall(r"^### (\d+\.\d+) ", body, re.M))
    figs = set(re.findall(r"^\*\*(?:Figure|Σχήμα) (\d+)\.\*\*", body, re.M))
    tabs = {str(i + 1) for i in range(len(re.findall(r"^: ", body, re.M)))}
    apps = set(re.findall(r"^## (?:Appendix ([A-Z])|Παράρτημα ([ΑΒΓΔ])) ",
                          body, re.M))
    apps = {a or b for a, b in apps} if apps and isinstance(
        next(iter(apps)), tuple) else apps
    lims = re.search(r"^## \d+\. (?:Limitations|Περιορισμοί)\n(.*?)^---",
                     body, re.M | re.S)
    n_lim = len(re.findall(r"^\d+\. ", lims.group(1), re.M)) if lims else 0
    refs = set(re.findall(r"^\[(\d+)\]", s[body_end:], re.M))

    # --- τι παραπέμπεται ---
    for m in re.finditer(r"§(\d+)\.(\d+)", body):
        if f"{m.group(1)}.{m.group(2)}" not in subs:
            bad.append(f"§{m.group(1)}.{m.group(2)} — δεν υπάρχει υποενότητα")
    for m in re.finditer(r"§(\d+)(?!\.\d)", body):
        if m.group(1) not in secs:
            bad.append(f"§{m.group(1)} — δεν υπάρχει ενότητα")
    for m in re.finditer(r"\*\*(?:Figure|Σχήμα) (\d+)\*\*|"
                         r"(?:Figure|Σχήμα) (\d+)", body):
        n = m.group(1) or m.group(2)
        if n not in figs:
            bad.append(f"Figure {n} — δεν υπάρχει λεζάντα")
    for m in re.finditer(r"(?:Table|Πίνακα[ς]?) (\d+)", body):
        if m.group(1) not in tabs:
            bad.append(f"Table {m.group(1)} — δεν υπάρχει τόσος πίνακας")
    for m in re.finditer(r"(?:Appendix ([A-Z])|Παράρτημα ([ΑΒΓΔ]))(?![α-ωa-z])",
                         body):
        g = m.group(1) or m.group(2)
        if g not in apps:
            bad.append(f"Appendix/Παράρτημα {g} — δεν υπάρχει")
    for m in re.finditer(r"(?:Limitation|Περιορισμ[όό]) (\d+)", body):
        if not (1 <= int(m.group(1)) <= n_lim):
            bad.append(f"Limitation {m.group(1)} — υπάρχουν {n_lim}")
    cited = {n for grp in re.findall(r"\[([\d,\s]+)\]", body)
             for n in grp.split(",") if n.strip().isdigit()}
    cited = {c.strip() for c in cited}
    for c in sorted(cited - refs, key=int):
        if int(c) > 1:            # τα {0,1} σύνολα δεν είναι παραπομπές
            bad.append(f"[{c}] — παραπομπή χωρίς εγγραφή")
    orphan = sorted(refs - cited, key=int)

    print(f"--- {os.path.basename(path)} ---")
    print(f"  ενότητες {sorted(secs, key=int)}")
    print(f"  υποενότητες {sorted(subs)}")
    print(f"  σχήματα {sorted(figs, key=int)} · πίνακες {len(tabs)} · "
          f"παραρτήματα {sorted(apps)} · περιορισμοί {n_lim} · "
          f"αναφορές {len(refs)}")
    print(f"  ορφανές αναφορές (στη λίστα, χωρίς παραπομπή): "
          f"{orphan if orphan else 'καμία'}")
    if bad:
        print("  ΣΠΑΣΜΕΝΕΣ ΠΑΡΑΠΟΜΠΕΣ:")
        for b in sorted(set(bad)):
            print("   ", b)
    else:
        print("  ΚΑΜΙΑ ΣΠΑΣΜΕΝΗ ΠΑΡΑΠΟΜΠΗ")
    return len(bad)


if __name__ == "__main__":
    paths = sys.argv[1:] or DEFAULT
    sys.exit(min(1, sum(check(p) for p in paths)))
