"""
ΕΛΕΓΧΟΣ ΟΡΦΑΝΩΝ ΚΕΦΑΛΙΔΩΝ ΠΙΝΑΚΑ ΣΤΟ PDF

Το longtable, όταν ο πίνακας αρχίζει πολύ χαμηλά στη σελίδα, τυπώνει την
κεφαλίδα, σπάει, και την ξανατυπώνει στην επόμενη — οπότε φαίνεται μία φορά
πάνω από τη λεζάντα και μία μέσα στον πίνακα. Με το μάτι το προσπερνάς.

Ο έλεγχος: για κάθε «Table N:», αν οι λέξεις της γραμμής ΑΜΕΣΩΣ ΜΕΤΑ τη λεζάντα
(η κεφαλίδα) εμφανίζονται και ΠΡΙΝ από τη λεζάντα στην ίδια σελίδα, υπάρχει
ορφανή κεφαλίδα.

Χρήση:  python3 code/check_tables.py paper/paper_en.pdf
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
            # η κεφαλίδα: η πρώτη μη κενή γραμμή μετά τη λεζάντα (η λεζάντα
            # μπορεί να πιάνει δύο γραμμές)
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
    print(f"--- {os.path.basename(pdf)}: {n_tab} πίνακες ---")
    if bad:
        for pno, n, h in bad:
            print(f"  ΟΡΦΑΝΗ ΚΕΦΑΛΙΔΑ: σελ {pno}, Table {n}: «{h}»")
    else:
        print("  ΚΑΜΙΑ ΟΡΦΑΝΗ ΚΕΦΑΛΙΔΑ")
    return len(bad)


if __name__ == "__main__":
    paths = sys.argv[1:] or ["paper/paper_en.pdf"]
    sys.exit(min(1, sum(check(p) for p in paths)))
