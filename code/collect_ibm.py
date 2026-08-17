"""
Συλλογή ΩΜΩΝ κβαντικών μετρήσεων ανά shot από επεξεργαστή IBM Quantum.

Το κρίσιμο: ζητάμε τα αποτελέσματα ΑΝΑ SHOT, με τη σειρά που έγιναν.
Όχι συγκεντρωτικά ("counts"). Η σειρά είναι όλο το νόημα του πειράματος.

ΤΡΕΞΕ ΤΟ ΣΤΟΝ ΥΠΟΛΟΓΙΣΤΗ ΣΟΥ, όχι σε online περιβάλλον.

Πριν τρέξεις:
    pip install qiskit qiskit-ibm-runtime numpy

Το token ΔΕΝ γράφεται μέσα στο αρχείο. Το βάζεις στο terminal:
    export IBM_QUANTUM_TOKEN="to_token_sou_edw"        (Mac / Linux)

Μετά:
    python3 collect_ibm.py --sim          # δοκιμή τοπικά, χωρίς hardware
    python3 collect_ibm.py                # πραγματικός επεξεργαστής
"""

import os, sys, argparse, json
from datetime import datetime, timezone
import numpy as np
from qiskit import QuantumCircuit, transpile


def bell_circuit():
    """Κατάσταση Bell (Φ+): δύο qubit μέγιστα εμπλεγμένα.
    Κάθε shot δίνει 00 ή 11, με 50/50 — τα bits που θέλουμε να ελέγξουμε."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def run_simulator(shots):
    """Τοπική προσομοίωση. Χρήσιμη για να επιβεβαιώσεις ότι ο κώδικας δουλεύει
    ΠΡΙΝ ξοδέψεις χρόνο σε πραγματικό hardware."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    qc = transpile(bell_circuit(), sim)
    res = sim.run(qc, shots=shots, memory=True).result()
    return res.get_memory()          # λίστα από strings, ΣΤΗ ΣΕΙΡΑ


def get_backend(region=None, instance=None, backend_name=None):
    """Συνδέεται ΜΙΑ φορά και επιστρέφει τον επεξεργαστή.

    Κρίσιμο: όταν μαζεύουμε πολλά jobs για έλεγχο drift, ΟΛΑ πρέπει να
    τρέξουν στο ΙΔΙΟ μηχάνημα. Αν καλούσαμε least_busy σε κάθε job, η
    συλλογή θα άλλαζε επεξεργαστή στη μέση και το «drift» που θα βλέπαμε
    στα όρια των jobs θα ήταν απλώς αλλαγή μηχανήματος.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        sys.exit("Δεν βρέθηκε το IBM_QUANTUM_TOKEN. Δες τις οδηγίες στην αρχή.")

    # region/instance: χρειάζονται για το EU instance (eu-de). Χωρίς αυτά
    # το service μπορεί να συνδεθεί σε λάθος περιοχή ή να μη βρει backend.
    kwargs = {"channel": "ibm_quantum_platform", "token": token}
    if region:
        kwargs["region"] = region
    if instance:
        kwargs["instance"] = instance
    service = QiskitRuntimeService(**kwargs)

    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(operational=True, simulator=False)
    print(f"Επεξεργαστής: {backend.name}")

    max_shots = getattr(backend.configuration(), "max_shots", None)
    if max_shots:
        print(f"max_shots ανά job: {max_shots:,}")
    return backend, max_shots


def run_hardware(backend, shots):
    """Ένα job στον δοσμένο επεξεργαστή. Επιστρέφει (bits, job_id)."""
    from qiskit_ibm_runtime import SamplerV2

    qc = transpile(bell_circuit(), backend)
    sampler = SamplerV2(mode=backend)
    job = sampler.run([qc], shots=shots)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}  (μπορεί να περιμένει σε ουρά)")

    result = job.result()[0]

    # Το όνομα του classical register δεν είναι πάντα 'c'. Το βρίσκουμε.
    names = list(result.data.keys())
    if "c" in names:
        creg = result.data.c
    elif len(names) == 1:
        creg = result.data[names[0]]
        print(f"Σημείωση: το classical register λέγεται '{names[0]}', όχι 'c'.")
    else:
        sys.exit(f"Ασαφή classical registers: {names}. Διόρθωσε το χέρι.")

    # ΠΡΟΣΟΧΗ: το .array είναι ΠΑΚΕΤΑΡΙΣΜΕΝΑ BYTES, όχι μία στήλη ανά qubit.
    # Με 2 qubit το arr[:,0] δίνει 0/3, όχι 0/1. Το slice_bits(0) δίνει
    # σωστά το bit του qubit 0, ανά shot, στη χρονική σειρά.
    bits = creg.slice_bits(0).array[:, 0].astype(np.int8)
    return bits, job_id


def to_bits(memory_strings):
    """Μετατροπή σε δυαδική ακολουθία. Κρατάμε ΜΟΝΟ το πρώτο qubit κάθε shot,
    ώστε η ακολουθία να είναι μία μέτρηση ανά χρονική στιγμή."""
    return np.array([int(s[-1]) for s in memory_strings], dtype=np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="τοπική προσομοίωση")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--repeats", type=int, default=1,
                    help="πόσες φορές να τρέξει (για να μαζέψεις περισσότερα)")
    ap.add_argument("--region", default=os.environ.get("IBM_QUANTUM_REGION"),
                    help="π.χ. eu-de  (ή μεταβλητή IBM_QUANTUM_REGION)")
    ap.add_argument("--instance", default=os.environ.get("IBM_QUANTUM_INSTANCE"),
                    help="CRN/όνομα instance (ή μεταβλητή IBM_QUANTUM_INSTANCE)")
    ap.add_argument("--backend", default=None,
                    help="κάρφωσε συγκεκριμένο επεξεργαστή (απαραίτητο για έλεγχο drift)")
    args = ap.parse_args()

    backend = max_shots = None
    if not args.sim:
        backend, max_shots = get_backend(args.region, args.instance, args.backend)
        if max_shots and args.shots > max_shots:
            sys.exit(f"--shots {args.shots:,} > max_shots {max_shots:,} του backend. "
                     f"Σπάσ' το σε περισσότερα --repeats.")

    all_bits, runs = [], []
    offset = 0
    for r in range(args.repeats):
        if args.sim:
            bits = to_bits(run_simulator(args.shots))
            tag, job_id = "simulator", None
        else:
            bits, job_id = run_hardware(backend, args.shots)
            tag = backend.name
        all_bits.append(bits)
        # κάθε job καταγράφεται ΞΕΧΩΡΙΣΤΑ, με τα ακριβή του όρια μέσα στην
        # τελική ακολουθία -> επιτρέπει έλεγχο drift στα όρια των jobs
        runs.append({"run": r + 1, "job_id": job_id, "shots": int(len(bits)),
                     "start": offset, "end": offset + len(bits),
                     "rate1": round(float(bits.mean()), 6),
                     "utc": datetime.now(timezone.utc).isoformat()})
        offset += len(bits)
        print(f"  run {r+1}/{args.repeats}: {len(bits):,} shots, "
              f"ρυθμός 1 = {bits.mean():.4f}")

    bits = np.concatenate(all_bits)
    fname = f"raw_{tag}_{len(bits)}.npy"
    np.save(fname, bits)

    meta = {"backend": tag, "total_bits": int(len(bits)),
            "rate1_overall": round(float(bits.mean()), 6),
            "shots_per_job": args.shots, "n_jobs": args.repeats, "runs": runs}
    mname = fname.replace(".npy", "_jobs.json")
    with open(mname, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nΑποθηκεύτηκε: {fname}   ({len(bits):,} bits)")
    print(f"Μεταδεδομένα jobs: {mname}")
    print("Επόμενο: python3 analyse_raw.py " + fname)


if __name__ == "__main__":
    main()
