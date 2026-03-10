import subprocess, datetime, os, sys, sqlite3

# ── Usage ─────────────────────────────────────────────────────────────────────
# Profile + parse:   python3 nsysrun.py run  pinned  ./2paulv5.o
# Parse only:        python3 nsysrun.py parse 20260306_1641_pinned_.nsys-rep

mode = sys.argv[1] if len(sys.argv) > 1 else "run"
out_dir = "./paul_implem/profiles"
os.makedirs(out_dir, exist_ok=True)

# ── Mode: parse existing file ─────────────────────────────────────────────────
if mode == "parse":
    nsys_rep = sys.argv[2]
    base     = nsys_rep.replace(".nsys-rep", "")
    last_epoch, gpu_time = None, None

# ── Mode: profile + parse ─────────────────────────────────────────────────────
elif mode == "run":
    suffix = sys.argv[2] if len(sys.argv) > 2 else "pinned"
    exe    = sys.argv[3] if len(sys.argv) > 3 else "./2paulv5.o"
    base   = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_{suffix}_"

    result = subprocess.run(["nsys", "profile", "-o", base, exe],
                            capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)

    all_out    = result.stdout + result.stderr
    last_epoch = next((l for l in reversed(all_out.splitlines()) if "Epoch" in l and "loss" in l), None)
    gpu_time   = next((l for l in all_out.splitlines() if "GPU compute" in l), None)

else:
    print("Usage:\n  python3 profile.py run  <suffix> <exe>\n  python3 profile.py parse <file.nsys-rep>")
    sys.exit(1)

# ── Generate sqlite ───────────────────────────────────────────────────────────
os.system(f"nsys stats '{base}.nsys-rep' > /dev/null 2>&1")

db  = sqlite3.connect(f"{base}.sqlite")
cur = db.cursor()

def ns(n):
    return f"{int(n):,}".replace(",", " ")

def section(title, sql, min_rows=5, max_rows=7):
    rows = cur.execute(sql).fetchall()
    if not rows: return []
    grand = sum(r[1] for r in rows)
    kept  = rows[:max(min_rows, sum(1 for r in rows if 100*r[1]/grand >= 4.0))][:max_rows]
    lines = [f"\n ** {title}",
             f"  {'Time%':>6}  {'Total(ns)':>18}  {'Calls':>13}  {'Avg(ns)':>15}  {'Max(ns)':>15}  Name",
             "  " + "-"*90]
    for r in kept:
        lines.append(f"  {100*r[1]/grand:6.1f}%  {ns(r[1]):>18}  {ns(r[2]):>13}  {ns(round(r[1]/r[2])):>15}  {ns(r[3]):>15}  {r[0]}")
    return lines

out = [f"# {base}"]
if mode == "run":
    if last_epoch: out.append(f"# {last_epoch.strip()}")
    if gpu_time:   out.append(f"# {gpu_time.strip()}")

out += section("OS Runtime Summary", """
    SELECT s.value, SUM(a.end-a.start), COUNT(*), MAX(a.end-a.start)
    FROM OSRT_API a JOIN StringIds s ON s.id=a.nameId
    GROUP BY a.nameId ORDER BY 2 DESC""", min_rows=3)

out += section("CUDA API Summary", """
    SELECT s.value, SUM(a.end-a.start), COUNT(*), MAX(a.end-a.start)
    FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id=a.nameId
    GROUP BY a.nameId ORDER BY 2 DESC""", min_rows=5)

out += section("CUDA GPU Kernel Summary", """
    SELECT s.value, SUM(k.end-k.start), COUNT(*), MAX(k.end-k.start)
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
    GROUP BY k.shortName ORDER BY 2 DESC""", min_rows=5)

# MemOps
rows  = cur.execute("""
    SELECT copyKind, SUM(end-start), COUNT(*), MAX(end-start)
    FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind ORDER BY 2 DESC""").fetchall()
grand = sum(r[1] for r in rows)
kinds = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
if grand:
    out += ["\n ** CUDA MemOps Summary by Time",
            f"  {'Time%':>6}  {'Total(ns)':>18}  {'Count':>13}  {'Avg(ns)':>15}  {'Max(ns)':>15}  Op",
            "  " + "-"*90]
    for r in rows:
        out.append(f"  {100*r[1]/grand:6.1f}%  {ns(r[1]):>18}  {ns(r[2]):>13}  {ns(round(r[1]/r[2])):>15}  {ns(r[3]):>15}  {kinds.get(r[0], r[0])}")

db.close()

# ── Save ──────────────────────────────────────────────────────────────────────
path = f"{out_dir}/{base}stats.txt"
open(path, "w").write("\n".join(out) + "\n")
print(f"\nSaved: {path}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
for ext in [".sqlite", ".nsys-rep"]:
    f = f"{base}{ext}"
    if os.path.exists(f):
        os.remove(f)
        print(f"Deleted: {f}")