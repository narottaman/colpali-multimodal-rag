#!/bin/bash
# =============================================================================
# fix_chromadb.sh — ChromaDB SQLite fix via LD_PRELOAD (corrected)
#
# Previous attempt failed because pysqlite3.__file__ returns __init__.py
# (a Python text file), not the compiled .so extension.
# We must find the actual .cpython-312-*.so file inside the pysqlite3 package.
#
# Run interactively:
#   source ~/envs/rag/bin/activate
#   bash sol/fix_chromadb.sh
# =============================================================================

set -e
VENV="$HOME/envs/rag"
source "$VENV/bin/activate"

echo "============================================================"
echo " ChromaDB SQLite Fix via LD_PRELOAD (corrected)"
echo "============================================================"

# ── Step 1: Install pysqlite3-binary ─────────────────────────────────────────
echo ""
echo "── Step 1: Install pysqlite3-binary ─────────────────────────"
pip install pysqlite3-binary --quiet

# Find the ACTUAL compiled .so (not __init__.py)
PYSQLITE_PKG_DIR=$(python -c "
import pysqlite3, pathlib
pkg_dir = pathlib.Path(pysqlite3.__file__).parent
print(pkg_dir)
")
echo "  pysqlite3 package dir: $PYSQLITE_PKG_DIR"

# List what's in the package dir
echo "  Contents:"
ls -la "$PYSQLITE_PKG_DIR"

# Find the .so file — it will be named like:
# _pysqlite3.cpython-312-x86_64-linux-gnu.so  OR
# pysqlite3.cpython-312-x86_64-linux-gnu.so
PYSQLITE_SO=$(find "$PYSQLITE_PKG_DIR" -name "*.so" | head -1)

if [ -z "$PYSQLITE_SO" ]; then
    echo "  ERROR: No .so file found in $PYSQLITE_PKG_DIR"
    echo "  Trying broader search..."
    PYSQLITE_SO=$(find "$VENV" -name "*pysqlite3*.so" 2>/dev/null | head -1)
fi

if [ -z "$PYSQLITE_SO" ]; then
    echo "  ERROR: Cannot find pysqlite3 .so anywhere in venv"
    echo "  Trying: pip install pysqlite3-binary --force-reinstall"
    pip install pysqlite3-binary --force-reinstall --quiet
    PYSQLITE_SO=$(find "$VENV" -name "*pysqlite3*.so" 2>/dev/null | head -1)
fi

echo "  Found .so: $PYSQLITE_SO"

# Verify it's a real ELF binary
file "$PYSQLITE_SO"

# Check SQLite version embedded in it
python -c "
import pysqlite3
print(f'  pysqlite3 SQLite version: {pysqlite3.sqlite_version}')
con = pysqlite3.connect(':memory:')
print(f'  Connection test: OK')
con.close()
"

# ── Step 2: Check what libsqlite3 pysqlite3.so links to ─────────────────────
echo ""
echo "── Step 2: Inspecting .so linkage ──────────────────────────"
ldd "$PYSQLITE_SO" 2>/dev/null || echo "  ldd not available"

# Check if it statically links sqlite3 (no libsqlite3.so in ldd output = static)
if ldd "$PYSQLITE_SO" 2>/dev/null | grep -q libsqlite3; then
    echo "  Dynamically links libsqlite3 — finding the linked .so"
    SQLITE_SO=$(ldd "$PYSQLITE_SO" | grep libsqlite3 | awk '{print $3}')
    echo "  libsqlite3.so path: $SQLITE_SO"
    LD_TARGET="$SQLITE_SO"
else
    echo "  Statically links sqlite3 (no libsqlite3.so dependency)"
    echo "  Using pysqlite3.so itself as LD_PRELOAD target"
    LD_TARGET="$PYSQLITE_SO"
fi

echo "  LD_PRELOAD target: $LD_TARGET"

# ── Step 3: Write LD_PRELOAD into venv activate ───────────────────────────────
echo ""
echo "── Step 3: Baking LD_PRELOAD into venv activate ─────────────"
ACTIVATE="$VENV/bin/activate"
MARKER="# SQLITE_FIX_V2"

# Remove any old broken fix first
grep -v "SQLITE_FIX\|sqlite3_fix\|libsqlite3" "$ACTIVATE" > /tmp/activate_clean || true
cp /tmp/activate_clean "$ACTIVATE"

cat >> "$ACTIVATE" << ENVEOF

$MARKER — prepend modern libsqlite3 before Mamba's broken one
export LD_PRELOAD="${LD_TARGET}:\${LD_PRELOAD:-}"
ENVEOF
echo "  Written to $ACTIVATE"

# ── Step 4: Re-source and test ────────────────────────────────────────────────
echo ""
echo "── Step 4: Testing with LD_PRELOAD active ───────────────────"
source "$ACTIVATE"
echo "  LD_PRELOAD = $LD_PRELOAD"

# Verify ELF header is valid
python - << 'PYCHECK'
import subprocess, os
ld_preload = os.environ.get("LD_PRELOAD", "").split(":")[0]
if ld_preload:
    result = subprocess.run(["file", ld_preload], capture_output=True, text=True)
    print(f"  file check: {result.stdout.strip()}")
PYCHECK

python - << 'PYCHECK'
import sys, tempfile
failed = []

for pkg in ["chromadb", "qdrant_client"]:
    try:
        __import__(pkg)
        print(f"  OK   {pkg}")
    except Exception as e:
        print(f"  FAIL {pkg}: {e}")
        failed.append(pkg)

if "chromadb" not in failed:
    try:
        import chromadb, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            c = chromadb.PersistentClient(path=tmp)
            col = c.get_or_create_collection("smoke")
            col.add(documents=["attention is all you need"], ids=["1"])
            r = col.query(query_texts=["transformer"], n_results=1)
            print(f"  OK   chromadb.PersistentClient — '{r['documents'][0][0][:35]}'")
    except Exception as e:
        print(f"  FAIL chromadb.PersistentClient: {e}")
        failed.append("PersistentClient")

if failed:
    print(f"\nStill failing: {failed}")
    print("\nDiagnostic info:")
    print(f"  LD_PRELOAD = {os.environ.get('LD_PRELOAD', 'not set')}")
    import subprocess
    result = subprocess.run(["python", "-c", "import sqlite3; print(sqlite3.sqlite_version)"],
                           capture_output=True, text=True)
    print(f"  sqlite3 version seen by Python: {result.stdout.strip()}")
    sys.exit(1)
else:
    print("\nAll OK! ChromaDB PersistentClient works.")
    print("LD_PRELOAD is baked into your venv activate script.")
    print("\nRe-submit:")
    print("  sbatch sol/approach_a.slurm")
    print("  sbatch sol/approach_b.slurm")
PYCHECK
