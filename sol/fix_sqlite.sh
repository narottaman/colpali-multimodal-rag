#!/bin/bash
# =============================================================================
# fix_sqlite.sh — Fix sqlite3_deserialize on Sol (Mamba conflict)
#
# The problem: Sol's Mamba Python loads _sqlite3.so from its own path before
# any venv sitecustomize.py can intercept it. We must patch chromadb and
# qdrant_client at the package level to swap sqlite3 → pysqlite3 themselves.
#
# Run interactively (not via SLURM):
#   source ~/envs/rag/bin/activate
#   bash sol/fix_sqlite.sh
# =============================================================================

set -e
VENV="$HOME/envs/rag"

echo "============================================================"
echo " SQLite Fix — patching chromadb + qdrant at package level"
echo " Venv: $VENV"
echo "============================================================"

source "$VENV/bin/activate"

# ── Install pysqlite3-binary ──────────────────────────────────────────────────
echo ""
echo "── Step 1: Install pysqlite3-binary ─────────────────────────"
pip install pysqlite3-binary --quiet
python -c "import pysqlite3; print(f'  pysqlite3 SQLite version: {pysqlite3.sqlite_version}')"

# ── Find package locations ────────────────────────────────────────────────────
echo ""
echo "── Step 2: Locating packages ────────────────────────────────"
CHROMA_INIT=$(python -c "import importlib.util; spec=importlib.util.find_spec('chromadb'); print(spec.origin)")
QDRANT_INIT=$(python -c "import importlib.util; spec=importlib.util.find_spec('qdrant_client'); print(spec.origin)")
echo "  chromadb    : $CHROMA_INIT"
echo "  qdrant_client: $QDRANT_INIT"

CHROMA_DIR=$(dirname "$CHROMA_INIT")
QDRANT_DIR=$(dirname "$QDRANT_INIT")

# ── Patch chromadb/__init__.py ────────────────────────────────────────────────
echo ""
echo "── Step 3: Patching chromadb/__init__.py ────────────────────"

# Check if already patched
if grep -q "pysqlite3" "$CHROMA_INIT"; then
    echo "  Already patched — skipping"
else
    # Prepend the swap to the top of chromadb's __init__.py
    TMPFILE=$(mktemp)
    cat > "$TMPFILE" << 'PATCH'
# --- SQLite patch for Sol HPC (prepended by fix_sqlite.sh) ---
try:
    import pysqlite3 as _sq3
    import sys as _sys
    _sys.modules["sqlite3"] = _sq3
except ImportError:
    pass
# --- end patch ---

PATCH
    cat "$CHROMA_INIT" >> "$TMPFILE"
    cp "$TMPFILE" "$CHROMA_INIT"
    rm "$TMPFILE"
    echo "  Patched: $CHROMA_INIT"
fi

# ── Patch qdrant_client/__init__.py ──────────────────────────────────────────
echo ""
echo "── Step 4: Patching qdrant_client/__init__.py ───────────────"

if grep -q "pysqlite3" "$QDRANT_INIT"; then
    echo "  Already patched — skipping"
else
    TMPFILE=$(mktemp)
    cat > "$TMPFILE" << 'PATCH'
# --- SQLite patch for Sol HPC (prepended by fix_sqlite.sh) ---
try:
    import pysqlite3 as _sq3
    import sys as _sys
    _sys.modules["sqlite3"] = _sq3
except ImportError:
    pass
# --- end patch ---

PATCH
    cat "$QDRANT_INIT" >> "$TMPFILE"
    cp "$TMPFILE" "$QDRANT_INIT"
    rm "$TMPFILE"
    echo "  Patched: $QDRANT_INIT"
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "── Step 5: Verifying ────────────────────────────────────────"
python - << 'PYCHECK'
import sys
failed = []
for pkg in ["chromadb", "qdrant_client"]:
    try:
        __import__(pkg)
        print(f"  OK   {pkg}")
    except Exception as e:
        print(f"  FAIL {pkg}: {e}")
        failed.append(pkg)

if failed:
    print(f"\nStill failing: {failed}")
    print("The package __init__.py patch didn't take. Try:")
    print("  python -c \"import chromadb\" 2>&1 | head -5")
    sys.exit(1)
else:
    print("\nSQLite fix confirmed! Both packages import cleanly.")
    print("Re-submit your SLURM jobs:")
    print("  sbatch sol/approach_a.slurm")
    print("  sbatch sol/approach_b.slurm")
PYCHECK
