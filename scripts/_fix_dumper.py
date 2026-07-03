"""Insert a Cell-5 source-dump block at the end of CELL5_LINES so DDP workers in Cell 6 can re-exec the model definition via /tmp/attnres_def.py."""

from pathlib import Path

p = Path("scripts/build_notebook_01.py")
src = p.read_text()

# Lines we want to insert (each becomes a Cell-5 source line).
new_lines = [
    "",
    "# --- Persist the model definition so DDP workers (Cell 6) can re-exec via /tmp/attnres_def.py ---",
    "# We read Cell 5's input via IPython's _ih (more reliable than inspect.getsource in Jupyter)",
    "# and slice at the '# === END_OF_MODEL_DEFINITION ===' marker to capture the model classes.",
    "try:",
    "    from IPython import get_ipython",
    "    _ih = get_ipython().user_ns.get('_ih', [])",
    "    print(f'CELL5 DDP-scan: _ih has {len(_ih)} entries; recent lengths: {[len(s) for s in _ih[-3:]]}')",
    "    _src = ''",
    "    _marker = '# === END_OF_MODEL_DEFINITION ==='",
    "    for _cell_src in _ih:",
    "        if _marker in _cell_src:",
    "            _src = _cell_src.split(_marker)[0]",
    "            break",
    "except Exception as _dump_err:",
    "    _src = ''",
    "    print(f'CELL5 DDP-scan FAILED: {_dump_err}')",
    "",
    "if _src:",
    "    with open('/tmp/attnres_def.py', 'w') as _attnres_f:",
    "        _attnres_f.write(_src)",
    "    print(f'Saved {len(_src)} bytes of Cell 5 source to /tmp/attnres_def.py for DDP workers')",
    "else:",
    "    print('WARNING: failed to extract Cell 5 source. DDP workers may fail.')",
]


def esq(s: str) -> str:
    """Escape backslashes + double quotes for embedding in Python source strings."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


# Format each line as a CELL5_LINES list entry. Last entry trailing comma stripped (followed by ']').
parts = []
for i, ln in enumerate(new_lines):
    is_last = i == len(new_lines) - 1
    sep = "" if is_last else ","
    parts.append('    "' + esq(ln) + '"' + sep)
new_entries_joined = "\n".join(parts)

# Anchor: the existing 'torch.cuda.empty_cache()' entry + closing ']'
anchor = '    "torch.cuda.empty_cache()",\n]\n'
replacement = '    "torch.cuda.empty_cache()",\n\n' + new_entries_joined + '\n]\n'

if anchor not in src:
    raise SystemExit("anchor not found - did the file change?")
new_src = src.replace(anchor, replacement, 1)
p.write_text(new_src)
print(f"wrote {p}: {len(src)} -> {len(new_src)} bytes (+{len(new_src) - len(src)})")
print(f"inserted {len(new_lines)} new cell-source lines into CELL5_LINES")
