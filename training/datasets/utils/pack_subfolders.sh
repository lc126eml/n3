#!/usr/bin/env bash
set -euo pipefail

parent="${1:-}"
outdir="${2:-_archives}"

if [[ -z "$parent" ]]; then
  echo "Usage: $0 /path/to/parent [output_dir_name_or_path]" >&2
  exit 2
fi
if [[ ! -d "$parent" ]]; then
  echo "Error: not a directory: $parent" >&2
  exit 1
fi

cd "$parent"
mkdir -p "$outdir"
outbase="${outdir%/}"

for d in */ ; do
  [[ -d "$d" ]] || continue
  [[ "${d%/}" == "$outbase" ]] && continue
  tar -czf "$outdir/${d%/}.tar.gz" "$d"
done
