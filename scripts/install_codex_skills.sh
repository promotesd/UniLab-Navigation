#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${REPO_ROOT}/codex/skills"
CODEX_ROOT="${CODEX_HOME:-${HOME}/.codex}"
DEST_ROOT="${CODEX_ROOT}/skills"

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Skill source directory not found: ${SOURCE_ROOT}" >&2
  exit 1
fi

mkdir -p "${DEST_ROOT}"

installed=0
for source_dir in "${SOURCE_ROOT}"/*; do
  [[ -d "${source_dir}" ]] || continue
  skill_name="$(basename "${source_dir}")"
  destination="${DEST_ROOT}/${skill_name}"

  rm -rf "${destination}"
  cp -a "${source_dir}" "${destination}"
  echo "Installed ${skill_name} -> ${destination}"
  installed=$((installed + 1))
done

if [[ "${installed}" -eq 0 ]]; then
  echo "No skills were found under ${SOURCE_ROOT}" >&2
  exit 1
fi

echo
echo "Installed ${installed} UniLab Codex skills."
echo "Restart Codex, then invoke: \$unilab-navigation-orchestrator"
