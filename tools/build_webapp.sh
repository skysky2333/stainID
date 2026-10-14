#!/usr/bin/env bash
# Build the web app into src/stainid/api/static (the bundle that is committed, so installing stainID needs no Node).
# npm needs symlinks in node_modules; on a drive that cannot hold them (exFAT, FAT32) the build runs in a copy of webapp/
# in a temporary folder. Uses ~/.local/node/bin when node is not on the PATH.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$repo/src/stainid/api/static"
command -v npm >/dev/null || export PATH="$HOME/.local/node/bin:$PATH"
command -v npm >/dev/null || { echo "Node.js (npm) not found: install Node 20 or later" >&2; exit 1; }

can_link() {
  local probe="$1/.symlink-probe-$$"
  ln -s . "$probe" 2>/dev/null && rm -f "$probe"
}

if can_link "$repo/webapp"; then
  work="$repo/webapp"
else
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/stainid-webapp.XXXXXX")"
  trap 'rm -rf "$tmp"' EXIT
  work="$tmp/webapp"
  mkdir -p "$work"
  echo "webapp/ is on a drive without symlinks; building in $work"
  (cd "$repo/webapp" && tar --exclude=./node_modules --exclude=./dist -cf - .) | (cd "$work" && tar -xf -)
fi

cd "$work"
npm ci --no-audit --no-fund
# npm leaves out the bundler's native part when Node is a little older than it asks for (npm/cli#4828); fetch it directly.
binding="$(node -p "const p = require('./node_modules/rolldown/package.json').optionalDependencies || {};
  const n = Object.keys(p).find((k) => k.endsWith('-' + process.platform + '-' + process.arch + (process.platform === 'linux' ? '-gnu' : '')));
  n ? n + '@' + p[n] : ''")"
if [ -n "$binding" ] && [ ! -d "node_modules/${binding%@*}" ]; then
  echo "Installing $binding by hand"
  mkdir -p "node_modules/${binding%@*}"
  tarball="$(npm pack --silent "$binding")"
  tar -xzf "$tarball" -C "node_modules/${binding%@*}" --strip-components=1
  rm -f "$tarball"
fi
npx tsc -b
npx vite build --outDir "$out" --emptyOutDir
echo "Built the web app into $out"
