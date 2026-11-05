"""Fetch/check the pinned ESIM source without installing old training dependencies."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Only verify the existing checkout; no network')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'third_party/esim.lock.json').read_text(encoding='utf-8'))
    target = ROOT / lock['directory']
    if not target.exists():
        if args.check:
            raise FileNotFoundError('Run scripts/setup_esim.py to fetch the pinned source first.')
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse', lock['repository'], str(target)], check=True)
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=target, text=True).strip()
    if current != lock['commit']:
        if args.check:
            raise RuntimeError(f'ESIM checkout {current} differs from the lock {lock["commit"]}.')
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=target, text=True).strip():
            raise RuntimeError('ESIM checkout has local edits; preserve them before switching the pinned version.')
        subprocess.run(['git', 'fetch', '--depth', '1', 'origin', lock['commit']], cwd=target, check=True)
        subprocess.run(['git', 'checkout', '--detach', lock['commit']], cwd=target, check=True)
    if not args.check:
        subprocess.run(['git', 'sparse-checkout', 'set', *lock['sparse_paths']], cwd=target, check=True)
    for name in ['LICENSE', 'esim/model.py', 'esim/layers.py', 'esim/utils.py']:
        if not (target / name).is_file():
            raise FileNotFoundError(f'Missing upstream file: {target / name}')
    if subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=target, text=True).strip():
        raise RuntimeError('The upstream checkout was modified. Record adaptations separately before training.')
    print(json.dumps({'repository': lock['repository'], 'commit': lock['commit'],
                      'directory': str(target), 'license': lock['license'], 'upstream_modified': False}))


if __name__ == '__main__':
    main()
