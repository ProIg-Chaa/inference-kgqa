"""Replay the teacher notebook's Python cells without a Jupyter dependency."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.data_dir:
        os.environ['METAQA_DATA_DIR'] = str(args.data_dir.resolve())
    notebook = json.loads((root/'notebooks/lesson02_teacher_rehearsal.ipynb').read_text(encoding='utf-8'))
    os.chdir(root)
    namespace = {'__name__': '__lesson02_rehearsal__'}
    executed = 0
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            print('\nCell:', cell['id'])
            exec(compile(''.join(cell['source']), cell['id'], 'exec'), namespace)
            executed += 1
    print(f'\nCompleted {executed} teacher notebook code cells.')


if __name__ == '__main__':
    main()
