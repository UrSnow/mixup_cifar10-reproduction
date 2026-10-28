"""Rebuild the small upload bundle after reviewed code/config/documentation changes."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
output = root / 'mixup-colab-project.zip'
folders = ['mixup_repro', 'configs', 'docs', 'notebooks', 'scripts', 'tests', 'vendor']
files = [root / name for name in ['README.md', 'requirements.txt', '.gitignore']]
for name in folders:
    files.extend(p for p in (root / name).rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts
                 and p.suffix.lower() not in ['.pyc', '.pdf', '.pt', '.zip'])
with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, path.relative_to(root).as_posix())
with ZipFile(output) as archive:
    assert archive.testzip() is None
print(f'Created {output.name}: {len(files)} files, {output.stat().st_size:,} bytes')
