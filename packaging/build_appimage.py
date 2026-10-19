#!/usr/bin/env python3
"""Offline, stdlib-only AppImage assembly from a pre-relocated Python runtime."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from email.parser import Parser
from pathlib import Path

GITHUB_ASSET_LIMIT = 2 * 1024 ** 3
EXCLUDED_COMPONENTS = {'__pycache__', '.git', '.pytest_cache', 'tests', 'test',
                       'credentials', '.cache', '.config'}
GLIBC = re.compile(r'^(?:ld-linux.*|lib(?:c|m|pthread|dl|rt|resolv|util|anl|BrokenLocale|nss_[^.]+)\.so(?:\..*)?)$')
DRIVER = re.compile(r'^(?:libcuda\.so.*|libnvidia.*|libGLX_nvidia.*|libEGL_nvidia.*|libvulkan_(?:intel|radeon|nouveau).*|.*_dri\.so)$')


def excluded(path: Path) -> bool:
    # The speech recognizer requires its small intrinsic VAD asset. Application
    # ASR/translation/voice weights still live outside the bundle.
    if path.as_posix().endswith(('faster_whisper/assets/silero_vad_v6.onnx','certifi/cacert.pem')):
        return False
    return (bool(set(path.parts) & EXCLUDED_COMPONENTS)
            or ('models' in path.parts and 'site-packages' not in path.parts)
            # The app is QtWidgets-only. The wheel's unrelated QML compositor
            # plugins reference libraries absent even from the original wheel.
            or 'PySide6/Qt/qml' in path.as_posix()
            or 'PySide6/Qt/plugins/sqldrivers' in path.as_posix()
            or path.suffix in {'.pyc', '.pyo', '.gguf', '.onnx', '.safetensors', '.pt', '.pth'}
            or path.name.startswith(('sitecustomize', 'usercustomize'))
            or path.name in {'direct_url.json', 'INSTALLER', 'RECORD', 'runtime-base.ready.json', 'pyvenv.cfg'}
            or path.name.endswith(('.egg-link', '.pem', '.key')))


def safe_copy_tree(source: Path, target: Path, without_cuda=False) -> None:
    source = source.resolve()
    def ignore(directory, names):
        ignored = []
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(source)
            if excluded(relative) or (without_cuda and (name == 'nvidia' or name.startswith('nvidia_') or name.startswith('libtorch_cuda') or name.startswith('libc10_cuda'))):
                ignored.append(name)
            elif path.is_symlink() and (not path.exists() or not path.resolve().is_relative_to(source)):
                ignored.append(name)
            elif path.is_symlink() and os.path.isabs(os.readlink(path)):
                ignored.append(name)
        return ignored
    shutil.copytree(source, target, symlinks=True, ignore=ignore)


def extract_backend(archive: Path, target: Path) -> None:
    target.mkdir(parents=True)
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        by_name = {m.name: m for m in members}
        for member in members:
            name = Path(member.name).name
            if name != 'llama-server' and '.so' not in name and not name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING')):
                continue
            if member.isdir():
                continue
            # Read hard/symbolic links only when they remain within the archive tree.
            current = member
            seen = set()
            while current.issym() or current.islnk():
                if current.name in seen:
                    raise BuildError('Cyclic backend archive link')
                seen.add(current.name)
                link = current.linkname if current.islnk() else str(Path(current.name).parent / current.linkname)
                if Path(link).is_absolute() or '..' in Path(link).parts or link not in by_name:
                    raise BuildError('Unsafe backend archive link')
                current = by_name[link]
            if not current.isfile():
                raise BuildError('Unsupported backend archive member')
            content = stream.extractfile(current)
            if content is None:
                raise BuildError('Unreadable backend member')
            with content, (target / name).open('wb') as output:
                shutil.copyfileobj(content, output)
            (target / name).chmod(0o755 if name == 'llama-server' or '.so' in name else 0o644)
    if not (target / 'llama-server').is_file() or not any(p.name.upper().startswith('LICENSE') for p in target.iterdir()):
        raise BuildError('Backend archive lacks llama-server or license')


def is_elf(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    with path.open('rb') as stream:
        return stream.read(4) == b'\x7fELF'


def collect_native_libraries(appdir: Path) -> list[Path]:
    """Resolve every bundled ELF with ldd; preserve package files and their RPATHs."""
    native = appdir / 'usr/lib/native'
    native.mkdir(parents=True, exist_ok=True)
    package_libraries = {}
    seeds = []
    for path in sorted((appdir / 'usr').rglob('*')):
        if is_elf(path):
            seeds.append(path)
            if '.so' in path.name:
                package_libraries.setdefault(path.name, path)
    env = dict(os.environ)
    env.pop('LD_PRELOAD', None)
    # Prefer packaged dependencies, not host copies with coincidentally matching SONAMEs.
    directories = sorted({str(p.parent) for p in package_libraries.values()})
    env['LD_LIBRARY_PATH'] = ':'.join(directories)
    copied, processed = [], set()
    while seeds:
        path = seeds.pop()
        if path in processed:
            continue
        processed.add(path)
        result = subprocess.run(['ldd', str(path)], capture_output=True, text=True, env=env, timeout=30)
        for line in result.stdout.splitlines():
            missing = re.match(r'\s*(\S+)\s+=>\s+not found', line)
            if missing:
                name = missing.group(1)
                if not DRIVER.match(name) and name not in package_libraries:
                    raise BuildError(f'Unresolved non-driver dependency {name} in {path.name}')
                continue
            match = re.match(r'\s*(\S+)\s+=>\s+(/\S+)', line)
            if not match:
                continue
            name, resolved = match.groups()
            if GLIBC.match(name) or DRIVER.match(name) or name in package_libraries:
                continue
            source = Path(resolved)
            if source.resolve().is_relative_to(appdir.resolve()):
                continue
            target = native / name
            if not target.exists():
                shutil.copy2(source.resolve(), target)
                package_libraries[name] = target
                copied.append(source)
                seeds.append(target)
    # Never carry driver/glibc binaries supplied by a mistakenly broad base.
    for path in native.iterdir():
        if GLIBC.match(path.name) or DRIVER.match(path.name):
            path.unlink()
    return copied


def collect_notices(appdir: Path, host_libraries: list[Path], tools: dict) -> None:
    doc = appdir / 'usr/share/doc/translitter'
    doc.mkdir(parents=True, exist_ok=True)
    inventory = []
    for info in sorted((appdir / 'usr/lib/python3.12/site-packages').glob('*.dist-info')):
        metadata = info / 'METADATA'
        if metadata.exists():
            record = Parser().parsestr(metadata.read_text(errors='replace'))
            inventory.append({'name': record.get('Name', info.stem), 'version': record.get('Version', ''),
                              'license': record.get('License-Expression') or record.get('License', ''),
                              'sources': record.get_all('Project-URL', []) + ([record['Home-page']] if record.get('Home-page') else []),
                              'license_directory': str(info.relative_to(appdir))})
    package_names = {'python3.12', 'python3.12-minimal', 'libpython3.12-stdlib', 'libpython3.12-minimal'}
    # Include copyright/source information for inherited native libraries too.
    paths = host_libraries + [Path('/usr/lib/x86_64-linux-gnu') / p.name for p in (appdir / 'usr/lib/native').iterdir()]
    for path in paths:
        query = subprocess.run(['dpkg-query', '-S', str(path)], capture_output=True, text=True)
        for line in query.stdout.splitlines():
            if ': ' in line:
                package_names.add(line.split(': ', 1)[0].split(':')[0])
    for name in sorted(package_names):
        source = Path('/usr/share/doc') / name / 'copyright'
        if source.is_file():
            target = doc / 'system' / name / 'copyright'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            inventory.append({'name': name, 'source': 'Debian/Ubuntu source package; see bundled copyright',
                              'copyright': str(target.relative_to(appdir))})
    for name, pin in tools.items():
        inventory.append({'name': name, **{k: pin[k] for k in ('repository', 'revision', 'sha256') if k in pin}})
    (doc / 'third-party.json').write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
    (doc / 'NOTICE').write_text('Translitter bundles Python, Python packages and native libraries.\n'
                              'Package license files remain in site-packages/*dist-info; host copyrights are in system/.\n'
                              'llama.cpp MIT licenses are in the application runtime/cpu and runtime/vulkan directories.\n'
                              'Model weights are NOT included; their licenses accompany explicit model installation.\n')


def prepare_appdir(root: Path, runtime_base: Path, tools_cache: Path, version: str,
                   without_cuda=False) -> Path:
    root = root.resolve()
    if not re.fullmatch(r'[0-9A-Za-z][0-9A-Za-z.+_-]*', version):
        raise BuildError('Unsafe version string')
    lock = json.loads((root / 'packaging/tools.lock.json').read_text())
    if lock.get('schema') != 1 or lock.get('architecture') != 'x86_64':
        raise BuildError('Unsupported tools lock schema or architecture')
    tools = lock['tools']
    for pin in tools.values():
        if Path(pin['name']).name != pin['name']:
            raise BuildError('Unsafe pinned filename')
        verify_pin(tools_cache / pin['name'], pin)
    if not (runtime_base / 'usr/bin/python3.12').is_file():
        raise BuildError('Runtime base must contain usr/bin/python3.12')
    appdir = root / 'build/Translitter.AppDir'
    if runtime_base.resolve() == appdir or appdir in runtime_base.resolve().parents:
        raise BuildError('Runtime base must not be inside the replaceable AppDir')
    if without_cuda and (runtime_base / 'usr/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so').exists():
        raise BuildError('--without-cuda requires a CPU-only PyTorch runtime base; removing NVIDIA libraries from a CUDA torch wheel breaks even CPU imports. No acceleration files were silently removed.')
    if appdir.is_symlink() or (root / 'build').is_symlink():
        raise BuildError('AppDir/build must not be symlinks')
    if appdir.exists():
        shutil.rmtree(appdir)
    appdir.mkdir(parents=True)
    safe_copy_tree(runtime_base / 'usr', appdir / 'usr', without_cuda)
    app = appdir / 'usr/lib/translitter/app'
    package = app / 'translitter'
    package.mkdir(parents=True)
    for source in sorted((root / 'translitter').glob('*.py')):
        shutil.copyfile(source, package / source.name)
    for name in ('model-assets.lock.json', 'quality-model.lock.json'):
        shutil.copyfile(root / name, app / name)
    (app / 'assets').mkdir()
    shutil.copyfile(root / 'assets/translitter.svg', app / 'assets/translitter.svg')
    for backend in ('cpu', 'vulkan'):
        pin = tools['llama-' + backend]
        extract_backend(tools_cache / pin['name'], app / 'runtime' / backend)
    shutil.copyfile(root / 'packaging/launch.py', app.parent / 'launch.py')
    shutil.copyfile(root / 'packaging/AppRun', appdir / 'AppRun')
    (appdir / 'AppRun').chmod(0o755)
    (appdir / 'usr/bin/python3.12').chmod(0o755)
    for name in ('translitter.svg', '.DirIcon'):
        shutil.copyfile(root / 'assets/translitter.svg', appdir / name)
    (appdir / 'translitter.desktop').write_text('[Desktop Entry]\nType=Application\nName=Translitter\n'
        'Exec=AppRun\nIcon=translitter\nCategories=AudioVideo;Utility;\nTerminal=false\n')
    meta = {'schema': 1, 'version': version, 'architecture': 'x86_64', 'minimum_glibc': '2.39',
            'cuda': not without_cuda, 'python': '3.12', 'layout': {'app': 'usr/lib/translitter/app',
            'bootstrap': 'usr/lib/translitter/launch.py', 'cpu': 'runtime/cpu/llama-server',
            'vulkan': 'runtime/vulkan/llama-server'}, 'tools': tools, 'models_bundled': False,
            'source_date_epoch': int(os.environ.get('SOURCE_DATE_EPOCH', '0'))}
    (app / 'appimage-build.json').write_text(json.dumps(meta, sort_keys=True, indent=2) + '\n')
    # Use the packaged public CA set, never the developer machine's custom
    # trust store. The latter could contain private enterprise trust anchors.
    cert = appdir / 'usr/lib/python3.12/site-packages/certifi/cacert.pem'
    if not cert.is_file():
        raise BuildError('Runtime base must include the certifi public CA bundle')
    destination = appdir / 'usr/share/translitter/ca-certificates.crt'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cert, destination)
    libraries = collect_native_libraries(appdir)
    collect_notices(appdir, libraries, tools)
    license_root=root / 'packaging/licenses'
    notices=json.loads((license_root / 'manifest.json').read_text(encoding='utf-8'))
    if not isinstance(notices,list) or not notices:
        raise BuildError('Pinned runtime license notices are required')
    destination=appdir / 'usr/share/doc/translitter/runtime-licenses'
    destination.mkdir(parents=True)
    for entry in notices:
        name=entry['file']
        if Path(name).name!=name or name in ('.','..') or (license_root/name).is_symlink():
            raise BuildError('Unsafe runtime license filename')
        notice=license_root/name
        verify_pin(notice,{'bytes':entry.get('bytes',notice.stat().st_size),'sha256':entry['sha256']})
        shutil.copyfile(notice,destination/name)
    shutil.copyfile(license_root/'manifest.json',destination/'manifest.json')
    return appdir


class BuildError(RuntimeError):
    """A release prerequisite or safety check failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pin(path: Path, pin: dict) -> None:
    if not path.is_file() or path.stat().st_size != pin["bytes"]:
        raise BuildError(f"Missing or incorrect size: {path.name}")
    if sha256(path) != pin["sha256"]:
        raise BuildError(f"SHA256 mismatch: {path.name}")


def validate_output_dir(root: Path, output: Path) -> Path:
    root, output = root.resolve(), output.resolve()
    # Outputs are intentionally constrained to disposable repository trees.
    if not any(output == root / name or (root / name) in output.parents
               for name in ("build", "dist")):
        raise BuildError("Output must be inside this repository's build/ or dist/")
    return output


def enforce_asset_limit(path: Path) -> None:
    if path.stat().st_size >= GITHUB_ASSET_LIMIT:
        raise BuildError(f'{path.name} exceeds the strict <2 GiB release-asset limit. '
                         'CUDA was NOT silently removed. Supply a CPU-only runtime base '
                         'with --without-cuda, or distribute through a different channel.')


def verify_runtime_compression(runtime: Path, workdir: Path) -> None:
    """A pinned ELF can still lack the decompressor required by our format."""
    with tempfile.TemporaryDirectory(prefix='runtime-probe-', dir=workdir) as directory:
        root = Path(directory)
        source = root / 'probe.AppDir'
        source.mkdir()
        (source / 'probe').write_bytes(b'zstd compatibility probe\n')
        squash = root / 'probe.squashfs'
        subprocess.run(['mksquashfs', str(source), str(squash), '-comp', 'zstd', '-b', '1M',
                        '-noappend', '-all-root', '-processors', '4', '-mkfs-time', '0',
                        '-all-time', '0', '-no-progress'], check=True, capture_output=True)
        image = root / 'probe.AppImage'
        with image.open('wb') as target:
            for path in (runtime, squash):
                with path.open('rb') as stream:
                    shutil.copyfileobj(stream, target)
        image.chmod(0o755)
        result = subprocess.run([str(image), '--appimage-extract'], cwd=root,
                                capture_output=True, text=True, timeout=30)
        if result.returncode or not (root / 'squashfs-root/probe').is_file():
            raise BuildError('pinned runtime cannot read zstd SquashFS: ' + result.stderr.strip()
                             + '. Use an independently verified zstd-capable type2 runtime pin, '
                             'or explicitly choose --portable-only. No AppImage was produced.')


def assemble_image(appdir: Path, runtime: Path, output: Path, epoch=0) -> Path:
    if not shutil.which('mksquashfs'):
        raise BuildError('mksquashfs is required (squashfs-tools); no download attempted')
    verify_runtime_compression(runtime, appdir.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    squash = output.with_suffix('.squashfs.partial')
    partial = output.with_suffix('.AppImage.partial')
    try:
        subprocess.run(['mksquashfs', str(appdir), str(squash), '-comp', 'zstd', '-Xcompression-level', '19', '-b', '1M',
                        '-noappend', '-all-root', '-processors', '4', '-mkfs-time', str(epoch),
                        '-all-time', str(epoch), '-no-progress'], check=True)
        with partial.open('wb') as target:
            for source in (runtime, squash):
                with source.open('rb') as stream:
                    shutil.copyfileobj(stream, target, 1024 * 1024)
        enforce_asset_limit(partial)
        partial.chmod(0o755)
        partial.replace(output)
    finally:
        squash.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    return output


def portable_archive(appdir: Path, output: Path, epoch=0) -> Path:
    import gzip
    partial = output.with_suffix('.gz.partial')
    try:
        with partial.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                for source in [appdir, *sorted(appdir.rglob('*'))]:
                    info = archive.gettarinfo(str(source), 'Translitter/' + str(source.relative_to(appdir)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = 'root'
                    info.mtime = epoch
                    info.pax_headers = {}
                    if info.isfile():
                        with source.open('rb') as stream:
                            archive.addfile(info, stream)
                    else:
                        archive.addfile(info)
        enforce_asset_limit(partial)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return output


def main(argv=None) -> int:
    import argparse
    import platform
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-base', type=Path, default=Path('build/appimage-runtime-base'))
    parser.add_argument('--tools-cache', type=Path, default=Path('.local/appimage-tools'))
    parser.add_argument('--output-dir', type=Path, default=Path('dist'))
    parser.add_argument('--version', default='0.2.0-preview.1')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--without-cuda', action='store_true', help='requires a CPU-only torch runtime base; retains Vulkan')
    parser.add_argument('--portable-tar', action='store_true', help='also create a deterministic portable tar.gz')
    parser.add_argument('--portable-only', action='store_true', help='explicit tar.gz fallback; do not claim an AppImage')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    def absolute(path):
        return path if path.is_absolute() else root / path
    try:
        if platform.machine() != 'x86_64':
            raise BuildError('Only x86_64 is supported by the pinned runtime')
        output_dir = validate_output_dir(root, absolute(args.output_dir))
        appdir = prepare_appdir(root, absolute(args.runtime_base), absolute(args.tools_cache),
                               args.version, args.without_cuda)
        print(f'Prepared {appdir}')
        if args.prepare_only:
            return 0
        lock = json.loads((root / 'packaging/tools.lock.json').read_text())
        runtime_pin = lock['tools']['appimage-runtime']
        runtime = absolute(args.tools_cache) / runtime_pin['name']
        verify_pin(runtime, runtime_pin)
        epoch = int(os.environ.get('SOURCE_DATE_EPOCH', '0'))
        variant = '-without-cuda' if args.without_cuda else ''
        stem = f'Translitter-{args.version}-x86_64{variant}'
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        if not args.portable_only:
            outputs.append(assemble_image(appdir, runtime, output_dir / (stem + '.AppImage'), epoch))
        if args.portable_tar or args.portable_only:
            outputs.append(portable_archive(appdir, output_dir / (stem + '.tar.gz'), epoch))
        sums = output_dir / 'SHA256SUMS'
        sums.write_text(''.join(f'{sha256(path)}  {path.name}\n' for path in outputs))
        for path in outputs:
            print(f'{path}: {path.stat().st_size} bytes; SHA256 {sha256(path)}')
        return 0
    except (BuildError, OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as exc:
        parser.exit(1, f'AppImage build failed: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
