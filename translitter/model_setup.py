"""Pinned, explicit, offline-first model installation (stdlib only).

All paths are relative to the caller-selected root, never this module or an
AppImage mount. Importing this module opens no devices and performs no I/O.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import re
from urllib.parse import quote, urlsplit


class ModelSetupError(RuntimeError):
    """Invalid catalog, unavailable assets, or a failed verified install."""


class SetupCancelled(ModelSetupError):
    """The caller cancelled installation; completed files remain valid."""


def _relative(value):
    if (not isinstance(value, str) or not value or '\\' in value
            or '\x00' in value or value.startswith('/')
            or any(p in ('', '.', '..') for p in value.split('/'))
            or ':' in value):
        raise ModelSetupError('Unsafe relative asset path')
    return value


def _https(value):
    if not isinstance(value, str):
        raise ModelSetupError('Invalid HTTPS URL')
    parts = urlsplit(value)
    if (parts.scheme != 'https' or not parts.hostname or parts.username
            or parts.password or parts.fragment or any(c.isspace() for c in value)):
        raise ModelSetupError('Asset URLs and redirects must be HTTPS without credentials')
    return value


@dataclass(frozen=True)
class AssetFile:
    relative_path: str
    url: str
    bytes: int
    sha256: str

    def __post_init__(self):
        _relative(self.relative_path)
        _https(self.url)
        if type(self.bytes) is not int or self.bytes < 0:
            raise ModelSetupError('Asset byte length must be a nonnegative integer')
        if not isinstance(self.sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.sha256):
            raise ModelSetupError('Invalid SHA256')


def _pinned(repo, revision, name):
    if (not isinstance(repo, str) or not re.fullmatch(r'[\w.-]+/[\w.-]+', repo)
            or not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision)):
        raise ModelSetupError('Repository must have an immutable commit revision')
    return f'https://huggingface.co/{repo}/resolve/{revision}/{quote(_relative(name), safe="/")}'


def load_catalog(bundle_root: Path) -> dict[str, tuple[AssetFile, ...]]:
    """Read bundled lockfiles; only the compatible quality GGUF is exposed."""
    try:
        root = Path(bundle_root)
        lock = json.loads((root / 'model-assets.lock.json').read_text())
        if lock['schema'] != 1:
            raise ModelSetupError('Unsupported model lock schema')
        result = {}
        for key, asset in lock['assets'].items():
            directory = _relative(asset['directory'])
            result[key] = tuple(AssetFile(
                directory + '/' + _relative(f['name']),
                _pinned(asset['repo'], asset['revision'], f['name']),
                f['bytes'], f['sha256']) for f in asset['files'])
        quality_path = root / 'quality-model.lock.json'
        if quality_path.exists():
            q = json.loads(quality_path.read_text())
            directory = _relative(q['directory'])
            compatible = q['compatible_source']
            # Validate the compatible URL is pinned to the declared repository/commit.
            prefix = _pinned(compatible['repo'], compatible['revision'], 'placeholder').rsplit('/', 1)[0] + '/'
            if not compatible['url'].startswith(prefix):
                raise ModelSetupError('Unpinned compatible quality source')
            license_file = next(f for f in q['files'] if f['name'] == 'LICENSE.txt')
            result['quality-translategemma-4b'] = (
                AssetFile(directory + '/' + _relative(q['runtime_file']), compatible['url'],
                          compatible['bytes'], compatible['sha256']),
                AssetFile(directory + '/LICENSE.txt', license_file['url'],
                          license_file['bytes'], license_file['sha256']))
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration) as exc:
        raise ModelSetupError('Invalid model lock catalog') from exc


def required_assets(languages=('fi', 'en'), speech_model='whisper-small', quality=False) -> tuple[str, ...]:
    """Baseline is mandatory because the engine initializes FI/EN eagerly."""
    if isinstance(languages, str):
        raise ModelSetupError('Languages must be a sequence of language codes')
    try:
        languages = tuple(languages)
    except TypeError as exc:
        raise ModelSetupError('Languages must be iterable') from exc
    if isinstance(languages, str) or any(lang not in ('fi', 'en', 'ru', 'uk') for lang in languages):
        raise ModelSetupError('Unsupported language')
    if speech_model not in ('whisper-small', 'whisper-turbo'):
        raise ModelSetupError('Unsupported speech model')
    keys = ['translate-en-fi', 'translate-fi-en', 'voice-fi', 'voice-en']
    for lang in ('ru', 'uk'):
        if lang in languages:
            keys.extend((f'translate-en-{lang}', f'translate-{lang}-en', f'voice-{lang}'))
    keys.append(speech_model)
    if quality:
        keys.append('quality-translategemma-4b')
    return tuple(keys)


# Linux/AppImage: directory descriptors + O_NOFOLLOW prevent path substitution
# through symlinks between validation and file creation.
import contextlib
import fcntl
import hashlib
import os
import stat
import time

_CHUNK = 128 * 1024


def _check(cancelled):
    if cancelled():
        raise SetupCancelled('Model setup cancelled')


@contextlib.contextmanager
def _directory(path):
    absolute = Path(os.path.abspath(path))
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for piece in absolute.parts[1:]:
            try:
                os.mkdir(piece, 0o700, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(piece, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


@contextlib.contextmanager
def _parent(root_fd, relative):
    pieces = _relative(relative).split('/')
    fd = os.dup(root_fd)
    try:
        for piece in pieces[:-1]:
            try:
                os.mkdir(piece, 0o700, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(piece, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd, pieces[-1]
    finally:
        os.close(fd)


def _regular(fd):
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ModelSetupError('Expected a regular, unlinked-to-other-paths file')
    return info


def _safe_existing(fd, name):
    try:
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ModelSetupError('Unsafe destination file')


def _verified(fd, name, asset, cancelled):
    try:
        raw = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    except FileNotFoundError:
        return False
    with os.fdopen(raw, 'rb') as stream:
        if _regular(stream.fileno()).st_size != asset.bytes:
            return False
        digest = hashlib.sha256()
        while True:
            _check(cancelled)
            chunk = stream.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest() == asset.sha256


@contextlib.contextmanager
def _locked(root_fd, cancelled):
    raw = os.open('.model-setup.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                  0o600, dir_fd=root_fd)
    try:
        _regular(raw)
        os.fchmod(raw, 0o600)
        while True:
            _check(cancelled)
            try:
                fcntl.flock(raw, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.05)
        yield
    finally:
        os.close(raw)


def _copy_import(source_root, asset, parent_fd, part, cancelled, notify):
    source = Path(source_root) / asset.relative_path
    # A local import never writes to the source tree.
    raw_source = os.open(source, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(raw_source, 'rb') as incoming:
        if not stat.S_ISREG(os.fstat(incoming.fileno()).st_mode):
            raise ModelSetupError('Import source must be a regular file')
        if os.fstat(incoming.fileno()).st_size != asset.bytes:
            raise ModelSetupError('Import size mismatch')
        raw = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                      0o600, dir_fd=parent_fd)
        try:
            _regular(raw)
            os.ftruncate(raw, 0)
            with os.fdopen(raw, 'wb', closefd=False) as outgoing:
                count = 0
                while True:
                    _check(cancelled)
                    chunk = incoming.read(min(_CHUNK, asset.bytes - count + 1))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > asset.bytes:
                        raise ModelSetupError('Import exceeded declared length')
                    outgoing.write(chunk)
                    notify(count)
                outgoing.flush()
                os.fsync(raw)
        finally:
            os.close(raw)


from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class _HTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https(newurl)
        if req.get_method() != 'GET' or code not in (301, 302, 303, 307, 308):
            raise ModelSetupError('Unsupported model redirect')
        # Never forward cookies, Authorization, proxy credentials or host headers.
        allowed = {name: value for name, value in req.header_items()
                   if name.lower() in ('range', 'accept-encoding', 'user-agent')}
        return Request(newurl, headers=allowed, method='GET')


def _download(asset, parent_fd, part, cancelled, notify):
    _check(cancelled)
    try:
        size = os.stat(part, dir_fd=parent_fd, follow_symlinks=False).st_size
    except FileNotFoundError:
        size = 0
    if size > asset.bytes:
        raise ModelSetupError('Partial file exceeds declared size; remove it before retrying')
    if size == asset.bytes and _verified(parent_fd, part, asset, cancelled):
        notify(size)
        return
    if size == asset.bytes and size:
        os.unlink(part, dir_fd=parent_fd)
        size = 0
    headers = {'Accept-Encoding': 'identity', 'User-Agent': 'Translitter-Model-Setup/1'}
    if size:
        headers['Range'] = f'bytes={size}-'
    request = Request(_https(asset.url), headers=headers, method='GET')
    # No ambient proxies, cookie jar, auth handlers, or credentials.
    opener = build_opener(ProxyHandler({}), _HTTPSRedirect())
    with opener.open(request, timeout=30) as response:
        _https(response.geturl())
        remaining = asset.bytes - size
        content_length = response.headers.get('Content-Length', '')
        encoding = response.headers.get('Content-Encoding', 'identity')
        if (not re.fullmatch(r'0|[1-9][0-9]*', content_length)
                or int(content_length) != remaining or encoding != 'identity'):
            raise ModelSetupError('Download length or encoding mismatch')
        content_range = response.headers.get('Content-Range')
        if size:
            expected = f'bytes {size}-{asset.bytes - 1}/{asset.bytes}'
            if response.status != 206 or content_range != expected:
                raise ModelSetupError('Resume Content-Range mismatch')
        elif response.status != 200 or content_range is not None:
            raise ModelSetupError('Unexpected full download status or range')
        _check(cancelled)
        raw = os.open(part, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                      0o600, dir_fd=parent_fd)
        try:
            if _regular(raw).st_size != size:
                raise ModelSetupError('Partial file changed during download')
            os.lseek(raw, size, os.SEEK_SET)
            with os.fdopen(raw, 'ab', closefd=False) as outgoing:
                count = size
                # HTTPResponse.read1 returns after one underlying read instead
                # of waiting to fill 128 KiB while a server trickles bytes.
                # Check cancellation between packets; socket inactivity is
                # separately bounded by the opener's 30-second timeout.
                read = getattr(response, 'read1', response.read)
                while True:
                    _check(cancelled)
                    chunk = read(min(_CHUNK, asset.bytes - count + 1))
                    if not chunk:
                        break
                    if count + len(chunk) > asset.bytes:
                        raise ModelSetupError('Download exceeds declared length')
                    outgoing.write(chunk)
                    count += len(chunk)
                    notify(count)
                outgoing.flush()
                os.fsync(raw)
                if count != asset.bytes:
                    raise ModelSetupError('Download interrupted; retry to resume')
        finally:
            os.close(raw)


def install_assets(destination_root, catalog, keys, *, source_root=None,
                   download=False, cancelled=lambda: False,
                   progress=lambda done, total, message: None) -> None:
    """Verify/import selected assets. Network requires explicit ``download=True``.

    Progress is cumulative installed bytes, total selected bytes, relative path.
    Files commit independently using atomic replace; completed files survive errors.
    The root is supplied by the caller and must be outside the read-only bundle.
    """
    try:
        if type(download) is not bool or isinstance(keys, str):
            raise ModelSetupError('Invalid install options')
        selected = {}
        for key in keys:
            if key not in catalog or not isinstance(catalog[key], (tuple, list)):
                raise ModelSetupError('Unknown asset key or invalid asset group')
            for asset in catalog[key]:
                if not isinstance(asset, AssetFile):
                    raise ModelSetupError('Catalog entries must be AssetFile objects')
                asset.__post_init__()
                previous = selected.setdefault(asset.relative_path, asset)
                if previous != asset:
                    raise ModelSetupError('Conflicting asset metadata')
        names = set(selected)
        if any(name.split('/')[0] == '.model-setup.lock' for name in names):
            raise ModelSetupError('Asset path conflicts with private install lock')
        if any(name + '.part' in names for name in names):
            raise ModelSetupError('Asset path conflicts with partial file')
        total = sum(asset.bytes for asset in selected.values())
        done = 0
        _check(cancelled)
        with _directory(destination_root) as root_fd, _locked(root_fd, cancelled):
            for asset in selected.values():
                _check(cancelled)
                with _parent(root_fd, asset.relative_path) as (fd, name):
                    part = name + '.part'
                    _safe_existing(fd, name)
                    _safe_existing(fd, part)
                    progress(done, total, asset.relative_path)
                    if not _verified(fd, name, asset, cancelled):
                        network = source_root is None and download
                        if source_root is None and not download:
                            raise ModelSetupError('A local source or explicit download is required')
                        try:
                            notify = lambda n: progress(done + n, total, asset.relative_path)
                            if network:
                                _download(asset, fd, part, cancelled, notify)
                            else:
                                _copy_import(source_root, asset, fd, part, cancelled, notify)
                            if not _verified(fd, part, asset, cancelled):
                                os.unlink(part, dir_fd=fd)
                                raise ModelSetupError('Asset checksum or size mismatch')
                            _check(cancelled)
                            _safe_existing(fd, name)
                            os.replace(part, name, src_dir_fd=fd, dst_dir_fd=fd)
                            os.fsync(fd)
                        except BaseException:
                            # Network prefixes are safe to resume only because every
                            # eventual commit revalidates the complete pinned digest.
                            if not network:
                                try:
                                    os.unlink(part, dir_fd=fd)
                                except FileNotFoundError:
                                    pass
                            raise
                    done += asset.bytes
                    progress(done, total, asset.relative_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelSetupError('Model installation failed: ' + str(exc)) from exc
