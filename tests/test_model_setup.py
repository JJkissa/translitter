"""Offline tests: no real model weights or network access."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import hashlib
import json

import pytest


def api():
    from translitter import model_setup
    return model_setup


def test_catalog_maps_real_pinned_files_and_only_compatible_quality():
    m = api()
    root = Path(__file__).resolve().parents[1]
    catalog = m.load_catalog(root)
    lock = json.loads((root / 'model-assets.lock.json').read_text())
    voice = catalog['voice-fi'][1]
    original = lock['assets']['voice-fi']
    assert voice.relative_path == 'models/voice-fi/' + original['files'][1]['name']
    assert voice.url == f"https://huggingface.co/{original['repo']}/resolve/{original['revision']}/{original['files'][1]['name']}"
    assert voice.bytes == original['files'][1]['bytes']
    with pytest.raises(FrozenInstanceError):
        voice.bytes = 0
    q = json.loads((root / 'quality-model.lock.json').read_text())
    quality = catalog['quality-translategemma-4b']
    assert {a.relative_path for a in quality} == {
        q['directory'] + '/' + q['runtime_file'], q['directory'] + '/LICENSE.txt'}
    runtime = next(a for a in quality if a.relative_path.endswith('.gguf'))
    assert (runtime.url, runtime.bytes, runtime.sha256) == (
        q['compatible_source']['url'], q['compatible_source']['bytes'], q['compatible_source']['sha256'])


def test_requirements_always_keep_baseline_and_select_one_whisper():
    m = api()
    assert set(m.required_assets(('ru', 'uk'), 'whisper-turbo', True)) == {
        'translate-en-fi', 'translate-fi-en', 'voice-fi', 'voice-en',
        'translate-en-ru', 'translate-ru-en', 'voice-ru',
        'translate-en-uk', 'translate-uk-en', 'voice-uk',
        'whisper-turbo', 'quality-translategemma-4b'}
    assert 'quality-translategemma-4b' not in m.required_assets()
    assert m.required_assets().count('whisper-small') == 1
    for args in [(('xx',), 'whisper-small'), (('fi',), 'unknown')]:
        with pytest.raises(m.ModelSetupError):
            m.required_assets(*args)


def one_asset(data=b'model bytes'):
    m = api()
    return m.AssetFile('models/test/model.bin', 'https://example.org/model',
                       len(data), hashlib.sha256(data).hexdigest())


def test_import_is_verified_atomic_and_skip_existing_needs_no_source(tmp_path):
    m = api()
    a = one_asset()
    source, dest = tmp_path / 'source', tmp_path / 'destination'
    f = source / a.relative_path
    f.parent.mkdir(parents=True)
    f.write_bytes(b'model bytes')
    events = []
    m.install_assets(dest, {'test': (a,)}, ['test'], source_root=source,
                     progress=lambda *event: events.append(event))
    target = dest / a.relative_path
    assert target.read_bytes() == b'model bytes'
    inode = target.stat().st_ino
    m.install_assets(dest, {'test': (a,)}, ['test'])
    assert target.stat().st_ino == inode
    assert events[-1][:2] == (a.bytes, a.bytes)
    assert not list(dest.rglob('*.part'))
    assert (dest / '.model-setup.lock').stat().st_mode & 0o777 == 0o600


def test_no_implicit_network_and_corrupt_import_preserves_destination(tmp_path, monkeypatch):
    m = api()
    monkeypatch.setattr('urllib.request.OpenerDirector.open', lambda *a, **k: pytest.fail('network'))
    a = one_asset()
    dest = tmp_path / 'dest'
    with pytest.raises(m.ModelSetupError):
        m.install_assets(dest, {'x': (a,)}, ['x'])
    target = dest / a.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'previous contents')
    source = tmp_path / 'source'
    f = source / a.relative_path
    f.parent.mkdir(parents=True)
    f.write_bytes(b'wrong bytes')
    with pytest.raises(m.ModelSetupError):
        m.install_assets(dest, {'x': (a,)}, ['x'], source_root=source)
    assert target.read_bytes() == b'previous contents'
    assert not list(dest.rglob('*.part'))


@pytest.mark.parametrize('path', ['/etc/passwd', '../outside', 'models/../outside',
                                  'models//x', 'models/./x', 'C:/x', 'models\\x'])
def test_unsafe_relative_paths_rejected(path):
    m = api()
    with pytest.raises(m.ModelSetupError):
        m.AssetFile(path, 'https://example.org/x', 1, '0' * 64)


@pytest.mark.parametrize('size', [-1, True, '1', 1.5])
def test_bad_metadata_lengths_rejected(size):
    m = api()
    with pytest.raises(m.ModelSetupError):
        m.AssetFile('models/x', 'https://example.org/x', size, '0' * 64)


@pytest.mark.parametrize('kind', ['root', 'directory', 'target', 'part', 'lock'])
def test_symlink_destination_components_are_never_followed(tmp_path, kind):
    m = api()
    a = one_asset()
    dest, outside = tmp_path / 'dest', tmp_path / 'outside'
    outside.mkdir()
    dest.mkdir()
    if kind == 'root':
        dest.rmdir()
        dest.symlink_to(outside, target_is_directory=True)
    elif kind == 'directory':
        (dest / 'models').symlink_to(outside, target_is_directory=True)
    elif kind == 'lock':
        (dest / '.model-setup.lock').symlink_to(outside / 'lock')
    else:
        t = dest / a.relative_path
        t.parent.mkdir(parents=True)
        (t if kind == 'target' else t.with_name(t.name + '.part')).symlink_to(outside / 'file')
    with pytest.raises(m.ModelSetupError):
        m.install_assets(dest, {'x': (a,)}, ['x'])
    assert list(outside.iterdir()) == []


def test_cancellation_preserves_completed_install_and_stops_import(tmp_path):
    m = api()
    a = one_asset(b'x' * 200000)
    source, dest = tmp_path / 'source', tmp_path / 'dest'
    f = source / a.relative_path
    f.parent.mkdir(parents=True)
    f.write_bytes(b'x' * 200000)
    events = []
    with pytest.raises(m.SetupCancelled):
        m.install_assets(dest, {'x': (a,)}, ['x'], source_root=source,
                         cancelled=lambda: len(events) > 1,
                         progress=lambda *args: events.append(args))
    assert not (dest / a.relative_path).exists()
    assert not list(dest.rglob('*.part'))


class Response:
    def __init__(self, data, status=200, headers=None, url='https://cdn.example.org/signed'):
        import io
        self.stream = io.BytesIO(data)
        self.status = status
        self.headers = headers if headers is not None else {'Content-Length': str(len(data))}
        self.url = url

    def geturl(self):
        return self.url

    def read(self, size):
        assert 0 < size <= 128 * 1024
        return self.stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()


def fake_network(monkeypatch, response):
    requests = []
    def open_fake(self, request, **kwargs):
        requests.append(request)
        return response
    monkeypatch.setattr('urllib.request.OpenerDirector.open', open_fake)
    return requests


def test_download_explicit_and_https_only(tmp_path, monkeypatch):
    m = api()
    a = one_asset()
    requests = fake_network(monkeypatch, Response(b'model bytes'))
    m.install_assets(tmp_path, {'x': (a,)}, ['x'], download=True)
    assert (tmp_path / a.relative_path).read_bytes() == b'model bytes'
    assert len(requests) == 1
    assert requests[0].get_header('Accept-encoding') == 'identity'


def test_packet_reads_allow_cancellation_without_waiting_for_a_full_chunk(tmp_path, monkeypatch):
    from threading import Event
    m=api()
    a=one_asset()
    stop=Event()
    class PacketResponse(Response):
        def read(self,size):
            raise AssertionError('A buffered read can wait indefinitely for a full chunk on a trickle')
        def read1(self,size):
            return self.stream.read(min(size,1))
    response=PacketResponse(b'model bytes')
    fake_network(monkeypatch,response)
    def progress(done,total,message):
        if done:
            stop.set()
    with pytest.raises(m.SetupCancelled):
        m.install_assets(tmp_path,{'x':(a,)},['x'],download=True,cancelled=stop.is_set,progress=progress)
    assert (tmp_path/(a.relative_path+'.part')).read_bytes()==b'm'
    assert response.stream.closed


@pytest.mark.parametrize('offset', [0, 5, 11])
def test_download_resumes_verified_prefix_or_complete_part(tmp_path, monkeypatch, offset):
    m = api()
    a = one_asset()
    part = tmp_path / (a.relative_path + '.part')
    part.parent.mkdir(parents=True)
    part.write_bytes(b'model bytes'[:offset])
    response = Response(b'model bytes'[offset:], 206 if offset else 200,
                        {'Content-Length': str(a.bytes - offset), **(
                            {'Content-Range': f'bytes {offset}-{a.bytes-1}/{a.bytes}'} if offset else {})})
    requests = fake_network(monkeypatch, response)
    m.install_assets(tmp_path, {'x': (a,)}, ['x'], download=True)
    assert (tmp_path / a.relative_path).read_bytes() == b'model bytes'
    assert not part.exists()
    if 0 < offset < a.bytes:
        assert requests[0].get_header('Range') == f'bytes={offset}-'
    if offset == a.bytes:
        assert requests == []


@pytest.mark.parametrize('status,headers,data,url', [
    (206, {'Content-Length': '6', 'Content-Range': 'bytes 4-9/11'}, b' bytes', 'https://cdn.example.org/x'),
    (206, {'Content-Length': '6', 'Content-Range': 'bytes 5-10/12'}, b' bytes', 'https://cdn.example.org/x'),
    (200, {'Content-Length': '6'}, b' bytes', 'https://cdn.example.org/x'),
    (206, {'Content-Length': '9', 'Content-Range': 'bytes 5-10/11'}, b' bytes', 'https://cdn.example.org/x'),
    (206, {'Content-Length': '6', 'Content-Range': 'bytes 5-10/11'}, b' bytes', 'http://cdn.example.org/x'),
])
def test_resume_mismatch_never_appends_or_replaces(tmp_path, monkeypatch, status, headers, data, url):
    m = api()
    a = one_asset()
    part = tmp_path / (a.relative_path + '.part')
    part.parent.mkdir(parents=True)
    part.write_bytes(b'model')
    fake_network(monkeypatch, Response(data, status, headers, url))
    with pytest.raises(m.ModelSetupError):
        m.install_assets(tmp_path, {'x': (a,)}, ['x'], download=True)
    assert not (tmp_path / a.relative_path).exists()
    assert part.read_bytes() == b'model'


@pytest.mark.parametrize('data,headers', [
    (b'wrong bytes', {'Content-Length': '11'}),
    (b'model bytesEXTRA', {'Content-Length': '11'}),
    (b'model', {'Content-Length': '11'}),
    (b'model bytes', {}),
    (b'model bytes', {'Content-Length': '11', 'Content-Encoding': 'gzip'}),
])
def test_bad_download_never_commits(tmp_path, monkeypatch, data, headers):
    m = api()
    a = one_asset()
    fake_network(monkeypatch, Response(data, headers=headers))
    with pytest.raises(m.ModelSetupError):
        m.install_assets(tmp_path, {'x': (a,)}, ['x'], download=True)
    assert not (tmp_path / a.relative_path).exists()


def test_cancelled_download_resumes_without_losing_prefix(tmp_path, monkeypatch):
    m = api()
    data = b'x' * 200000
    a = one_asset(data)
    events = []
    fake_network(monkeypatch, Response(data))
    with pytest.raises(m.SetupCancelled):
        m.install_assets(tmp_path, {'x': (a,)}, ['x'], download=True,
                         cancelled=lambda: len(events) > 1,
                         progress=lambda *args: events.append(args))
    part = tmp_path / (a.relative_path + '.part')
    offset = part.stat().st_size
    assert 0 < offset < len(data)
    requests = fake_network(monkeypatch, Response(data[offset:], 206, {
        'Content-Length': str(len(data)-offset),
        'Content-Range': f'bytes {offset}-{len(data)-1}/{len(data)}'}))
    m.install_assets(tmp_path, {'x': (a,)}, ['x'], download=True)
    assert requests[0].get_header('Range') == f'bytes={offset}-'
    assert (tmp_path / a.relative_path).read_bytes() == data


def test_redirects_validate_https_and_do_not_forward_credentials():
    from urllib.request import Request
    m = api()
    redirect = m._HTTPSRedirect()
    request = Request('https://example.org/model', headers={
        'Authorization': 'secret', 'Cookie': 'secret', 'Range': 'bytes=5-',
        'Accept-Encoding': 'identity'})
    redirected = redirect.redirect_request(request, None, 302, 'Found', {}, 'https://cdn.example.org/file?signature=x')
    assert redirected.get_header('Authorization') is None
    assert redirected.get_header('Cookie') is None
    assert redirected.get_header('Range') == 'bytes=5-'
    for url in ['http://example.org/x', 'https://user:secret@example.org/x', 'file:///etc/passwd']:
        with pytest.raises(m.ModelSetupError):
            redirect.redirect_request(request, None, 302, 'Found', {}, url)


def test_requirement_iterables_and_invalid_types_are_consistent():
    m = api()
    assert m.required_assets(iter(('ru', 'uk'))) == m.required_assets(('ru', 'uk'))
    with pytest.raises(m.ModelSetupError):
        m.required_assets(None)


def test_catalog_cannot_overwrite_install_lock(tmp_path):
    m = api()
    a = m.AssetFile('.model-setup.lock', 'https://example.org/x', 0, hashlib.sha256(b'').hexdigest())
    with pytest.raises(m.ModelSetupError):
        m.install_assets(tmp_path, {'x': (a,)}, ['x'])


def test_fifo_source_is_rejected_without_blocking(tmp_path, monkeypatch):
    import os
    m = api()
    a = one_asset()
    source = tmp_path / 'source'
    f = source / a.relative_path
    f.parent.mkdir(parents=True)
    os.mkfifo(f)
    # Prove open is nonblocking without hanging an unattended test.
    original = Path.open
    def reject_blocking_fifo(self, *args, **kwargs):
        if self == f:
            pytest.fail('FIFO opened using blocking Path.open')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', reject_blocking_fifo)
    with pytest.raises(m.ModelSetupError):
        m.install_assets(tmp_path / 'dest', {'x': (a,)}, ['x'], source_root=source)


def test_completed_files_survive_later_failure(tmp_path, monkeypatch):
    m = api()
    good = one_asset()
    bad = m.AssetFile('models/second/file', good.url, 1, hashlib.sha256(b'z').hexdigest())
    calls = []
    def serve(self, request, **kwargs):
        calls.append(request)
        return Response(b'model bytes') if len(calls) == 1 else Response(b'x')
    monkeypatch.setattr('urllib.request.OpenerDirector.open', serve)
    with pytest.raises(m.ModelSetupError):
        m.install_assets(tmp_path, {'x': (good, bad)}, ['x'], download=True)
    assert (tmp_path / good.relative_path).read_bytes() == b'model bytes'
    assert not (tmp_path / bad.relative_path).exists()


@pytest.mark.parametrize('mutation', ['traversal', 'unpinned', 'badsize', 'badsha'])
def test_malformed_catalog_rejected(tmp_path, mutation):
    m = api()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / 'model-assets.lock.json').read_text())
    asset = lock['assets']['whisper-small']
    if mutation == 'traversal':
        asset['files'][0]['name'] = '../outside'
    elif mutation == 'unpinned':
        asset['revision'] = 'main'
    elif mutation == 'badsize':
        asset['files'][0]['bytes'] = True
    else:
        asset['files'][0]['sha256'] = 'bad'
    (tmp_path / 'model-assets.lock.json').write_text(json.dumps(lock))
    with pytest.raises(m.ModelSetupError):
        m.load_catalog(tmp_path)
