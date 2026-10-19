"""Offline AppImage builder contracts; no model/hardware access."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def builder():
    path = ROOT / "packaging/build_appimage.py"
    if not path.exists():
        raise AssertionError("offline AppImage builder is not implemented")
    spec = importlib.util.spec_from_file_location("appimage_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuilderSafetyTests(unittest.TestCase):
    def test_pins_fail_closed_and_output_cannot_replace_source(self):
        b = builder()
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            root = Path(directory)
            payload = root / "runtime"
            payload.write_bytes(b"verified bytes")
            pin = {"bytes": 14, "sha256": b.sha256(payload)}
            b.verify_pin(payload, pin)
            payload.write_bytes(b"corrupt bytes!")
            with self.assertRaisesRegex(b.BuildError, "(size|SHA256)"):
                b.verify_pin(payload, pin)
            for unsafe in (ROOT, ROOT / "translitter", ROOT / "packaging", Path.home(), Path("/")):
                with self.assertRaises(b.BuildError):
                    b.validate_output_dir(ROOT, unsafe)
            self.assertEqual(b.validate_output_dir(ROOT, ROOT / "dist"), ROOT / "dist")


class PreparationTests(unittest.TestCase):
    def test_verified_archives_and_resources_are_separate_from_user_data(self):
        import io
        import tarfile
        import shutil
        b = builder()
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            root = Path(directory)
            (root / "packaging").mkdir()
            (root / 'packaging/licenses').mkdir()
            license_file=root / 'packaging/licenses/type2-runtime.txt'
            license_file.write_text('MIT License fixture\n')
            (root / 'packaging/licenses/manifest.json').write_text(json.dumps([
                {'file':license_file.name,'sha256':b.sha256(license_file),'source':'https://upstream.example/LICENSE'}]))
            for name in ("AppRun", "launch.py"):
                self.assertTrue((ROOT / 'packaging' / name).is_file(), 'launch template not implemented: ' + name)
                shutil.copyfile(ROOT / "packaging" / name, root / "packaging" / name)
            (root / "translitter").mkdir()
            (root / "translitter/__init__.py").write_text("")
            (root / "translitter/packaged.py").write_text("print('packaged')")
            (root / "assets").mkdir()
            (root / "assets/translitter.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            for name in ("model-assets.lock.json", "quality-model.lock.json"):
                (root / name).write_text('{}')
            (root / "models").mkdir()
            (root / "models/private.gguf").write_bytes(b"NEVER BUNDLE")
            base = root / "base"
            site = base / "usr/lib/python3.12/site-packages"
            site.mkdir(parents=True)
            (site / 'certifi').mkdir()
            (site / 'certifi/cacert.pem').write_text('Public CA bundle fixture\n')
            (site / "sitecustomize.py").write_text("raise RuntimeError('unsafe')")
            (site / "foreign").symlink_to(root / "models", target_is_directory=True)
            (site / "example.pyc").write_bytes(b"bytecode")
            (site / 'ctranslate2/models').mkdir(parents=True)
            (site / 'ctranslate2/models/__init__.py').write_text('# Required Python model API, not weights\n')
            (site / 'faster_whisper/assets').mkdir(parents=True)
            (site / 'faster_whisper/assets/silero_vad_v6.onnx').write_bytes(b'intrinsic VAD fixture')
            (site / "nvidia/cublas/lib").mkdir(parents=True)
            (site / "nvidia/cublas/lib/large.so").write_bytes(b"CUDA")
            (base / "usr/bin").mkdir()
            (base / "usr/bin/python3.12").write_bytes(b"test-only interpreter fixture")
            (base / "runtime-base.ready.json").write_text(str(Path.home()))
            cache = root / "cache"
            cache.mkdir()
            runtime = cache / "runtime-x86_64"
            runtime.write_bytes(b"pinned type2 runtime fixture")
            tools = {"appimage-runtime": {"name": runtime.name, "bytes": runtime.stat().st_size, "sha256": b.sha256(runtime)}}
            for key in ("llama-cpu", "llama-vulkan"):
                archive = cache / (key + '.tar.gz')
                with tarfile.open(archive, 'w:gz') as tar:
                    for name, data in (("llama-server", b"fixture server"), ("libggml.so", b"fixture library"), ("llama-cli", b"exclude CLI"), ("LICENSE", b"MIT License fixture")):
                        info = tarfile.TarInfo("llama-test/" + name)
                        info.size = len(data)
                        info.mode = 0o755
                        tar.addfile(info, io.BytesIO(data))
                tools[key] = {"name": archive.name, "bytes": archive.stat().st_size, "sha256": b.sha256(archive), "repository": "ggml-org/llama.cpp", "revision": "a" * 40}
            (root / "packaging/tools.lock.json").write_text(json.dumps({"schema": 1, "architecture": "x86_64", "tools": tools}))
            appdir = b.prepare_appdir(root, base, cache, "0.2.0-preview.1", without_cuda=True)
            app = appdir / "usr/lib/translitter/app"
            self.assertTrue((app / "model-assets.lock.json").is_file())
            self.assertTrue((app / "translitter/packaged.py").is_file())
            self.assertFalse((app / "models").exists())
            self.assertTrue((appdir / 'usr/lib/python3.12/site-packages/ctranslate2/models/__init__.py').is_file())
            self.assertTrue((appdir / 'usr/lib/python3.12/site-packages/faster_whisper/assets/silero_vad_v6.onnx').is_file())
            self.assertTrue((app / "runtime/cpu/llama-server").is_file())
            self.assertFalse((app / "runtime/cpu/llama-cli").exists())
            self.assertTrue((appdir / ".DirIcon").is_file())
            self.assertTrue((appdir / "translitter.desktop").is_file())
            self.assertFalse(list(appdir.rglob('*.pyc')))
            self.assertFalse(list(appdir.rglob('sitecustomize*')))
            self.assertFalse((appdir / "usr/lib/python3.12/site-packages/foreign").exists())
            self.assertFalse((appdir / "usr/lib/python3.12/site-packages/nvidia").exists())
            meta = json.loads((app / "appimage-build.json").read_text())
            self.assertEqual(meta['minimum_glibc'], '2.39')
            self.assertFalse(meta['cuda'])
            self.assertNotIn(str(Path.home()), json.dumps(meta))
            self.assertTrue((appdir / "usr/share/doc/translitter/third-party.json").is_file())
            self.assertEqual((appdir / 'usr/share/translitter/ca-certificates.crt').read_text(),'Public CA bundle fixture\n')
            self.assertTrue((appdir / 'usr/lib/python3.12/site-packages/certifi/cacert.pem').is_file())
            self.assertTrue((appdir / 'usr/share/doc/translitter/runtime-licenses/type2-runtime.txt').is_file())
            self.assertFalse(list(appdir.rglob('runtime-base.ready.json')))
            self.assertTrue(b.excluded(Path('lib/python3.12/site-packages/PySide6/Qt/qml/QtWayland/plugin.so')))
            self.assertTrue(b.excluded(Path('lib/python3.12/site-packages/PySide6/Qt/plugins/sqldrivers/libqsqlpsql.so')))
            torchlib = site / 'torch/lib/libtorch_cuda.so'
            torchlib.parent.mkdir(parents=True)
            torchlib.write_bytes(b'CUDA-enabled torch fixture')
            with self.assertRaisesRegex(b.BuildError, 'CPU-only PyTorch runtime base'):
                b.prepare_appdir(root, base, cache, '0.2.0-preview.1', without_cuda=True)


class LaunchAndAssemblyTests(unittest.TestCase):
    def test_bootstrap_discards_foreign_paths_and_restores_only_bundle(self):
        import os
        import sys
        spec = importlib.util.spec_from_file_location('bundle_launch', ROOT / 'packaging/launch.py')
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as directory:
            root = Path(directory)
            site = root / 'usr/lib/python3.12/site-packages'
            site.mkdir(parents=True)
            paths, env, bytecode = sys.path[:], dict(os.environ), sys.dont_write_bytecode
            try:
                sys.path[:] = ['', '/untrusted', str(site.parent), str(site / 'untrusted-nested')]
                os.environ.update(PYTHONPATH='/untrusted', LD_PRELOAD='/untrusted', QT_PLUGIN_PATH='/untrusted')
                launch.configure(root)
                self.assertEqual(sys.path, [str(root / 'usr/lib/translitter/app'), str(site), str(site.parent)])
                self.assertNotIn('LD_PRELOAD', os.environ)
                self.assertNotIn('PYTHONPATH', os.environ)
                self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
                self.assertEqual(os.environ['QT_PLUGIN_PATH'], str(site / 'PySide6/Qt/plugins'))
            finally:
                sys.path[:] = paths
                os.environ.clear()
                os.environ.update(env)
                sys.dont_write_bytecode = bytecode
        shell = (ROOT / 'packaging/AppRun').read_text()
        self.assertIn('python3.12" -I', shell)
        self.assertIn('unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_PRELOAD LD_LIBRARY_PATH', shell)

    def test_tiny_squashfs_assembly_is_reproducible_and_size_guard_fails_closed(self):
        import shutil
        import subprocess
        b = builder()
        self.assertTrue(callable(getattr(b, 'assemble_image', None)), 'image assembler not implemented')
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as directory:
            root = Path(directory)
            huge = root / 'oversize'
            with huge.open('wb') as stream:
                stream.truncate(b.GITHUB_ASSET_LIMIT)
            with self.assertRaisesRegex(b.BuildError, '2 GiB'):
                b.enforce_asset_limit(huge)
            runtime = ROOT / '.local/appimage-tools/runtime-x86_64'
            if not shutil.which('mksquashfs') or not runtime.is_file():
                self.skipTest('optional local type2 runtime/mksquashfs unavailable')
            appdir = root / 'tiny.AppDir'
            appdir.mkdir()
            (appdir / 'AppRun').write_text('#!/bin/sh\nprintf tiny-test\\n\n')
            (appdir / 'AppRun').chmod(0o755)
            one, two = root / 'one.AppImage', root / 'two.AppImage'
            b.assemble_image(appdir, runtime, one, epoch=0)
            b.assemble_image(appdir, runtime, two, epoch=0)
            self.assertEqual(b.sha256(one), b.sha256(two))
            result = subprocess.run([str(one), '--appimage-extract'], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / 'squashfs-root/AppRun').read_bytes(), (appdir / 'AppRun').read_bytes())

    def test_driver_and_glibc_excluded_but_cuda_runtime_libraries_allowed(self):
        b = builder()
        for name in ('libc.so.6', 'libm.so.6', 'ld-linux-x86-64.so.2'):
            self.assertTrue(b.GLIBC.match(name))
        for name in ('libcuda.so.1', 'libnvidia-ml.so.1', 'iris_dri.so'):
            self.assertTrue(b.DRIVER.match(name))
        for name in ('libcudart.so.12', 'libcublas.so.12', 'libvulkan.so.1'):
            self.assertFalse(b.DRIVER.match(name))


if __name__ == "__main__":
    unittest.main()
