"""Packaged entrypoint keeps setup/diagnostics headless and downloads explicit."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def test_packaged_defaults_do_not_assume_cuda_or_installed_models():
    from translitter.packaged import build_parser
    args=build_parser().parse_args([])
    assert args.device is None and args.model is None
    assert not args.download and not args.accept_model_licenses
    assert not args.install_models


def test_installer_requires_terms_before_any_network_or_copy(tmp_path,monkeypatch,capsys):
    from translitter import packaged
    calls=[]
    monkeypatch.setattr(packaged,'install_assets',lambda *a,**k:calls.append((a,k)))
    assert packaged.main(['--data-dir',str(tmp_path/'data'),'--install-models','--download'])==2
    assert not calls
    assert 'license' in capsys.readouterr().err.lower()


def test_invalid_screenshot_is_rejected_before_creating_any_runtime(tmp_path,monkeypatch,capsys):
    from translitter import packaged
    def forbidden(*args):
        raise AssertionError('Screenshot arguments must be checked before runtime preparation')
    monkeypatch.setattr(packaged,'load_catalog',forbidden)
    assert packaged.main(['--data-dir',str(tmp_path/'data'),'--screenshot',str(tmp_path/'invalid.jpg')])==2
    assert 'PNG' in capsys.readouterr().err
    assert not (tmp_path/'data').exists()


def test_diagnostics_never_loads_models_or_starts_a_window(tmp_path,monkeypatch,capsys):
    from translitter import packaged
    monkeypatch.setattr(packaged,'detect_cuda',lambda:False)
    assert packaged.main(['--data-dir',str(tmp_path/'data'),'--diagnose'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['data_root']==str(tmp_path/'data')
    assert report['models_present'] is False
    assert report['cuda_available'] is False
    assert report['microphones_opened'] is False
    assert not (tmp_path/'data/models').exists()


@pytest.mark.parametrize('failure',[False,True])
def test_synthetic_self_test_closes_models_without_capture(tmp_path,monkeypatch,capsys,failure):
    from translitter import packaged,engine
    closed=[]
    class Model:
        def __init__(self,*a,**k): pass
        def load(self): pass
        def interpret(self,samples,route,generation):
            if failure:
                raise RuntimeError('injected model failure')
            return SimpleNamespace(original='Hei',translation='Hello',audio=np.ones(160),rate=16000,elapsed=.1)
        def close(self): closed.append(True)
    monkeypatch.setattr(engine,'ModelEngine',Model)
    monkeypatch.setattr(packaged,'synthetic_input',lambda *a:np.ones(8000,dtype=np.float32),raising=False)
    result=packaged.main(['--data-dir',str(tmp_path/'data'),'--self-test'])
    assert result==(2 if failure else 0)
    assert closed==[True]
    output=capsys.readouterr()
    if not failure:
        report=json.loads(output.out)
        assert report['microphones_opened'] is False and report['audio_played'] is False
        assert report['translation']=='Hello'


def test_setup_screenshot_is_a_real_native_window_without_installation(qtbot,tmp_path,monkeypatch):
    from translitter import packaged,setup_ui
    monkeypatch.setattr(packaged,'detect_cuda',lambda:False)
    monkeypatch.setattr(setup_ui,'cuda_is_available',lambda:False)
    target=tmp_path/'guide/setup.png'
    assert packaged.main(['--data-dir',str(tmp_path/'data'),'--setup','--screenshot',str(target)])==0
    from PySide6.QtGui import QImage
    image=QImage(str(target))
    assert not image.isNull() and image.width()>=700 and image.height()>=500
    assert not (tmp_path/'data/models').exists()
