"""Standalone application entrypoint: explicit setup, persistent models, offline use."""
import argparse
import json
from pathlib import Path
import platform
import sys

from .model_setup import load_catalog, required_assets, install_assets, ModelSetupError
from .package_paths import AppPaths


def build_parser():
    parser=argparse.ArgumentParser(description='Translitter AppImage — offline two-headset interpreter')
    parser.add_argument('--data-dir',type=Path,help='Writable application data folder; contains models/ and settings.json')
    parser.add_argument('--setup',action='store_true',help='Open model selection/download/import setup')
    parser.add_argument('--diagnose',action='store_true',help='Print runtime/model diagnostics without loading models or opening audio')
    parser.add_argument('--install-models',action='store_true',help='Install selected pinned weights, without starting the GUI')
    parser.add_argument('--download',action='store_true',help='Explicitly allow HTTPS model downloads during --install-models')
    parser.add_argument('--from-root',type=Path,help='Offline import from a folder containing models/')
    parser.add_argument('--accept-model-licenses',action='store_true',help='Confirm review of selected publisher model/voice terms')
    parser.add_argument('--languages',help='Comma-separated language codes: fi,en,ru,uk; FI/EN baseline is always installed')
    parser.add_argument('--model',choices=('whisper-small','whisper-turbo'))
    parser.add_argument('--device',choices=('cpu','cuda'))
    parser.add_argument('--quality',action='store_true',help='Include the optional compatible TranslateGemma 4B weights')
    parser.add_argument('--self-test',action='store_true',help='Run real offline models using generated speech, without capture/playback')
    parser.add_argument('--source',choices=('fi','en','ru','uk'),default='fi')
    parser.add_argument('--target',choices=('fi','en','ru','uk'),default='en')
    parser.add_argument('--translator',choices=('marian','translategemma'),default='marian')
    parser.add_argument('--screenshot',type=Path,metavar='PNG',help='Save a startup/setup screenshot and close; does not capture speech')
    return parser


def detect_cuda():
    from .setup_ui import cuda_is_available
    return cuda_is_available()


def _settings(args, saved=None):
    settings=dict(saved) if saved else {'speech_model':'whisper-small','device':'cpu','languages':['fi','en'],'quality':False}
    if args.model:
        settings['speech_model']=args.model
    if args.device:
        settings['device']=args.device
    if args.languages:
        settings['languages']=[value.strip() for value in args.languages.split(',')]
    if args.quality:
        settings['quality']=True
    return settings


def _keys(settings):
    return required_assets(settings['languages'],settings['speech_model'],settings['quality'])


def _present(paths,catalog,keys):
    for key in keys:
        for asset in catalog[key]:
            path=paths.data_root/asset.relative_path
            try:
                if (path.is_symlink() or not path.resolve().is_relative_to(paths.data_root.resolve())
                        or not path.is_file() or path.stat().st_size!=asset.bytes):
                    return False
            except OSError:
                return False
    return True


def _diagnose(paths,catalog,settings):
    keys=_keys(settings)
    return {'application_root':str(paths.bundle_root),'data_root':str(paths.data_root),
            'python_executable':sys.executable,'python_prefix':sys.prefix,'python_path':sys.path,
            'architecture':platform.machine(),'libc':platform.libc_ver(),
            'cuda_available':detect_cuda(),'models_present':_present(paths,catalog,keys),
            'required_assets':list(keys),'microphones_opened':False,
            'packaged_build':(paths.bundle_root/'appimage-build.json').is_file()}


def _install(args,paths,catalog,settings):
    if not args.accept_model_licenses:
        raise ValueError('Review the publisher model/voice licenses and pass --accept-model-licenses before installing')
    if args.download == (args.from_root is not None):
        raise ValueError('Choose exactly one of --download or --from-root')
    keys=_keys(settings)
    paths.prepare()
    last=[-1]
    def progress(done,total,message):
        percent=int(done*100/total) if total else 0
        if percent!=last[0]:
            print(f'{percent}% · {message}',file=sys.stderr,flush=True)
            last[0]=percent
    install_assets(paths.data_root,catalog,keys,source_root=args.from_root,download=args.download,progress=progress)
    paths.save_settings(settings)
    print(json.dumps({'installed':list(keys),'data_root':str(paths.data_root),'verified':True}))
    return 0


def synthetic_input(root,language):
    """Generate a fixed fixture in memory, never record or play human speech."""
    import math
    import numpy as np
    from piper import PiperVoice, SynthesisConfig
    from scipy.signal import resample_poly
    from .languages import VOICE_FILES
    phrases={'fi':'Hei, miten voit tänään?', 'en':'Hello, how are you today?',
             'ru':'Здравствуйте, как у вас дела сегодня?', 'uk':'Привіт, як справи сьогодні?'}
    voice=PiperVoice.load(root/'models'/f'voice-{language}'/VOICE_FILES[language],use_cuda=False)
    chunks=[]
    count=0
    rate=None
    for chunk in voice.synthesize(phrases[language],syn_config=SynthesisConfig(noise_scale=0,noise_w_scale=0)):
        if rate is None:
            rate=chunk.sample_rate
        samples=np.array(chunk.audio_float_array,dtype=np.float32,copy=True)
        count+=len(samples)
        if chunk.sample_rate!=rate or count>rate*30 or not np.isfinite(samples).all():
            raise RuntimeError('Generated speech fixture was invalid')
        chunks.append(samples)
    if not chunks:
        raise RuntimeError('Generated speech fixture was empty')
    divisor=math.gcd(rate,16000)
    pcm=resample_poly(np.concatenate(chunks),16000//divisor,rate//divisor).astype(np.float32)
    return np.pad(pcm,(1600,1600))


def _self_test(args,paths,settings):
    import numpy as np
    from .engine import ModelEngine
    from .core import Route
    from .languages import translation_steps
    translation_steps(args.source,args.target)
    engine=ModelEngine(paths.data_root,model_name=settings['speech_model'],device=settings['device'])
    try:
        engine.load()
        samples=synthetic_input(paths.data_root,args.source)
        if args.translator!='marian':
            if not engine.set_translation_backend(args.translator):
                raise RuntimeError('Selected translation runtime was not ready')
        route=Route(0,1,args.source,args.target,'synthetic-input-not-opened','synthetic-output-not-opened')
        result=engine.interpret(samples,route,1)
        if (result is None or not result.original or not result.translation or not len(result.audio)
                or not np.isfinite(result.audio).all()):
            raise RuntimeError('The offline pipeline did not return complete finite output')
        report={'synthetic_input':True,'source':args.source,'target':args.target,
                'translator':args.translator,'device':settings['device'],'speech_model':settings['speech_model'],
                'original':result.original,'translation':result.translation,'output_samples':len(result.audio),
                'output_rate':result.rate,'elapsed_seconds':result.elapsed,
                'microphones_opened':False,'audio_played':False,'accuracy_certified':False}
    finally:
        engine.close()
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


def _schedule_screenshot(widget,path,ready=lambda:True):
    from time import monotonic
    from PySide6.QtCore import QTimer
    if path.suffix.lower()!='.png':
        raise ValueError('--screenshot requires a PNG filename')
    capture={'done':False,'error':None}
    deadline=monotonic()+90
    def attempt():
        if not widget.isVisible():
            return
        if not ready():
            if monotonic()<deadline:
                QTimer.singleShot(100,attempt)
                return
            capture['error']='Model startup did not become ready before screenshot timeout'
        else:
            try:
                path.parent.mkdir(parents=True,exist_ok=True)
                if not widget.grab().save(str(path),'PNG'):
                    raise OSError('Native screenshot could not be saved')
                capture['done']=True
            except OSError as error:
                capture['error']=str(error)
        widget.close()
    QTimer.singleShot(350,attempt)
    return capture


def _check_screenshot(capture):
    if capture is not None and (capture['error'] or not capture['done']):
        raise RuntimeError(capture['error'] or 'Screenshot cancelled before capture')


def main(argv=None):
    args=build_parser().parse_args(argv)
    paths=AppPaths.discover(data_root=args.data_dir)
    try:
        if args.screenshot and args.screenshot.suffix.lower()!='.png':
            raise ValueError('--screenshot requires a PNG filename')
        catalog=load_catalog(paths.bundle_root)
        try:
            saved=paths.load_settings() if paths.settings_file.exists() or paths.settings_file.is_symlink() else None
        except ValueError:
            if args.diagnose or args.install_models:
                raise
            saved=None
        settings=_settings(args,saved)
        if args.diagnose:
            print(json.dumps(_diagnose(paths,catalog,settings),indent=2))
            return 0
        if args.install_models:
            return _install(args,paths,catalog,settings)
        if args.download or args.from_root is not None or args.accept_model_licenses:
            raise ValueError('Use --install-models for command-line weight installation, or --setup for the GUI')
        if args.self_test:
            return _self_test(args,paths,settings)
        from PySide6.QtWidgets import QApplication, QDialog
        from .setup_ui import SetupDialog
        from .ui import Window
        from .worker import ModelWorker
        app=QApplication.instance() or QApplication([sys.argv[0]])
        app.setApplicationName('Translitter')
        app.setOrganizationName('Translitter')
        needs_setup=(args.setup or saved is None or not _present(paths,catalog,_keys(settings))
                     or (settings['device']=='cuda' and not detect_cuda()))
        if needs_setup:
            dialog=SetupDialog(paths,catalog=catalog)
            if args.model:
                dialog.speech_model.setCurrentIndex(dialog.speech_model.findData(args.model))
            if args.device and dialog.device.findData(args.device)>=0:
                dialog.device.setCurrentIndex(dialog.device.findData(args.device))
            if args.languages:
                for code,box in dialog.languages.items():
                    if code not in ('fi','en'):
                        box.setChecked(code in settings['languages'])
            if args.quality:
                dialog.quality.setChecked(True)
            capture=_schedule_screenshot(dialog,args.screenshot) if args.screenshot else None
            if dialog.exec()!=QDialog.DialogCode.Accepted:
                _check_screenshot(capture)
                return 0
            settings=paths.load_settings()
        worker=ModelWorker(paths.data_root,model_name=settings['speech_model'],device=settings['device'])
        window=Window(paths.data_root,worker=worker)
        window.show()
        capture=_schedule_screenshot(window,args.screenshot,lambda:window.models_ready) if args.screenshot else None
        code=app.exec()
        _check_screenshot(capture)
        return code
    except (ValueError,OSError,RuntimeError,ModelSetupError) as error:
        print(f'Translitter: {error}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
