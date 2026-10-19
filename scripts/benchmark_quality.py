"""Explicit synthetic text benchmark: current Marian vs local TranslateGemma.

Starts and closes its own private local model runtime. No microphone or playback.
"""
import json
from pathlib import Path
import statistics
import sys
from tempfile import NamedTemporaryFile
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from translitter.languages import translation_steps
from translitter.quality_runtime import QualityRuntime

CASES = (
    ("ru", "fi", "Я не могу прийти сегодня, но смогу завтра после работы."),
    ("ru", "fi", "Если бы я знал об этом раньше, я бы поступил иначе."),
    ("ru", "fi", "Мне сказали не открывать дверь до восьми."),
    ("ru", "fi", "Встречу перенесли с четверга на пятницу, а не наоборот."),
    ("fi", "ru", "En sanonut, että hän varasti rahat."),
    ("fi", "ru", "Vaikka sataa, lähdemme kävelylle, ellei tuuli voimistu."),
    ("fi", "en", "En olisi lähtenyt, jos olisin tiennyt, kuinka myöhään juna saapuu."),
    ("en", "fi", "I didn't say he stole the money."),
    ("en", "fi", "The appointment is at 14:30, not 4:30."),
    ("en", "ru", "Hello, how are you today?"),
    ("en", "ru", "We won't leave unless everyone has arrived."),
    ("ru", "en", "Он едва не забыл закрыть дверь."),
    ("uk", "fi", "Я не хочу змінювати план, якщо це не потрібно."),
    ("fi", "uk", "Voisitko kertoa, miksi kokous siirrettiin ensi viikkoon?"),
)


def main():
    runtime = QualityRuntime(ROOT)
    try:
        client = runtime.start()
        client.probe()
        return _benchmark(client)
    finally:
        runtime.close()


def _save_evidence(path, data):
    """Publish a complete checkpoint only after its stream closes successfully."""
    payload = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                prefix=f'.{path.name}-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _benchmark(client):
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    torch.set_num_threads(4)
    models = {}
    records = []
    path = ROOT / '.local/quality-benchmark.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    for index, (source, target, text) in enumerate(CASES):
        steps = translation_steps(source, target)
        for a, b in steps:
            if (a, b) not in models:
                directory = ROOT / 'models' / f'translate-{a}-{b}'
                models[a, b] = (MarianTokenizer.from_pretrained(directory, local_files_only=True),
                    MarianMTModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                                  weights_only=True).to('cpu').eval())
        baseline = text
        started = monotonic()
        for edge in steps:
            tokenizer, model = models[edge]
            with torch.inference_mode():
                output = model.generate(**tokenizer([baseline], return_tensors='pt', padding=True),
                                        max_new_tokens=96, num_beams=1)
            baseline = tokenizer.decode(output[0], skip_special_tokens=True)
        baseline_seconds = monotonic() - started
        candidate = client.translate(text, source, target)
        record = {'case': index + 1, 'source': source, 'target': target, 'input': text,
                  'baseline': baseline, 'baseline_steps': steps, 'baseline_seconds': baseline_seconds,
                  'candidate': candidate, 'candidate_direct': True, **client.last_metrics}
        records.append(record)
        _save_evidence(path, {'fixture': 'synthetic test sentences, not a recorded conversation',
                                     'quality_verdict': 'requires human bilingual review; execution is not an accuracy score',
                                     'records': records})
        print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {'cases': len(records),
               'baseline_median_seconds': statistics.median(r['baseline_seconds'] for r in records),
               'candidate_median_seconds': statistics.median(r['seconds'] for r in records),
               'candidate_median_first_token_seconds': statistics.median(r['first_token_seconds'] for r in records),
               'human_accuracy_verdict': 'not yet evaluated'}
    data = json.loads(path.read_text(encoding='utf-8'))
    data['summary'] = summary
    _save_evidence(path, data)
    print('BENCHMARK_SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
