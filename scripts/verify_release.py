"""Check public prompt examples, network checksums, and local documentation links."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waterbench.methods import INSTRUCTIONS, SYSTEM, prompt
from waterbench.storage import file_hash, read


def main():
    assert (ROOT / 'prompts/system.txt').read_text(encoding='utf-8') == SYSTEM + '\n'
    for task, instruction in INSTRUCTIONS.items():
        assert (ROOT / f'prompts/{task}.txt').read_text(encoding='utf-8') == instruction + '\n'
        example = read(ROOT / f'prompts/examples/{task}.json')
        messages = example['messages']
        assert example['task'] == task
        assert messages[0] == {'role': 'system', 'content': SYSTEM}
        prefix, serialized = messages[1]['content'].split('\nOBSERVATIONS\n', 1)
        observation = json.loads(serialized)
        assert prefix == instruction and observation['task'] == task
        assert messages[1] == {'role': 'user', 'content': prompt(observation)}
        if task == 'leak':
            assert len(observation['hours']) == len(observation['pressure_m'])
        elif task == 'forecast':
            assert len(observation['history_L_s']) == 168
        else:
            assert len(observation['consumer_ids']) == 12 and observation['k'] == 3
        print(f'{task}: exact prompt example verified')
    for network, metadata in read(ROOT / 'data/networks/manifest.json').items():
        assert file_hash(ROOT / f'data/networks/{network}.inp') == metadata['sha256']
    for document in ROOT.rglob('*.md'):
        if '.git' in document.parts or 'data/datasets' in document.as_posix():
            continue
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if '://' not in target and not target.startswith('#'):
                assert (document.parent / target.split('#')[0]).exists(), (document, target)
    print('Network checksums and local documentation links verified')


if __name__ == '__main__':
    main()
