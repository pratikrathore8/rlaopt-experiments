from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from rlaopt_experiments.records import TrialRecord


def load_script(name):
    path = Path(__file__).parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(('outcomes', 'expected'), [
    ([(True, True)], 1),
    ([(False, False)], 1),
    ([(True, False), (True, True)], 2),
    ([(True, False), (True, False)], 2),
])
def test_refinement_only_retries_native_success_accuracy_misses(tmp_path, monkeypatch, outcomes, expected):
    module = load_script('run_ridge_refinement')
    source = tmp_path / 'original.json'
    source.write_text('{}')
    monkeypatch.setenv('RLAOPT_CUDA_IMAGE_SHA256', 'frozen-image')
    job = {
        'trial_id': 'trial', 'original_record': str(source),
        'original_record_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'original_image_sha256': 'original-image', 'image_sha256': 'frozen-image',
        'original_runtime_seconds': 4.0, 'native_tolerances': [1e-10, 1e-11],
        'problem': {'n': 16, 'p': 8, 'alpha': 2., 'ridge': 1e-6,
                    'seed': 0, 'solver': 'scipy_lsqr', 'backend': 'cpu'},
        'controls': {'suite': 'synthetic_ridge', 'warmups': 1, 'repetitions': 1,
                     'kkt_tolerance': 1e-6, 'timeout_seconds': 900,
                     'startup_timeout_seconds': 900, 'rank': 128},
    }
    calls = []

    def fake_runner(**kwargs):
        native, external = outcomes[len(calls)]
        calls.append(kwargs)
        return [TrialRecord(
            run_id='result', problem_id='problem', suite='synthetic_ridge',
            solver='scipy_lsqr', backend='cpu', seed=0, repetition=0,
            timings={'runtime_seconds': 2.0}, iterations=10,
            native_status='istop_2' if native else 'iteration_limit',
            metrics={'relative_kkt': 1e-7 if external else 2e-6},
            native_success=native, external_success=external,
            runtime_eligible=native, timed_out=False,
        )]

    output = tmp_path / 'results'
    summary = module.run_trial(job, output, runner=fake_runner)
    assert len(calls) == expected
    assert [c['native_tolerance'] for c in calls] == [1e-10, 1e-11][:expected]
    assert all(c['ridges'] == [1e-6] and c['warmups'] == 1 for c in calls)
    assert len(list(output.glob('trial/native-*/records/*.json'))) == expected
    assert summary['total_measured_solver_seconds_including_original'] == 4 + 2 * expected
    assert summary['qualifying_runtime_seconds'] == (
        2.0 if any(native and external for native, external in outcomes) else None
    )
    assert source.read_text() == '{}'
    with pytest.raises(FileExistsError):
        module.run_trial(job, output, runner=fake_runner)


@pytest.mark.parametrize('active_node', ['soal-12', 'soal-9'])
def test_submission_checks_only_selected_node(tmp_path, monkeypatch, active_node):
    import json
    import sys

    module = load_script('submit_ridge_refinement')
    monkeypatch.setattr(module, '__file__', str(tmp_path / 'scripts' / 'submit.py'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('USER', 'test-user')
    root = tmp_path / 'plan'
    root.mkdir()
    containers = tmp_path / 'containers'
    containers.mkdir()
    (containers / 'image.sif').write_bytes(b'image')
    (containers / 'cuda.env').write_text('RLAOPT_CUDA_IMAGE=containers/image.sif\n')
    digest = hashlib.sha256(b'image').hexdigest()
    original = root / 'original.json'
    original.write_bytes(b'{}')
    manifest = root / 'jobs.jsonl'
    manifest.write_text(json.dumps({
        'node': 'soal-9', 'cpu_threads': 64, 'image_sha256': digest,
        'original_record': str(original),
        'original_record_sha256': hashlib.sha256(b'{}').hexdigest(),
    }) + '\n')
    manifest.with_suffix('.jsonl.sha256').write_text(
        hashlib.sha256(manifest.read_bytes()).hexdigest() + '\n')
    (root / 'plan.json').write_text(json.dumps({
        'image_sha256': digest, 'manifests': [
            {'wave': 0, 'node': node, 'tasks': 1, 'concurrency': 1, 'path': str(manifest)}
            for node in ['soal-8', 'soal-9']
        ],
    }))
    monkeypatch.setattr(sys, 'argv', ['submit', str(root), '0', '--node', 'soal-9'])
    submitted = []

    def fake_command(command, **kwargs):
        if command[0] == 'git':
            return b''
        if command[0] == 'squeue':
            return f'real-erm-prod-test|{active_node}|{active_node}\n'
        if command[0] == 'scontrol':
            return active_node + '\n'
        assert command[0] == 'sbatch'
        submitted.append(command)
        return '12345\n'

    monkeypatch.setattr(module.subprocess, 'check_output', fake_command)
    if active_node == 'soal-9':
        with pytest.raises(RuntimeError, match='conflicting benchmark'):
            module.main()
        assert not submitted
    else:
        module.main()
        assert len(submitted) == 1
        assert '--nodelist=soal-9' in submitted[0]
        assert (root / 'submission-000-soal-9.json').exists()
        assert not (root / 'submission-000-soal-8.json').exists()
        with pytest.raises(RuntimeError, match='submission receipt'):
            module.main()
        assert len(submitted) == 1
