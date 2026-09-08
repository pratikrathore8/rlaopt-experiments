"""Submit one prepared ridge-refinement wave after validating its frozen inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('wave', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--node', choices=['soal-8', 'soal-9', 'soal-12'])
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    os.chdir(repository)
    root = args.plan.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    manifests = [m for m in plan['manifests'] if m['wave'] == args.wave
                 and (args.node is None or m['node'] == args.node)]
    if not manifests:
        raise ValueError('Wave does not exist')
    commands = []
    for item in manifests:
        path = Path(item['path'])
        expected = path.with_suffix('.jsonl.sha256').read_text().split()[0]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Manifest checksum mismatch: {path}')
        jobs = [json.loads(line) for line in path.read_text().splitlines()]
        if not 1 <= len(jobs) <= 6 or len(jobs) != item['tasks']:
            raise ValueError('Invalid task count')
        node = item['node']
        if node not in {'soal-8', 'soal-9', 'soal-12'}:
            raise ValueError('Unsupported benchmark node')
        if item['concurrency'] != (2 if node == 'soal-12' else 1):
            raise ValueError('Unexpected concurrency')
        for job in jobs:
            if job['node'] != node or job['cpu_threads'] != 64:
                raise ValueError('Unexpected job allocation')
            if job['image_sha256'] != plan['image_sha256']:
                raise ValueError('Unexpected job image')
            original = Path(job['original_record'])
            if hashlib.sha256(original.read_bytes()).hexdigest() != job['original_record_sha256']:
                raise ValueError('Frozen original record checksum mismatch')
        command = ['sbatch', '--parsable', f'--nodelist={node}',
                   f"--array=0-{len(jobs)-1}%{item['concurrency']}",
                   f'--job-name=ridge-refinement-{args.wave:03d}-{node}',
                   f'--output={root}/logs/wave-{args.wave:03d}-{node}-%A_%a.out',
                   f'--export=ALL,MANIFEST={path},OUTPUT_DIR={root}/results']
        if node == 'soal-12':
            command.append('--gres=gpu:h200nvl:1')
        commands.append(command + [str(repository / 'slurm/run_ridge_refinement.sh')])
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    if subprocess.check_output(['git', 'status', '--porcelain']):
        raise RuntimeError('Commit the follow-up setup before submitting')
    active = subprocess.check_output(['squeue', '-h', '-r', '-u', os.environ['USER'],
                                     '-o', '%j|%N|%n'], text=True).splitlines()
    targets = {m['node'] for m in manifests}
    for line in active:
        name, allocated, requested = line.split('|')
        if not name.startswith(('ridge-refinement-', 'real-erm-prod-', 'scs-year-tol-')):
            continue
        nodes = set()
        for expression in (allocated.strip(), requested.strip()):
            if expression and expression not in {'(null)', 'N/A', 'None'}:
                nodes.update(subprocess.check_output(
                    ['scontrol', 'show', 'hostnames', expression], text=True).splitlines())
        if not nodes or targets.intersection(nodes):
            raise RuntimeError(f'Wait for conflicting benchmark job: {name}')
    if len(active) + sum(m['tasks'] for m in manifests) > 20:
        raise RuntimeError('Wave would exceed the user-wide 20-job limit')
    env = dict(line.split('=', 1) for line in (repository / 'containers/cuda.env').read_text().splitlines()
               if line and not line.startswith('#'))
    image = repository / env['RLAOPT_CUDA_IMAGE']
    with image.open('rb') as f:
        actual_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    if actual_hash != plan['image_sha256']:
        raise RuntimeError('Container image differs from the frozen refinement plan')
    receipts = [root / f"submission-{args.wave:03d}-{m['node']}.json" for m in manifests]
    if (root / f'submission-{args.wave:03d}.json').exists():
        raise RuntimeError('This wave already has a legacy submission receipt')
    if any(receipt.exists() for receipt in receipts):
        raise RuntimeError('A selected node already has a submission receipt')
    for receipt in receipts:
        with receipt.open('x') as f:
            f.write('[]\n')
    (root / 'logs').mkdir(exist_ok=True)
    for command, receipt in zip(commands, receipts, strict=True):
        job_id = subprocess.check_output(command, text=True).strip()
        receipt.write_text(json.dumps([{'job_id': job_id, 'command': command}], indent=2) + '\n')
        print(job_id, flush=True)


if __name__ == '__main__':
    main()
