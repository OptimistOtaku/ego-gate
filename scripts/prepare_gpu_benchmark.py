"""Package private Kaggle GPU benchmark with source and protocol hashes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--smoke',action='store_true')
a=p.parse_args()
dest=ROOT/'kaggle'/'gpu_benchmark'
dest.mkdir(parents=True,exist_ok=True)
source=(ROOT/'experiments/offline_benchmark.py').read_text(encoding='utf-8')
protocol=(ROOT/'paper/ieee/BENCHMARK_PROTOCOL.md').read_text(encoding='utf-8')
analysis=(ROOT/'scripts/analyze_offline_benchmark.py').read_text(encoding='utf-8')
cells=[]
def code(s):
    cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True),id=f'gpu-{len(cells)}'))
cells.append(dict(cell_type='markdown',metadata={},source=[
    '# Ego Gate offline benchmark expansion\n',
    'Private GPU execution. Synthetic/unit tests are separate from benchmark evidence.\n',
    'This run is an execution check, not inferential evidence.\n' if a.smoke else 'Fixed protocol; preserve all outcomes.\n'],id='intro'))
code('from pathlib import Path\nimport os, sys, subprocess, json, hashlib\nimport torch\nbase=Path("/kaggle/working/benchmark")\nbase.mkdir(exist_ok=True)\nassert torch.cuda.is_available(), "GPU unavailable"\nprint("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],flush=True)\nos.environ["OMP_NUM_THREADS"]="2"\nos.environ["MKL_NUM_THREADS"]="2"\nos.environ["OPENBLAS_NUM_THREADS"]="2"\n')
code(f'(base/"offline_benchmark.py").write_text({source!r},encoding="utf-8")\n(base/"BENCHMARK_PROTOCOL.md").write_text({protocol!r},encoding="utf-8")\n')
code(f'(base/"analyze_offline_benchmark.py").write_text({analysis!r},encoding="utf-8")\n')
manifest=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),protocol_sha256=hashlib.sha256(protocol.encode()).hexdigest(),analysis_sha256=hashlib.sha256(analysis.encode()).hexdigest(),smoke=a.smoke)
code(f'manifest={manifest!r}\n(base/"source_manifest.json").write_text(json.dumps(manifest,indent=2))\n')
if a.smoke:
    code('for dataset in ["digits","fashion","cifar10"]:\n    subprocess.run([sys.executable,str(base/"offline_benchmark.py"),"--dataset",dataset,"--seeds","999","--output",str(base/dataset),"--smoke"],check=True)\n')
else:
    code('sys.path.insert(0,str(base))\nfrom offline_benchmark import load_data\n# Download/extract once, before workers, to avoid racing over shared archives.\nfor dataset in ["fashion","cifar10"]:\n    staged=load_data(dataset,Path("/kaggle/working/data"),100)\n    del staged\nprint("Datasets staged",flush=True)\n')
    code('from concurrent.futures import ThreadPoolExecutor\nworkers=min(2,torch.cuda.device_count())\ndef shard(worker):\n    for dataset in ["digits","fashion","cifar10"]:\n        seeds=list(range(100+worker,110,workers))\n        subprocess.run([sys.executable,str(base/"offline_benchmark.py"),"--dataset",dataset,"--seeds",*map(str,seeds),"--device",f"cuda:{worker}","--output",str(base/dataset/f"shard{worker}")],check=True)\nwith ThreadPoolExecutor(max_workers=workers) as pool:\n    list(pool.map(shard,range(workers)))\n')
if not a.smoke:
    code('subprocess.run([sys.executable,str(base/"analyze_offline_benchmark.py"),str(base),"--output",str(base/"analysis")],check=True)\n')
code('(base/"COMPLETE.json").write_text(json.dumps(manifest,indent=2))\nprint("GPU benchmark complete",flush=True)\n')
nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
(dest/'benchmark.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf-8')
(dest/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
metadata=dict(id='optimistotaku/ego-gate-gpu-benchmarks',title='Ego Gate GPU Benchmarks',code_file='benchmark.ipynb',language='python',kernel_type='notebook',is_private='true',enable_gpu='true',enable_internet='true',dataset_sources=[],competition_sources=[],kernel_sources=[])
metadata['dataset_sources']=['pankrzysiu/cifar10-python']
(dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print('Prepared GPU '+('execution check' if a.smoke else 'full benchmark'))
