"""Package exact source into a self-contained PRIVATE Kaggle notebook.

No credentials are copied. Username is metadata, never an API key.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--username')
p.add_argument('--seeds',type=int,default=5)
a=p.parse_args()
dest=ROOT/'kaggle'; dest.mkdir(exist_ok=True)
cells=[]
def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
def code(s): cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)})
md('# Ego Gate: IEEE evidence package\nPrivate execution notebook. Runs the existing study and a separately specified online extension. No numerical result is assumed in advance. Export all raw outputs and manifests. The new gate is a fixed heuristic, not a learned meta-controller.\n')
code('from pathlib import Path\nimport sys, subprocess, os, json, hashlib, platform\nimport torch, numpy, scipy, sklearn\ntorch.set_num_threads(2)\nos.environ["OMP_NUM_THREADS"]="2"\nos.environ["MKL_NUM_THREADS"]="2"\nbase=Path("/kaggle/working/ego_gate")\nbase.mkdir(exist_ok=True)\n')
hashes={}
for path in ['experiments/run_digits.py','experiments/stream_gate.py','scripts/analyze_stream.py']:
    source=(ROOT/path).read_text(encoding='utf-8')
    hashes[path]=hashlib.sha256(source.encode()).hexdigest()
    code(f'(base/{Path(path).name!r}).write_text({source!r},encoding="utf-8")\n')
code('manifest='+repr({'source_hashes':hashes,'seeds_online':a.seeds})+'\nmanifest.update({"python":platform.python_version(),"torch":torch.__version__,"numpy":numpy.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"kaggle":bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))})\n(base/"execution_manifest.json").write_text(json.dumps(manifest,indent=2))\n')
md('## Study A: independent reproduction of the existing two-task result\nThe manuscript currently reports archived local results. This run is independent evidence and may differ with platform and package versions.\n')
code('subprocess.run([sys.executable,str(base/"run_digits.py"),"--num-seeds","20","--output-dir",str(base/"digits_reproduction"),"--device","cpu"],check=True)\n')
md('## Study B: single-pass online admission and consolidation\nFive two-class phases, randomized class order; boundaries never enter the learner. Clean and 20% label-noise streams; 200 stored examples. Ten conditions separate storage, admission and scheduling. Every method has equal training exposure/update budgets; scoring time and temporary memory are additional reported costs.\n')
code('subprocess.run([sys.executable,str(base/"stream_gate.py"),"--seeds",'+repr(str(a.seeds))+',"--output",str(base/"stream_results"),"--device","cpu"],check=True)\n')
code('subprocess.run([sys.executable,str(base/"analyze_stream.py"),str(base/"stream_results")],check=True)\n')
md('## Completion and integrity\nA completion marker is written only after both studies finish successfully. Partial raw outputs remain available after a failure.\n')
code('(base/"COMPLETE.json").write_text(json.dumps({"complete":True,"source_hashes":manifest["source_hashes"]},indent=2))\nprint("Both studies complete; download /kaggle/working/ego_gate")\n')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells): c['id']=f'egogate-{i:02}'
(dest/'ego_gate_ieee.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf-8')
metadata={'title':'Ego Gate IEEE Experiments','code_file':'ego_gate_ieee.ipynb','language':'python','kernel_type':'notebook',
          'is_private':'true','enable_gpu':'false','enable_internet':'true','dataset_sources':[], 'competition_sources':[], 'kernel_sources':[]}
if a.username:
    metadata['id']=a.username+'/ego-gate-ieee-experiments'
    (dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
(dest/'source_manifest.json').write_text(json.dumps(hashes,indent=2))
print('Notebook prepared. '+('Private metadata ready.' if a.username else 'Provide --username to create submission metadata.'))
