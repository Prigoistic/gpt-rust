"""Verify portable M0/M1 evidence and the exact measured package source.

No torch, checkpoints, CUDA, network access or original machine paths required.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from blt_baseline.comparison import paired_summary
from blt_baseline.runner import timing_summary
from blt_baseline.workloads import output_accounting


def require(condition,message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a,b):
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,float):
        return isinstance(b,(float,int)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b


def check_sample(sample):
    observed=[output_accounting(ids) for ids in sample['generated_ids']]
    require(observed==sample['outputs'],'Generated IDs/output accounting mismatch')
    require(sum(o['raw_byte_count'] for o in observed)==sample['raw_byte_count'],'Incorrect byte denominator')


def verify(evidence: Path):
    def read(name):return json.loads((evidence/(name+'.json')).read_text())
    provenance=read('provenance')
    for name,record in provenance['original_files'].items():
        require(digest(evidence/(name+'.json'))==record['export_sha256'],f'Public evidence hash mismatch: {name}')
    for record in provenance['measured_package_source']:
        path=ROOT/'src/blt_baseline'/record['path']
        require(digest(path)==record['sha256'],f'Measured package source changed: {record["path"]}')
    for record in provenance['measured_test_source']:
        require(digest(ROOT/'tests'/record['path'])==record['sha256'],f'Measured test changed: {record["path"]}')
    m0,m1,official,gate,audit=[read(n) for n in ('m0','m1','official','entropy-gate','decode-audit')]
    require(all(r['status']=='COMPLETED' for r in (m0,m1,official)),'Incomplete measurement')
    require(len(m0['performance'])==6 and len(m1['performance'])==9 and len(official['cases'])==3,'Unexpected matrix size')
    golden={(c['actual_context_bytes'],c['batch_size']):c for c in m0['performance']}
    for case in m0['performance']:
        for sample in case['samples']:check_sample(sample)
        require(close(timing_summary(case['samples'],None),case['summary']),'M0 summary does not recompute')
    batches=raw_bytes=0
    for case in m1['performance']:
        require(len(case['samples'])==case['repeats'],'M1 sample count changed')
        for pair in case['samples']:
            for sample in pair.values():
                check_sample(sample);batches+=1;raw_bytes+=sample['raw_byte_count']
        require(close(paired_summary(case['samples']),case['summary']),'M1 paired summary does not recompute')
        if case['primary']:
            shape=(len(case['prompts'][0].encode()),len(case['prompts']))
            require(case['samples'][0]['padded']['generated_ids']==golden[shape]['samples'][0]['generated_ids'],'M0 reference output changed')
    for case in official['cases']:
        require(len(case['samples'])==5 and case['batch']==1 and case['steps']==8,'Official protocol changed')
        for pair in case['samples']:
            for sample in pair.values():
                check_sample(sample);batches+=1;raw_bytes+=sample['raw_byte_count']
        require(close(paired_summary(case['samples'],reference='official_generate_nocache'),case['summary']),'Official paired summary does not recompute')
    require(batches==201 and raw_bytes==6600,'Unexpected total measurement denominator')
    require(len(m1['correctness'])==27 and all(c['patches_equal'] and c['logits_max_abs_error']==0 for c in m1['correctness']),'Full-model parity failed')
    require(gate['status']=='PASS' and len(gate['cases'])==36 and all(c['patches_equal'] for c in gate['cases']),'Entropy gate failed')
    require(gate['causal_suffix_invariance'] and gate['threshold_witness']['guarded_patches_equal'] and not gate['threshold_witness']['unguarded_patches_equal'],'Boundary guard/causality witness failed')
    require(audit['status']=='PASS' and len(audit['states'])==108 and all(c['patches_equal'] for c in audit['states']),'Generated-prefix audit failed')
    entropy_source=digest(ROOT/'src/blt_baseline/entropy.py')
    require(audit['entropy_source_sha256']==entropy_source,'Audit implementation differs from published source')
    scores={}
    for name in ('padded','length_aware_rows','length_aware_batch'):
        count=sum(r['policies'][name]['scored_bytes'] for r in m1['quality'])
        nats=math.fsum(r['policies'][name]['nll_nats'] for r in m1['quality'])
        require(count==144,'Quality denominator changed')
        scores[name]=(count,nats)
    require(scores['padded']==scores['length_aware_rows']==scores['length_aware_batch'],'Diagnostic likelihood changed')
    require(m1['admission']['length_aware_batch'],'Registered speedup gate failed')
    return {'status':'PASS','timed_policy_batches':batches,'raw_bytes':raw_bytes,
            'measured_package_source':'exact match','M0_M1_and_official_summaries':'recomputed',
            'quality':'identical diagnostic likelihood, not broad task quality',
            'excluded':'does not rerun CUDA, certify all inputs, or validate unavailable original archives'}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=ROOT/'reports/evidence')
    args=parser.parse_args()
    print(json.dumps(verify(args.evidence),indent=2))
