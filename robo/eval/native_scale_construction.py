"""Instance build decisions joined independently of policy outcomes."""
import json
from pathlib import Path


def construction_decisions(units,paths):
    from robo.eval.native_scale_evidence import Sources
    src=Sources();found={};planned={(u['canonical_instance_id'],u.get('controller_method')) for u in units if u.get('canonical_instance_id') is not None}
    for path in paths:
        rows=[json.loads(line) for line in src.bind(path).read_text().splitlines() if line.strip()]
        for row in rows:
            key=row['canonical_instance_id'],row['controller_method']
            if key not in planned or key in found:raise ValueError('duplicate or unplanned constructor decision')
            status=row['terminal_status'];accepted=row['accepted']
            if status=='NOT_READY':
                if accepted is not None or row.get('object_dir') is not None:raise ValueError('pending construction is unmeasured')
                found[key]=row;continue
            expected={'BUILT':True,'READY':True,'ABSTAINED':False,'BUILD_FAILED':False}
            if status not in expected or type(accepted) is not bool or accepted!=expected[status]:raise ValueError('constructor status/acceptance mismatch')
            if bool(row.get('object_dir'))!=accepted:raise ValueError('constructor object/acceptance mismatch')
            if row.get('native_success') is not None:raise ValueError('constructor join may not contain policy outcomes')
            source=src.bind(row['binding_source'],row['binding_source_sha256'])
            original=[r for r in map(json.loads,source.read_text().splitlines()) if (r['canonical_instance_id'],r['controller_method'])==key]
            if len(original)!=1 or any(original[0].get(k)!=row.get(k) for k in ('accepted','terminal_status','object_dir','build_manifest','build_manifest_sha256')):
                raise ValueError('snapshot constructor decision differs from bound source')
            manifest=src.read(row['build_manifest'],row['build_manifest_sha256'])
            if 'accepted' in manifest and manifest['accepted']!=accepted:raise ValueError('manifest acceptance differs')
            if manifest.get('canonical_instance_id',key[0])!=key[0]:raise ValueError('manifest canonical instance differs')
            for key_path,key_hash in (('parent_build_manifest','parent_build_manifest_sha256'),('metadata_repair_receipt','metadata_repair_receipt_sha256'),('plan_path','plan_sha256')):
                if row.get(key_path):src.bind(row[key_path],row[key_hash])
            for file,digest in row.get('source_evidence',{}).items():src.bind(file,digest)
            found[key]=row
    for unit in units:
        row=found.get((unit.get('canonical_instance_id'),unit.get('controller_method')))
        if row and row['accepted'] is False and unit.get('executed') is True:raise ValueError('unaccepted construction was executed')
    src.bind(Path(__file__))
    return found,src.lineage
