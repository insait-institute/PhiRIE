# Portable audit of a published paper

The canonical `robo.eval.paper_pipeline --bundle-existing` mode copies an
already generated publication; it never calls a metric or experiment producer.
The bundle contains the tracked paper sources, both built PDFs and their logs,
existing page renders and font/page QA, generated tables/figures, original
numeric JSON sources, producer config/E0, claim ledger and publication receipt.
Restricted datasets and checkpoints are excluded. Absolute original paths are
preserved as provenance labels; the standalone verifier opens only local bundle
members and needs Python's standard library, no cluster, packages or network.

The verifier checks the externally supplied manifest hash, exact file roster
and bytes, original source fields, finite numbers and exact decimal/percentage
rounding, unavailable cells, enabled claim sentences, producer/paper identities,
PDF hashes and recorded page/font QA. It preserves the original scientific
submission gate. Integrity PASS does not turn a negative or unmeasured result
into a scientific PASS, nor establish raw-data/model reproducibility.

```bash
python -m robo.eval.paper_pipeline \
  --bundle-existing /group/worldcept/code/SimAny/outputs/icra2027/20260906-8cae8a0-v1/paper_tables \
  --paper-root /group/worldcept/code/SimAnyRoom \
  --publication-receipt /group/worldcept/code/SimAny/outputs/icra2027/20260906-8cae8a0-v1/paper_publication/receipt.json \
  --out /group/worldcept/code/SimAny/outputs/icra2027/20260906-8cae8a0-v1/portable_bundle

# After relocating the complete directory, supply the manifest SHA from the
# separately retained packaging receipt, not an untrusted replacement manifest.
python3 -I BUNDLE/verify.py --verify BUNDLE --expected-sha256 EXPECTED_SHA
```

The destination is exclusively created and existing bundles are never
overwritten. Interrupted/invalid packages have no successful external receipt;
preserve them and reserve a new destination for recovery. Source/config edits
require a new freeze. The packaging freeze separately pins the original
publication and does not relabel its distinct experimental source freezes.

Reproducible smoke including stale value, altered source, wrong paper revision,
missing/extra member, alias, no-overwrite, and relocated standalone execution:
`python -m pytest -q tests/test_paper_bundle.py tests/test_paper_pipeline_audit.py`.

Latest completed bundle: paper83c97b1, producer93c9566,337numeric fields,22decisions,283members. Archive62959334bytes, SHA552b46dc5aaea7bb6888e64b431f3a279716f63f1c711816c79d37424ddd92b1. Standalone manifest6076bfc54976da0b8dbbfa6713877a54b6630bff2481988bbe97b403b51cfbac. Relocated `python3 -I` verification PASS; scientific submission remainsFAIL. Original18def27 archive remains unchanged.
