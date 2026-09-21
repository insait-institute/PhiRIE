# N4 — Replace interaction context, not only one object

**Priority P0 in parallel with DEV. Owner: native-import agent.** Read common matrix and N0 scorer/frame contract. Own `robo/roundtrip/importers/robocasa.py` and importer hooks; coordinate rather than concurrently editing them with N3/N5.

Current importer supports one world-parented free rigid target, estimated Sim(3), generated collision, and native room/destination as retained oracle context. It explicitly rejects non-target roles. Extending this boundary requires validated new functionality, not renaming the old scope.

## Replacement ladder

- L0: one manipulated target. Native support, destination and room remain privileged context.
- L1: target plus destination container or destination support surface. Build/open-container geometry must come from observations. Retained fixture shells and metadata are listed explicitly.
- L2: target, destination, source/destination supports and all relevant obstacles within a declared task workspace. Both approach and transport regions are included. Record any physical interaction with retained geometry outside that region.
- L3: a declared whole-room extent and inventory. Use two predeclared cases initially with their own budget. Do not describe L0 or a small reconstructed tabletop as room-scale reconstruction.

## TODO

- [ ] Create role-aware asset import for target, receptacle, support and obstacle. Support free/static body semantics without changing the original robot or task goals. Articulations remain unsupported unless separately implemented and tested.
- [ ] Preserve metric placement with scale applied once. Convert parent/local/world frames explicitly. When re-expressing a mesh frame, preserve world vertices, collision, inertial pose and evaluator site semantics together.
- [ ] Rebind native contact geom IDs, object handles, bounds/sites and task state functions after each replacement. Never retain old geometry under hidden group/contact settings. Check the scorer-point invariance tests from N0.
- [ ] Handle multi-object imports by joint/body names, not positional qpos offsets. Inserting/removing static or free bodies can reorder integration-state indices. Preserve unchanged robot/controller/fixture state exactly and record changed state mapping.
- [ ] Preserve reconstructed container openings. Test small rigid-object entry through the opening and contact with the interior/bottom, without assisted welding or fabricated state toggles. Keep native inside semantics plus independent collision/containment diagnostics.
- [ ] Build reconstructed support and obstacle collision from the capture, not by copying native meshes into a purported full reconstruction. Allowed retained reference components remain explicitly tagged.
- [ ] Native evaluator access occurs only after constructor outputs are sealed. Reference-role IDs map outputs for scoring, not for choosing which candidate best matches GT.
- [ ] Integrate N3 corrections transactionally: changed pose/collision invalidates affected native handles and appearance dependencies; untouched canonical components retain their hashes.
- [ ] Produce unchanged-import identity controls for every new role/import route. A physically identical export/import must preserve state, observations and native predicates under a declared action sequence. Existing target control is not enough for a new container importer.

## Scope experiment

Freeze 4 layouts from the TEST roster by a non-outcome rule before quality inspection. This gives 240 reset units. Evaluate B3/B4 at L1 and L2 (960 new episodes) and reuse matching L0/reference records. Failure to import one destination stays a failed L1/L2 unit, not an automatic return to L0 under the same label.

If no generic sink/fixture replacement is ready, implement a bounded supported destination type and narrow the NEW prospective task roster before TEST begins. Do not select only scenes whose reconstructed containers already work. Cabinet opening or hinge recovery is a separate capability; do not silently hold a previously closed cabinet open to make a rigid-task result easier.

## Outputs

Per scope: `scope_manifest.json` lists reconstructed, retained-reference, intentionally excluded and failed entities; workspace bounds; oracle-context flags; source mesh/transform/collision hashes; native role bindings; actual material policy; contact interactions with retained context. Produce `import_identity.json`, actual MJCF, state remap, initialization evidence and scoped failed-unit record. Keep the original imported candidate files immutable.

## Tests / acceptance

Test transformed parent support, multiple free bodies with different joint ordering, bowl/tray cavity versus closed hull, duplicate old collision, removed geom references, scorer invariance, estimated-pose preservation and a support correction that does not move the robot. Real DEV pilot must execute a continuous native episode after target+destination replacement. Successful completion is not required to validate integration; the episode and diagnosis must be interpretable.

Write `STATUS.md` with exactly which L0-L3 levels are implemented, tested and measured. A scope is complete only with a full retained/reconstructed inventory. Hand N5 source-bound per-method import manifests and N6 the same live-pose mapping.
