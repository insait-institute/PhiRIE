# PhiRIE release plan

[Project page](https://insait-institute.github.io/PhiRIE/) · [Demo video](https://youtu.be/3-YdcBh6Tbw) · [PhiView](https://github.com/RunyiYang/PhysicalView)

## Development milestones

```text
0.0.0  Reconstruction prototype
└── 1.0.0  Object construction and simulation
    └── 2.0.0  Modular pipeline and PhiView
        └── 2.0.1  Consolidated release, documentation, and demonstrations
```

The early numbers describe development milestones. Release notes for the
packaged versions are available for [2.0.0](docs/releases/v2.0.0.md) and
[2.0.1](docs/releases/v2.0.1.md).

## Available now

- Component interfaces for reconstruction, discovery, generation, registration,
  background editing, rendering, physics, and simulation.
- PhiView source integration and a link to its independently maintained codebase.
- Project page with an overview video, demos 01–09, a mass/friction comparison,
  and three browser playgrounds.
- English installation instructions, illustrated components, and basic demos.

## Next maintenance release — planned

- [ ] Simplify environment setup and document supported GPU/runtime combinations.
- [ ] Provide one complete walkthrough from input scene to PhiView interaction.
- [ ] Publish prepared scene packs with explicit model and data requirements.
- [ ] Package repeatable shooting, grasp-and-place, and harmonizer demo commands.

Ready when a fresh checkout can complete the walkthrough and reproduce its
documented demonstration outputs.

## Following release — planned

- [ ] Expand the prepared scene collection and simulator examples.
- [ ] Document evaluation tasks using descriptive names and complete configurations.
- [ ] Publish benchmark instructions and finalized results.

Ready when the configurations, inputs, and result-generation steps are available
and verified together. Release dates and version numbers will be assigned when
the corresponding work is ready.

## Release checklist

- [ ] Check installation and component interfaces from a clean checkout.
- [ ] Run relevant GPU and simulator checks for changed components.
- [ ] Verify videos, images, downloads, and browser behavior.
- [ ] Update guides and release notes to match the delivered behavior.
- [ ] Tag the validated commit and publish matching release assets.
