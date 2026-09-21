import RAPIER from "@dimforge/rapier3d-compat";
import { ConvexGeometry } from "three/addons/geometries/ConvexGeometry.js";
import { THREE, loader, createStage, tabletop } from "./three-scene.js";

let initialized;
export async function createDemo(container, options) {
  initialized ??= RAPIER.init();
  const [_, ...models] = await Promise.all([
    initialized,
    ...["cup", "headphone", "keyboard"].map((name) =>
      loader.loadAsync(`models/${name}.glb`),
    ),
  ]);
  const stage = await createStage(container, options, {
    position: [0.68, 0.52, 0.75],
    target: [0, 0.04, 0],
  });
  const { scene, on } = stage;
  tabletop(scene, 1, 0.75);
  const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
  world.timestep = 1 / 120;
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(0.5, 0.025, 0.375)
      .setTranslation(0, -0.027, 0)
      .setFriction(0.5),
  );
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(4, 0.02, 4).setTranslation(0, -0.31, 0),
  );
  const assets = {};
  ["cup", "headphone", "keyboard"].forEach((name, i) => {
    const root = models[i].scene;
    const points = [];
    root.updateMatrixWorld(true);
    root.traverse((mesh) => {
      if (!mesh.isMesh) return;
      mesh.material.roughness = 0.65;
      mesh.material.metalness = 0.03;
      const p = mesh.geometry.getAttribute("position");
      for (let j = 0; j < p.count; j++)
        points.push(
          new THREE.Vector3()
            .fromBufferAttribute(p, j)
            .applyMatrix4(mesh.matrixWorld),
        );
    });
    const geometry = new ConvexGeometry(points);
    const hull = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        color: 0xdbedac,
        wireframe: true,
        transparent: true,
        opacity: 0.36,
        depthTest: false,
      }),
    );
    root.add(hull);
    hull.visible = false;
    const bounds = new THREE.Box3().setFromObject(root);
    assets[name] = {
      root,
      hull,
      vertices: Float32Array.from(points.flatMap((p) => p.toArray())),
      halfHeight: (bounds.max.y - bounds.min.y) / 2,
    };
  });
  let object,
    body,
    collider,
    selected = "cup",
    time = 0,
    accumulator = 0,
    shots = 0;
  const projectiles = [];
  function reset() {
    if (body) world.removeRigidBody(body);
    if (object) scene.remove(object.root);
    for (const p of projectiles) {
      world.removeRigidBody(p.body);
      scene.remove(p.mesh);
      p.mesh.geometry.dispose();
      p.mesh.material.dispose();
    }
    projectiles.length = 0;
    object = assets[selected];
    object.hull.visible = document.querySelector("#physics-hull").checked;
    scene.add(object.root);
    body = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(0, 0.16 + object.halfHeight, 0)
        .setCcdEnabled(true)
        .setLinearDamping(0.05)
        .setAngularDamping(0.12),
    );
    const desc = RAPIER.ColliderDesc.convexHull(object.vertices);
    if (!desc) throw new Error("Object collision geometry could not load.");
    collider = world.createCollider(
      desc
        .setMass(+document.querySelector("#physics-mass").value)
        .setFriction(+document.querySelector("#physics-friction").value)
        .setRestitution(0.12),
      body,
    );
    time = 0;
    shots = 0;
    accumulator = 0;
    container.dataset.physicsObject = selected;
  }
  reset();
  on("#physics-object", "change", (e) => {
    selected = e.target.value;
    reset();
  });
  on("#physics-mass", "input", (e) => {
    document.querySelector("#physics-mass-value").textContent =
      `${e.target.value} kg`;
    collider.setMass(+e.target.value);
    body.recomputeMassPropertiesFromColliders();
    body.wakeUp();
  });
  on("#physics-friction", "input", (e) => {
    document.querySelector("#physics-friction-value").textContent =
      e.target.value;
    collider.setFriction(+e.target.value);
    body.wakeUp();
  });
  on("#physics-power", "input", (e) => {
    document.querySelector("#physics-power-value").textContent =
      `${e.target.value} m/s`;
  });
  on("#physics-hull", "change", (e) => {
    object.hull.visible = e.target.checked;
  });
  on("#reset-demo", "click", () => {
    reset();
    stage.resetCamera();
  });
  on("#physics-throw", "click", () => {
    const power = +document.querySelector("#physics-power").value;
    body.setTranslation({ x: -0.18, y: 0.18 + object.halfHeight, z: 0 }, true);
    body.setLinvel({ x: power * 0.3, y: power * 0.35, z: power * 0.06 }, true);
    body.setAngvel({ x: 1.5, y: 0, z: -1 }, true);
    container.dataset.throwCount = String(
      +(container.dataset.throwCount || 0) + 1,
    );
  });
  on("#physics-shoot", "click", () => {
    if (projectiles.length >= 8) {
      const old = projectiles.shift();
      world.removeRigidBody(old.body);
      scene.remove(old.mesh);
      old.mesh.geometry.dispose();
      old.mesh.material.dispose();
    }
    const target = body.translation(),
      power = +document.querySelector("#physics-power").value;
    const r = 0.016;
    const projectile = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(target.x - 0.25, target.y + 0.012, target.z + 0.006)
        .setLinvel(power, (9.81 * 0.25) / (2 * power), 0)
        .setCcdEnabled(true),
    );
    world.createCollider(
      RAPIER.ColliderDesc.ball(r)
        .setMass(0.08)
        .setRestitution(0.25)
        .setFriction(0.3),
      projectile,
    );
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(r, 16, 12),
      new THREE.MeshStandardMaterial({
        color: 0xeac582,
        metalness: 0.55,
        roughness: 0.25,
      }),
    );
    scene.add(mesh);
    projectiles.push({ body: projectile, mesh });
    shots++;
    container.dataset.shots = String(shots);
  });
  stage.start((dt) => {
    accumulator += dt;
    while (accumulator >= world.timestep) {
      world.step();
      accumulator -= world.timestep;
      time += world.timestep;
    }
    object.root.position.copy(body.translation());
    object.root.quaternion.copy(body.rotation());
    for (const p of projectiles) {
      p.mesh.position.copy(p.body.translation());
      p.mesh.quaternion.copy(p.body.rotation());
    }
    const v = body.linvel(),
      speed = Math.hypot(v.x, v.y, v.z);
    document.querySelector("#physics-readout").textContent =
      `${time.toFixed(1)} s · ${speed.toFixed(2)} m/s · ${shots} shots`;
    container.dataset.physicsY = body.translation().y.toFixed(5);
    container.dataset.physicsX = body.translation().x.toFixed(5);
    container.dataset.physicsTime = time.toFixed(3);
  });
  return {
    dispose() {
      // Dispose detached alternatives as well as the currently visible model.
      for (const asset of Object.values(assets))
        if (asset !== object) scene.add(asset.root);
      stage.dispose();
      world.free();
      for (const key of [
        "physicsObject",
        "physicsX",
        "physicsY",
        "physicsTime",
        "shots",
        "throwCount",
      ])
        delete container.dataset[key];
    },
  };
}
