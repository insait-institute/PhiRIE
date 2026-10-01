import { THREE, loader, createStage } from "./three-scene.js";

export async function createDemo(container, options) {
  const [gltf, motion] = await Promise.all([
    loader.loadAsync("models/robot.glb"),
    fetch("models/robot-motion.json").then((r) => {
      if (!r.ok) throw new Error("Robot motion could not load");
      return r.json();
    }),
  ]);
  const stage = await createStage(container, options, {
    zUp: true,
    position: [1.25, -1.05, 1.12],
    target: [0.12, 0.18, 0.62],
  });
  const { scene, on } = stage;
  gltf.scene.traverse((o) => {
    if (o.isMesh) {
      o.material.metalness = 0;
      o.material.roughness = 0.65;
      o.material.side = THREE.DoubleSide;
      o.geometry.computeVertexNormals();
    }
  });
  const wires = [];
  gltf.scene.traverse((o) => {
    if (o.isMesh) {
      const wire = new THREE.LineSegments(
        new THREE.WireframeGeometry(o.geometry),
        new THREE.LineBasicMaterial({
          color: 0x3c6658,
          transparent: true,
          opacity: 0.55,
        }),
      );
      wire.visible = false;
      wires.push([o, wire]);
    }
  });
  for (const [mesh, wire] of wires) mesh.add(wire);
  scene.add(gltf.scene);
  // The source export carries only the robot, target and selected tabletop mesh.
  // A translucent plane marks the recorded support height for spatial context.
  const support = new THREE.Mesh(
    new THREE.BoxGeometry(1.15, 0.85, 0.018),
    new THREE.MeshStandardMaterial({ color: 0x82946c, roughness: 0.9 }),
  );
  support.position.set(-0.17, 0.05, 0.427);
  scene.add(support);
  const nodes = motion.ids.map((id) =>
    gltf.scene.getObjectByName(`geom_${id}`),
  );
  if (nodes.some((node) => !node)) {
    stage.dispose();
    throw new Error("Recorded robot geometry is incomplete.");
  }
  const grid = new THREE.GridHelper(3, 30, 0x617957, 0x2b4836);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.015;
  scene.add(grid);
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(
      motion.toolpath.map((p) => new THREE.Vector3(...p)),
    ),
    new THREE.LineBasicMaterial({
      color: 0xe9c976,
      transparent: true,
      opacity: 0.8,
    }),
  );
  scene.add(line);
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.008, 12, 8),
    new THREE.MeshBasicMaterial({ color: 0xf9dd93 }),
  );
  scene.add(marker);
  let playing = false,
    time = 0,
    frame = 0,
    speed = 1;
  function apply(index) {
    frame = index;
    time = motion.times[index];
    motion.frames[index].forEach((pose, i) => {
      nodes[i].position.fromArray(pose);
      nodes[i].quaternion.fromArray(pose, 3);
    });
    marker.position.fromArray(motion.toolpath[index]);
    document.querySelector("#robot-time").value = String(index);
    document.querySelector("#robot-time-value").textContent =
      `${time.toFixed(2)} s`;
    document.querySelector("#robot-readout").textContent =
      `State ${index + 1} / ${motion.times.length} · ${time.toFixed(2)} s${time >= motion.contact_time ? " · after first contact" : ""}`;
    container.dataset.robotFrame = String(index);
    container.dataset.robotTime = String(time);
  }
  apply(0);
  on("#robot-time", "input", (e) => {
    playing = false;
    document.querySelector("#robot-play").textContent = "Play";
    apply(+e.target.value);
  });
  on("#robot-speed", "change", (e) => (speed = +e.target.value));
  on("#robot-path", "change", (e) => {
    line.visible = e.target.checked;
    marker.visible = e.target.checked;
  });
  on("#robot-wire", "change", (e) => {
    for (const [, wire] of wires) wire.visible = e.target.checked;
  });
  on("#robot-play", "click", () => {
    if (time >= motion.duration) apply(0);
    playing = !playing;
    document.querySelector("#robot-play").textContent = playing
      ? "Pause"
      : "Play";
  });
  on("#robot-contact", "click", () => {
    playing = false;
    document.querySelector("#robot-play").textContent = "Play";
    apply(motion.times.findIndex((t) => t >= motion.contact_time));
  });
  on("#reset-demo", "click", () => {
    playing = false;
    document.querySelector("#robot-play").textContent = "Play";
    apply(0);
    stage.resetCamera();
  });
  let playhead = 0;
  stage.start((dt) => {
    if (!playing) {
      playhead = time;
      return;
    }
    playhead = Math.min(playhead + dt * speed, motion.duration);
    let index = frame;
    while (
      index < motion.times.length - 1 &&
      motion.times[index + 1] <= playhead
    )
      index++;
    apply(index);
    if (playhead >= motion.duration) {
      playing = false;
      document.querySelector("#robot-play").textContent = "Play";
    }
  });
  return {
    dispose() {
      stage.dispose();
      delete container.dataset.robotFrame;
      delete container.dataset.robotTime;
    },
  };
}
