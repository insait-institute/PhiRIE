const $ = (s) => document.querySelector(s);
let current = "scene",
  active,
  generation = 0,
  gpuPromise;
async function getGPU() {
  gpuPromise ??= (async () => {
    if (!navigator.gpu || !window.isSecureContext)
      throw new Error(
        "WebGPU is not available. Open this page over HTTPS in a current WebGPU-capable browser.",
      );
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter)
      throw new Error(
        "This browser could not access a WebGPU adapter. Check that hardware acceleration is enabled, or try another supported browser.",
      );
    const info = adapter.info;
    const software =
      !!info?.isFallbackAdapter ||
      /swiftshader|llvmpipe|software/i.test(
        `${info?.architecture} ${info?.device} ${info?.description}`,
      );
    const device = await adapter.requestDevice();
    device.lost.then(() => {
      gpuPromise = undefined;
    });
    return { adapter, device, software };
  })();
  try {
    return await gpuPromise;
  } catch (error) {
    gpuPromise = undefined;
    throw error;
  }
}
const demos = {
  scene: {
    title: "Make the scene your own.",
    copy: "Orbit a captured Gaussian desk and reposition its reconstructed bottle.",
    poster: "bottle-09",
    heading: "One room.<br> Many possibilities.",
    intro:
      "Inspect the captured scene, remove the original target, and move its reconstructed replacement.",
    scope:
      "An interactive preview of saved 3D Gaussian assets. Reduced density and SH0 color for the web. GT-assisted registration.",
    controls: `<div class="control"><label for="scene-stage">Scene stage</label><select id="scene-stage"><option value="3">Reassembled</option><option value="0">Original capture</option><option value="1">Target removed</option><option value="2">Background completed</option></select></div><div class="control"><label for="scene-x">Move sideways <output id="scene-x-value">0 cm</output></label><input id="scene-x" type="range" min="-30" max="30" value="0"></div><div class="control"><label for="scene-y">Move in depth <output id="scene-y-value">0 cm</output></label><input id="scene-y" type="range" min="-30" max="30" value="0"></div><div class="control"><label for="scene-angle">Rotate object <output id="scene-angle-value">0°</output></label><input id="scene-angle" type="range" min="-180" max="180" value="0"></div><div class="control-buttons"><button id="reset-demo">Reset scene</button></div>`,
    module: () => import("./gaussian.js"),
  },
  physics: {
    title: "Put reconstruction in motion.",
    copy: "Drop real reconstructed assets, adjust the physical parameters, and launch a projectile.",
    poster: "cup-06_shooting_1",
    heading: "A little impulse.<br> A different outcome.",
    intro:
      "Choose a reconstructed object. Adjust the parameters, then drop, throw, or strike it.",
    scope:
      "Live Rapier browser physics with simplified convex hulls and estimated parameters. Mesh colors derive from the source Gaussians. This is an interactive illustration, not a calibrated experiment.",
    controls: `<div class="control"><label for="physics-object">Reconstructed object</label><select id="physics-object"><option value="cup">Cup · SAM3D</option><option value="headphone">Headphones · SAM3D</option><option value="keyboard">Keyboard · SAM3D</option></select></div><div class="control"><label for="physics-mass">Mass <output id="physics-mass-value">0.3 kg</output></label><input id="physics-mass" type="range" min="0.1" max="2" step="0.1" value="0.3"></div><div class="control"><label for="physics-friction">Friction <output id="physics-friction-value">0.5</output></label><input id="physics-friction" type="range" min="0" max="1" step="0.05" value="0.5"></div><div class="control"><label for="physics-power">Launch speed <output id="physics-power-value">2 m/s</output></label><input id="physics-power" type="range" min="0.5" max="5" step="0.5" value="2"></div><label class="control-check"><input type="checkbox" id="physics-hull"> Show collision hull</label><div class="control-buttons"><button class="accent" id="physics-shoot">Shoot</button><button id="physics-throw">Throw</button><button id="reset-demo">Reset</button></div><div class="readout" id="physics-readout" role="status">Ready for physics</div>`,
    module: () => import("./physics.js"),
  },
  robot: {
    title: "Every state, in your hands.",
    copy: "Inspect a recorded robot push from any angle. Scrub time, reveal the tool path, and explore the contact.",
    poster: "Q05_libero_living_room_scene5",
    heading: "Follow the motion.<br> Inspect the contact.",
    intro:
      "Orbit the original robot geometry and step through the recorded cup push and retraction.",
    scope:
      "213 recorded MuJoCo integration states. Scripted cup push with SAM3D convex collision; no learned policy or grasp-success claim. Playback selects original states without interpolation.",
    controls: `<div class="control"><label for="robot-time">Recorded time <output id="robot-time-value">0.00 s</output></label><input id="robot-time" type="range" min="0" max="212" step="1" value="0"></div><div class="control"><label for="robot-speed">Playback speed</label><select id="robot-speed"><option value="0.25">0.25× · slow motion</option><option value="0.5">0.5×</option><option value="1" selected>1× · original speed</option><option value="2">2×</option></select></div><label class="control-check"><input type="checkbox" id="robot-path" checked> Show tool trajectory</label><label class="control-check"><input type="checkbox" id="robot-wire"> Wireframe overlay</label><div class="control-buttons"><button class="accent" id="robot-play">Play</button><button id="robot-contact">First contact</button><button id="reset-demo">Reset</button></div><div class="readout" id="robot-readout" role="status">State 1 / 213 · 0.00 s</div>`,
    module: () => import("./robot.js"),
  },
};
function fields(enabled = false) {
  const demo = demos[current];
  $("#lab-controls").innerHTML =
    `<p class="eyebrow">${current === "scene" ? "SCENE EDITING" : current === "physics" ? "PHYSICS SANDBOX" : "ROBOT REPLAY"}</p><h3>${demo.heading}</h3><p class="controls-intro">${demo.intro}</p><div id="control-fields">${demo.controls}</div><p class="demo-scope">${demo.scope}</p>`;
  $("#control-fields")
    .querySelectorAll("button,input,select")
    .forEach((el) => (el.disabled = !enabled));
}
function status(message) {
  $("#gpu-status").textContent = message;
}
export async function selectDemo(name) {
  generation++;
  active?.dispose();
  active = null;
  current = name;
  const demo = demos[name];
  document.querySelectorAll("[data-demo]").forEach((b) => {
    b.setAttribute("aria-selected", String(b.dataset.demo === name));
    b.tabIndex = b.dataset.demo === name ? 0 : -1;
  });
  $("#lab-panel").setAttribute("aria-labelledby", `tab-${name}`);
  $("#viewport").innerHTML =
    `<img id="demo-poster" src="media/${demo.poster}.webp" alt="${demo.title}"><div id="demo-overlay"><span class="viewport-icon" aria-hidden="true">◇</span><h3>${demo.title}</h3><p>${demo.copy}</p><button id="launch-demo" class="button primary">Launch WebGPU demo ↗</button><p class="download-note">3D assets load only when you launch.</p></div>`;
  $("#launch-demo").addEventListener("click", launch);
  fields();
  status("Ready to explore");
}
export async function launch() {
  const ticket = ++generation;
  const button = $("#launch-demo");
  if (button) {
    button.disabled = true;
    button.textContent = "Loading 3D assets…";
  }
  status("Checking WebGPU…");
  fields();
  try {
    const gpu = await getGPU();
    status("Loading the scene…");
    const module = await demos[current].module();
    if (ticket !== generation) return;
    const instance = await module.createDemo($("#viewport"), {
      ...gpu,
      status,
      isCurrent: () => ticket === generation,
    });
    if (ticket !== generation) {
      instance.dispose();
      return;
    }
    active = instance;
    $("#demo-overlay")?.remove();
    $("#demo-poster")?.remove();
    $("#control-fields")
      .querySelectorAll("button,input,select")
      .forEach((el) => (el.disabled = false));
    status(`WebGPU active${gpu.software ? " · software adapter" : ""}`);
    $("#lab-panel").dataset.activeDemo = current;
    $("#lab-panel").dataset.backend = "webgpu";
  } catch (error) {
    if (ticket !== generation) return;
    active?.dispose();
    active = null;
    status("WebGPU unavailable");
    const view = $("#viewport");
    view.querySelectorAll("canvas,#demo-overlay").forEach((e) => e.remove());
    const message = document.createElement("div");
    message.className = "gpu-error";
    const title = document.createElement("h3");
    title.textContent = "Continue with the recorded results.";
    const p = document.createElement("p");
    p.textContent = error.message;
    const link = document.createElement("a");
    link.href = "#motion";
    link.textContent = "Watch interaction videos ↓";
    message.append(title, p, link);
    view.append(message);
    delete $("#lab-panel").dataset.activeDemo;
    delete $("#lab-panel").dataset.backend;
    console.warn("PhiRIE demo unavailable:", error.message);
  }
}
fields();
