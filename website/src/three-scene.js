import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
export { THREE };
export const loader = new GLTFLoader();

export async function createStage(
  container,
  options,
  { zUp = false, position = [1.2, 1, 1.3], target = [0, 0.2, 0] } = {},
) {
  const renderer = new THREE.WebGPURenderer({
    antialias: true,
    alpha: false,
    device: options.device,
  });
  await renderer.init();
  if (!renderer.backend.isWebGPUBackend) {
    renderer.dispose();
    throw new Error("A native WebGPU backend could not initialize.");
  }
  if (!options.isCurrent()) {
    renderer.dispose();
    throw new Error("Demo loading cancelled.");
  }
  renderer.setPixelRatio(
    options.software ? 1 : Math.min(window.devicePixelRatio, 1.6),
  );
  renderer.setClearColor(0x152b23);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(43, 1, 0.01, 60);
  if (zUp) camera.up.set(0, 0, 1);
  camera.position.fromArray(position);
  // Software WebGPU on Linux can render correctly while its compositor cannot
  // present swapchain textures. Read back the same WebGPU render target into a
  // visible canvas on software adapters only; hardware uses direct presentation.
  const display = options.software
    ? document.createElement("canvas")
    : renderer.domElement;
  const displayContext = options.software ? display.getContext("2d") : null;
  const renderTarget = options.software
    ? new THREE.RenderTarget(1, 1, { type: THREE.UnsignedByteType })
    : null;
  if (renderTarget) renderTarget.texture.colorSpace = THREE.SRGBColorSpace;
  if (renderTarget) renderer.setRenderTarget(renderTarget);
  const controls = new OrbitControls(camera, display);
  controls.target.fromArray(target);
  controls.enableDamping = true;
  controls.minDistance = 0.15;
  controls.maxDistance = 5;
  controls.update();
  scene.add(new THREE.HemisphereLight(0xe9f6e2, 0x687f61, 2.8));
  const key = new THREE.DirectionalLight(0xfff4db, 3);
  key.position.set(2, 4, 5);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x8fb5d2, 1.4);
  fill.position.set(-2, 1, -1);
  scene.add(fill);
  container.append(display);
  const resize = () => {
    const { width, height } = container.getBoundingClientRect();
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
    renderer.setSize(Math.max(width, 1), Math.max(height, 1), false);
    if (renderTarget) {
      display.width = Math.max(1, Math.round(width));
      display.height = Math.max(1, Math.round(height));
      renderTarget.setSize(display.width, display.height);
    }
  };
  resize();
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  const abort = new AbortController();
  const on = (selector, type, handler) =>
    document
      .querySelector(selector)
      .addEventListener(type, handler, { signal: abort.signal });
  let stopped = false,
    last = performance.now(),
    frames = 0,
    busy = false,
    pending;
  const visible = () =>
    !document.hidden &&
    container.getBoundingClientRect().bottom > 0 &&
    container.getBoundingClientRect().top < window.innerHeight;
  return {
    renderer,
    scene,
    camera,
    controls,
    on,
    start(update) {
      renderer.setAnimationLoop(async (now) => {
        if (stopped || busy) return;
        const dt = Math.min((now - last) / 1000, 0.05);
        last = now;
        if (!visible()) return;
        busy = true;
        try {
          update(dt);
          controls.update();
          renderer.render(scene, camera);
          if (renderTarget) {
            const width = display.width,
              height = display.height;
            pending = renderer.readRenderTargetPixelsAsync(
              renderTarget,
              0,
              0,
              width,
              height,
            );
            const pixels = await pending;
            if (!stopped)
              displayContext.putImageData(
                new ImageData(
                  new Uint8ClampedArray(
                    pixels.buffer,
                    pixels.byteOffset,
                    pixels.byteLength,
                  ),
                  width,
                  height,
                ),
                0,
                0,
              );
          }
          if (!stopped) container.dataset.frames = String(++frames);
        } catch (error) {
          if (!stopped) {
            options.status("Rendering paused · reload demo");
            console.error(error);
          }
        } finally {
          busy = false;
        }
      });
    },
    resetCamera() {
      camera.position.fromArray(position);
      controls.target.fromArray(target);
      controls.update();
    },
    dispose() {
      stopped = true;
      abort.abort();
      observer.disconnect();
      controls.dispose();
      renderer.setAnimationLoop(null);
      const release = () => {
        scene.traverse((object) => {
          object.geometry?.dispose();
          for (const material of object.material
            ? Array.isArray(object.material)
              ? object.material
              : [object.material]
            : []) {
            for (const value of Object.values(material))
              if (value?.isTexture) value.dispose();
            material.dispose();
          }
        });
        renderTarget?.dispose();
        renderer.dispose();
      };
      if (pending) pending.then(release, release);
      else release();
      display.remove();
      delete container.dataset.frames;
    },
  };
}

export function tabletop(scene, width = 1.2, depth = 0.85) {
  const table = new THREE.Mesh(
    new THREE.BoxGeometry(width, 0.055, depth),
    new THREE.MeshStandardMaterial({ color: 0x82946c, roughness: 0.88 }),
  );
  table.position.y = -0.028;
  scene.add(table);
  const grid = new THREE.GridHelper(2.8, 28, 0x617957, 0x2b4836);
  grid.position.y = -0.29;
  scene.add(grid);
  for (const x of [-width * 0.4, width * 0.4])
    for (const z of [-depth * 0.4, depth * 0.4]) {
      const leg = new THREE.Mesh(
        new THREE.CylinderGeometry(0.018, 0.018, 0.23, 10),
        new THREE.MeshStandardMaterial({
          color: 0x4c6250,
          metalness: 0.3,
          roughness: 0.6,
        }),
      );
      leg.position.set(x, -0.17, z);
      scene.add(leg);
    }
}
