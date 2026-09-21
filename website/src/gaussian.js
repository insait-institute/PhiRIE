import { PerspectiveCamera, Matrix4, Vector3 } from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const shader = /* wgsl */ `
struct Uniforms { view: mat4x4f, projection: mat4x4f, viewport: vec4f, params: vec4f, center: vec4f };
struct Splat { position: vec4f, color: vec4f, covarianceA: vec4f, covarianceB: vec4f };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
struct Output { @builtin(position) position: vec4f, @location(0) local: vec2f, @location(1) color: vec4f };
@vertex fn vertexMain(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Output {
  let s = splats[order[instance]];
  let group = i32(s.position.w); let mode = i32(u.params.x);
  var out: Output;
  out.position = vec4f(0,0,2,1); out.local=vec2f(0);out.color=vec4f(0);
  if ((group == 1 && mode != 0) || (group == 2 && mode < 2) || (group == 3 && mode != 3)) { return out; }
  var p = s.position.xyz;
  var covariance = mat3x3f(vec3f(s.covarianceA.x,s.covarianceA.y,s.covarianceA.z),
    vec3f(s.covarianceA.y,s.covarianceA.w,s.covarianceB.x),vec3f(s.covarianceA.z,s.covarianceB.x,s.covarianceB.y));
  // Widen background kernels for the downsampled browser preview.
  if (group == 0) { covariance *= 1.96; }
  if (group == 3) {
    let c=cos(u.params.w);let sn=sin(u.params.w);
    let r=mat3x3f(vec3f(c,sn,0),vec3f(-sn,c,0),vec3f(0,0,1));
    p=r*(p-u.center.xyz)+u.center.xyz+vec3f(u.params.y,u.params.z,0);
    covariance=r*covariance*transpose(r);
  }
  let viewP = (u.view*vec4f(p,1)).xyz;
  if (viewP.z > -0.04) { return out; }
  let clip = u.projection*vec4f(viewP,1);
  let rotation = mat3x3f(u.view[0].xyz,u.view[1].xyz,u.view[2].xyz);
  let cov=rotation*covariance*transpose(rotation);
  let fx=u.projection[0].x*u.viewport.x*.5; let fy=u.projection[1].y*u.viewport.y*.5;
  let jx=vec3f(fx/-viewP.z,0,fx*viewP.x/(viewP.z*viewP.z));
  let jy=vec3f(0,fy/-viewP.z,fy*viewP.y/(viewP.z*viewP.z));
  let a=dot(jx,cov*jx)+.3;let b=dot(jx,cov*jy);let d=dot(jy,cov*jy)+.3;
  let mid=(a+d)*.5;let radius=sqrt(max(.0001,(a-d)*(a-d)*.25+b*b));
  let l1=min(20000.0,max(.1,mid+radius));let l2=min(20000.0,max(.1,mid-radius));
  var axis=vec2f(1,0);if (abs(b)>.0001) {axis=normalize(vec2f(b,l1-a));} else if(d>a){axis=vec2f(0,1);}
  let quad=array<vec2f,6>(vec2f(-3,-3),vec2f(3,-3),vec2f(3,3),vec2f(-3,-3),vec2f(3,3),vec2f(-3,3));
  let corner=quad[vertex];
  let offset=axis*sqrt(l1)*corner.x+vec2f(-axis.y,axis.x)*sqrt(l2)*corner.y;
  out.position=vec4f(clip.xy/clip.w+offset*2/u.viewport.xy,.5,1);
  out.local=corner;out.color=s.color;return out;
}
@fragment fn fragmentMain(in: Output) -> @location(0) vec4f {
  let radius=dot(in.local,in.local); if(radius>9){discard;}
  let alpha=min(.99,in.color.a*exp(-.5*radius));if(alpha<.003){discard;}
  return vec4f(in.color.rgb*alpha,alpha);
}`;

export async function createDemo(container, options) {
  const [meta, raw] = await Promise.all([
    fetch("models/desk.json").then((r) => {
      if (!r.ok) throw new Error("Scene metadata failed to load");
      return r.json();
    }),
    fetch("models/desk.splat").then((r) => {
      if (!r.ok) throw new Error("Gaussian scene failed to load");
      return r.arrayBuffer();
    }),
  ]);
  if (!options.isCurrent()) throw new Error("Demo loading cancelled.");
  const device = options.device;
  const canvas = document.createElement("canvas");
  canvas.setAttribute("aria-label", "Interactive 3D Gaussian scene");
  container.append(canvas);
  const context = options.software ? null : canvas.getContext("webgpu");
  const displayContext = options.software ? canvas.getContext("2d") : null;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context?.configure({ device, format, alphaMode: "opaque" });
  const module = device.createShaderModule({ code: shader });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((m) => m.type === "error");
  if (errors.length) {
    canvas.remove();
    throw new Error(errors.map((e) => e.message).join("; "));
  }
  const pipeline = await device.createRenderPipelineAsync({
    layout: "auto",
    vertex: { module, entryPoint: "vertexMain" },
    fragment: {
      module,
      entryPoint: "fragmentMain",
      targets: [
        {
          format,
          blend: {
            color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        },
      ],
    },
    primitive: { topology: "triangle-list" },
  });
  const points = new Float32Array(raw);
  const count = points.length / 16;
  const dataBuffer = device.createBuffer({
    size: raw.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(dataBuffer, 0, raw);
  const orderBuffer = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const uniformBuffer = device.createBuffer({
    size: 176,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: dataBuffer } },
      { binding: 2, resource: { buffer: orderBuffer } },
    ],
  });
  const camera = new PerspectiveCamera(68, 1, 0.03, 30);
  camera.up.set(0, 0, 1);
  const source = new Matrix4()
    .fromArray(meta.camera.w2c.flat())
    .transpose()
    .invert();
  const home = new Vector3().setFromMatrixPosition(source);
  // Frame the real source desk while retaining a comfortable orbit pivot.
  const target = new Vector3(...meta.center);
  home.sub(target).multiplyScalar(1.5).add(target);
  camera.position.copy(home);
  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(target);
  controls.minDistance = 0.25;
  controls.maxDistance = 5;
  controls.enableDamping = false;
  controls.update();
  const abort = new AbortController();
  const on = (id, event, fn) =>
    document
      .querySelector(id)
      .addEventListener(event, fn, { signal: abort.signal });
  let dirty = true,
    stopped = false,
    frame = 0,
    texture,
    readback,
    busy = false;
  const params = { stage: 3, x: 0, y: 0, angle: 0 };
  const resize = () => {
    const r = container.getBoundingClientRect(),
      ratio = options.software ? 1 : Math.min(devicePixelRatio, 1.5);
    canvas.width = Math.max(1, Math.round(r.width * ratio));
    canvas.height = Math.max(1, Math.round(r.height * ratio));
    camera.aspect = r.width / r.height;
    camera.updateProjectionMatrix();
    dirty = true;
  };
  resize();
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  controls.addEventListener("change", () => (dirty = true));
  on("#scene-stage", "change", (e) => {
    params.stage = +e.target.value;
    dirty = true;
  });
  for (const axis of ["x", "y", "angle"])
    on(`#scene-${axis}`, "input", (e) => {
      params[axis] = +e.target.value;
      document.querySelector(`#scene-${axis}-value`).textContent =
        `${e.target.value}${axis === "angle" ? "°" : " cm"}`;
      params.stage = 3;
      document.querySelector("#scene-stage").value = "3";
      dirty = true;
    });
  on("#reset-demo", "click", () => {
    Object.assign(params, { stage: 3, x: 0, y: 0, angle: 0 });
    camera.position.copy(home);
    controls.target.copy(target);
    controls.update();
    document.querySelector("#scene-stage").value = "3";
    for (const axis of ["x", "y", "angle"]) {
      document.querySelector(`#scene-${axis}`).value = 0;
      document.querySelector(`#scene-${axis}-value`).textContent =
        axis === "angle" ? "0°" : "0 cm";
    }
    dirty = true;
  });
  const order = new Uint32Array(count);
  const depths = new Float32Array(count);
  const uniforms = new Float32Array(44);
  async function draw() {
    const rendered = { ...params };
    camera.updateMatrixWorld();
    const view = camera.matrixWorldInverse.elements;
    const angle = (rendered.angle * Math.PI) / 180,
      c = Math.cos(angle),
      s = Math.sin(angle);
    for (let i = 0; i < count; i++) {
      const k = i * 16;
      let x = points[k],
        y = points[k + 1];
      if (points[k + 3] === 3) {
        const dx = x - meta.center[0],
          dy = y - meta.center[1];
        x = c * dx - s * dy + meta.center[0] + rendered.x / 100;
        y = s * dx + c * dy + meta.center[1] + rendered.y / 100;
      }
      depths[i] =
        view[2] * x + view[6] * y + view[10] * points[k + 2] + view[14];
      order[i] = i;
    }
    order.sort((a, b) => depths[a] - depths[b]);
    device.queue.writeBuffer(orderBuffer, 0, order);
    uniforms.set(view, 0);
    uniforms.set(camera.projectionMatrix.elements, 16);
    uniforms.set([canvas.width, canvas.height, 0, 0], 32);
    uniforms.set(
      [rendered.stage, rendered.x / 100, rendered.y / 100, angle],
      36,
    );
    uniforms.set([...meta.center, 0], 40);
    device.queue.writeBuffer(uniformBuffer, 0, uniforms);
    const width = canvas.width,
      height = canvas.height,
      bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    if (options.software) {
      texture?.destroy();
      readback?.destroy();
      texture = device.createTexture({
        size: [width, height],
        format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      readback = device.createBuffer({
        size: bytesPerRow * height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
    const command = device.createCommandEncoder();
    const pass = command.beginRenderPass({
      colorAttachments: [
        {
          view: (texture || context.getCurrentTexture()).createView(),
          clearValue: { r: 0.075, g: 0.12, b: 0.09, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.draw(6, count);
    pass.end();
    if (readback)
      command.copyTextureToBuffer(
        { texture },
        { buffer: readback, bytesPerRow },
        [width, height],
      );
    device.queue.submit([command.finish()]);
    if (readback) {
      await readback.mapAsync(GPUMapMode.READ);
      const rawPixels = new Uint8Array(readback.getMappedRange()),
        pixels = new Uint8ClampedArray(width * height * 4);
      for (let y = 0; y < height; y++)
        pixels.set(
          rawPixels.subarray(y * bytesPerRow, y * bytesPerRow + width * 4),
          y * width * 4,
        );
      if (format.startsWith("bgra"))
        for (let i = 0; i < pixels.length; i += 4) {
          const r = pixels[i];
          pixels[i] = pixels[i + 2];
          pixels[i + 2] = r;
        }
      readback.unmap();
      if (!stopped)
        displayContext.putImageData(new ImageData(pixels, width, height), 0, 0);
    }
    if (!stopped) {
      container.dataset.frames = String(++frame);
      container.dataset.gaussians = String(count);
      container.dataset.sceneStage = String(rendered.stage);
      container.dataset.scenePose = JSON.stringify([
        rendered.x,
        rendered.y,
        rendered.angle,
      ]);
    }
  }
  let raf, pending;
  const loop = () => {
    if (stopped) return;
    if (dirty && !document.hidden && !busy) {
      dirty = false;
      busy = true;
      pending = draw()
        .catch((e) => {
          if (!stopped) {
            options.status("Rendering paused · reload demo");
            console.error(e);
          }
        })
        .finally(() => (busy = false));
    }
    raf = requestAnimationFrame(loop);
  };
  loop();
  device.lost.then((info) => {
    if (!stopped) {
      options.status("WebGPU device lost · reload the demo");
      console.warn(info.message);
    }
  });
  return {
    dispose() {
      stopped = true;
      cancelAnimationFrame(raf);
      abort.abort();
      observer.disconnect();
      controls.dispose();
      const release = () => {
        dataBuffer.destroy();
        orderBuffer.destroy();
        uniformBuffer.destroy();
        texture?.destroy();
        readback?.destroy();
        context?.unconfigure();
      };
      if (pending) pending.then(release, release);
      else release();
      canvas.remove();
      delete container.dataset.frames;
      delete container.dataset.gaussians;
      delete container.dataset.sceneStage;
    },
  };
}
