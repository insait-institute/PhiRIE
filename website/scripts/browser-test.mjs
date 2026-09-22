import { chromium } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";

const base = (process.env.BASE_URL || "http://127.0.0.1:4173/PhiRIE/").replace(
  /\/?$/,
  "/",
);
const output = process.env.OUTPUT_DIR || "test-results";
await fs.mkdir(output, { recursive: true });
const report = {
  timestamp: new Date().toISOString(),
  base,
  checks: [],
  errors: [],
  warnings: [],
  adapter: null,
  limitations: [
    "Software WebGPU adapter; physical GPU/browser performance is not measured.",
  ],
};
const browser = await chromium.launch({
  headless: true,
  args: [
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    "--use-angle=vulkan",
    "--enable-features=Vulkan",
    "--disable-vulkan-surface",
  ],
});
const context = await browser.newContext({
  viewport: { width: 1440, height: 1050 },
});
const page = await context.newPage();
page.on("pageerror", (e) => report.errors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") {
    const origin = m.location().url;
    if (origin && !origin.startsWith(base)) report.warnings.push(m.text());
    else report.errors.push(m.text());
  }
  if (m.type() === "warning") report.warnings.push(m.text());
});
const record = (name, detail) => {
  report.checks.push({ name, status: "PASS", detail });
  console.log("PASS", name);
};
const setRange = async (selector, value) =>
  page.locator(selector).evaluate((el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }, value);
const shot = async (name) =>
  page
    .locator("#lab-panel")
    .screenshot({ path: path.join(output, name + ".png") });
try {
  const requests = [];
  page.on("request", (r) => requests.push(r.url()));
  await page.goto(base, { waitUntil: "networkidle" });
  await page.waitForSelector(".result-card");
  await page.waitForSelector(".video-picker button");
  assert.equal(await page.locator(".result-card").count(), 12);
  assert(!requests.some((r) => r.includes("/models/")));
  record(
    "Landing page and deferred 3D assets",
    "12 initial gallery cards; no model downloads before launch",
  );
  await page.screenshot({ path: path.join(output, "desktop.png") });
  assert.equal(await page.locator(".hero-summary + .featured-film").count(), 1);
  assert.equal(
    await page.locator(".contact a").getAttribute("href"),
    "mailto:runyi.yang@insait.ai",
  );
  assert.equal(
    (await page.locator(".corresponding").innerText()).trim(),
    "∗ Corresponding author",
  );
  assert(
    (await page
      .locator('a[href="https://github.com/RunyiYang/PhysicalView"]')
      .count()) >= 1,
  );
  assert.equal(
    await page
      .locator('a[href*="provenance"], img[src*="paper-uncertainty"]')
      .count(),
    0,
  );
  assert(
    !/How to read the demonstrations|View source manifest/.test(
      await page.locator("body").innerText(),
    ),
  );
  assert.deepEqual(await page.locator(".version-number").allTextContents(), [
    "0.0.0",
    "1.0.0",
    "2.0.0",
    "2.0.1",
  ]);
  assert.equal(
    await page.locator("main > section:last-child").getAttribute("id"),
    "versions",
  );
  await page
    .locator("#versions")
    .screenshot({ path: path.join(output, "version-tree.png") });
  record(
    "Page content and development tree",
    "Contact, PhiView link, video below TL;DR, requested removals, and four milestones",
  );
  const gallery = await page.evaluate(() =>
    fetch("gallery.json").then((r) => r.json()),
  );
  const demos = await page.evaluate(() =>
    fetch("demos.json").then((r) => r.json()),
  );
  assert.equal(demos.length, 9);
  assert.equal(gallery.length, 45);
  const files = [
    ...new Set(gallery.map((x) => x.src)),
    ...demos.flatMap((d) => [d.src, d.poster]),
    "media/phirie-demo.mp4",
    "media/spray-mass-friction-2x2.mp4",
    "media/spray-mass-friction-impact.png",
    "media/spray-mass-friction-final.png",
    "media/paper-qualitative.webp",
    "media/paper-teaser.webp",
  ];
  const failures = [];
  for (const file of files) {
    const r = await page.request.get(new URL(file, base).href);
    if (!r.ok()) failures.push([file, r.status()]);
  }
  assert.deepEqual(failures, []);
  record("Media and download links", `${files.length} URLs return HTTP 200`);
  await page.click("[data-filter=Objects]");
  assert.equal(await page.locator(".result-card").count(), 3);
  await page.click("[data-filter=All]");
  await page.fill("#gallery-search", "keyboard");
  assert((await page.locator("#gallery-count").innerText()).includes("6 of 6"));
  await page.locator(".result-card").first().click();
  assert(await page.locator("#lightbox").isVisible());
  assert(
    await page
      .locator("#lightbox-image")
      .evaluate((im) => im.complete && im.naturalWidth > 0),
  );
  await page.keyboard.press("Escape");
  assert(!(await page.locator("#lightbox").isVisible()));
  await page.fill("#gallery-search", "nonexistent-123");
  assert(await page.locator(".empty-state").isVisible());
  await page.fill("#gallery-search", "");
  await page.click("#load-more");
  assert.equal(await page.locator(".result-card").count(), 24);
  record(
    "Gallery controls",
    "Filtering, search, empty state, pagination, image dialog, Escape dismissal",
  );
  await page.selectOption("#compare-object", "keyboard");
  await setRange("#compare-slider", 27);
  assert(
    (await page.locator("#compare-after").getAttribute("src")).includes(
      "keyboard",
    ),
  );
  assert.equal(
    await page
      .locator("#comparison")
      .evaluate((el) => el.style.getPropertyValue("--split")),
    "27%",
  );
  record("Matched-view comparison", "Object selection and divider controls");
  for (let i = 0; i < 9; i++) {
    await page.locator(".video-picker button").nth(i).click();
    await page.waitForFunction(
      () => document.querySelector("#result-video").readyState >= 1,
    );
    const data = await page
      .locator("#result-video")
      .evaluate((v) => ({ duration: v.duration, error: v.error?.message }));
    assert(Math.abs(data.duration - demos[i].duration) < 0.1);
    assert(!data.error);
    await page.locator("#result-video").evaluate((v) => v.play());
    await page.waitForFunction(
      () => document.querySelector("#result-video").currentTime > 0.2,
    );
    await page.locator("#result-video").evaluate((v) => v.pause());
  }
  await page.locator("#result-video").evaluate((v) => v.play());
  await page.waitForFunction(
    () => document.querySelector("#result-video").currentTime > 0.2,
  );
  await page.locator("#result-video").evaluate((v) => v.pause());
  record(
    "Nine recorded videos",
    "Every video decoded and played; durations match the delivered recordings",
  );
  await page.click('[data-film-source="local"]');
  await page.locator("#overview-local").evaluate((v) => {
    v.muted = true;
    return v.play();
  });
  await page.waitForFunction(
    () => document.querySelector("#overview-local").currentTime > 0.2,
  );
  assert(
    await page
      .locator("#overview-local")
      .evaluate((v) => v.duration > 178 && v.duration < 179),
  );
  await page.locator("#overview-local").evaluate((v) => v.pause());
  await page.click('[data-film-source="youtube"]');
  assert(
    (await page.locator("#overview-youtube").getAttribute("src")).includes(
      "3-YdcBh6Tbw",
    ),
  );
  await page.locator("#parameter-video").evaluate((v) => {
    v.muted = true;
    return v.play();
  });
  await page.waitForFunction(
    () => document.querySelector("#parameter-video").currentTime > 0.2,
  );
  await page.locator("#parameter-video").evaluate((v) => {
    v.pause();
    v.currentTime = 4.5;
  });
  await page
    .locator("#physics-comparison")
    .screenshot({ path: path.join(output, "parameter-comparison.png") });
  record(
    "Overview fallback and parameter comparison",
    "YouTube source selected; direct overview and 2x2 comparison played",
  );
  report.adapter = await page.evaluate(async () => {
    const a = await navigator.gpu.requestAdapter();
    return {
      vendor: a.info.vendor,
      architecture: a.info.architecture,
      isFallbackAdapter: a.info.isFallbackAdapter,
    };
  });
  for (const name of ["scene", "physics", "robot"]) {
    await page.click(`[data-demo=${name}]`);
    await page.click("#launch-demo");
    await page.waitForFunction(
      (n) => document.querySelector("#lab-panel").dataset.activeDemo === n,
      name,
      { timeout: 60000 },
    );
    await page.waitForFunction(
      () => +document.querySelector("#viewport").dataset.frames > 0,
      null,
      { timeout: 60000 },
    );
    assert(
      (await page.locator("#gpu-status").innerText()).includes("WebGPU active"),
    );
    if (name === "scene") {
      const before = await page.locator("#viewport canvas").screenshot();
      for (const value of ["0", "1", "2", "3"]) {
        await page.selectOption("#scene-stage", value);
        await page.waitForFunction(
          (v) => document.querySelector("#viewport").dataset.sceneStage === v,
          value,
          { timeout: 60000 },
        );
      }
      const old = await page.locator("#viewport").getAttribute("data-frames");
      await setRange("#scene-x", 25);
      await setRange("#scene-angle", 80);
      await page.waitForFunction(
        () =>
          document.querySelector("#viewport").dataset.scenePose === "[25,0,80]",
        null,
        { timeout: 60000 },
      );
      assert.notDeepEqual(
        before,
        await page.locator("#viewport canvas").screenshot(),
      );
      await page.click("#reset-demo");
      assert.equal(await page.inputValue("#scene-x"), "0");
      await shot("scene");
      record(
        "WebGPU scene editing",
        "Native WGSL Gaussian rendering; all four stages, movement, rotation, reset; rendered pixels change",
      );
    } else if (name === "physics") {
      await page.waitForFunction(
        () => +document.querySelector("#viewport").dataset.physicsTime > 0.7,
      );
      assert(
        +(await page.locator("#viewport").getAttribute("data-physics-y")) <
          0.12,
      );
      await page.click("#physics-shoot");
      await page.waitForFunction(
        () =>
          Math.abs(+document.querySelector("#viewport").dataset.physicsX) >
          0.01,
        {},
        { timeout: 20000 },
      );
      await setRange("#physics-mass", 1.2);
      assert.equal(
        await page.locator("#physics-mass-value").innerText(),
        "1.2 kg",
      );
      for (const object of ["headphone", "keyboard", "cup"]) {
        await page.selectOption("#physics-object", object);
        await page.waitForFunction(
          (o) =>
            document.querySelector("#viewport").dataset.physicsObject === o,
          object,
        );
      }
      await page.click("#physics-throw");
      assert.equal(
        await page.locator("#viewport").getAttribute("data-throw-count"),
        "1",
      );
      await page.check("#physics-hull");
      await shot("physics");
      await page.click("#reset-demo");
      record(
        "WebGPU physics interaction",
        "Gravity settles object; projectile moves target; mass, object switching, throw, convex-hull display and reset",
      );
    } else {
      await page.click("#robot-contact");
      assert(
        +(await page.locator("#viewport").getAttribute("data-robot-time")) >=
          2.838,
      );
      await setRange("#robot-time", 150);
      assert.equal(
        await page.locator("#viewport").getAttribute("data-robot-frame"),
        "150",
      );
      await page.click("#robot-play");
      await page.waitForFunction(
        () => +document.querySelector("#viewport").dataset.robotFrame > 150,
      );
      await page.click("#robot-play");
      await page.uncheck("#robot-path");
      await page.check("#robot-wire");
      await page.uncheck("#robot-wire");
      await shot("robot");
      await page.click("#reset-demo");
      assert.equal(
        await page.locator("#viewport").getAttribute("data-robot-frame"),
        "0",
      );
      record(
        "WebGPU robot replay",
        "213-state timeline, exact contact jump, playback, trajectory toggle, wireframe, reset",
      );
    }
  }
  await page.click("[data-demo=scene]");
  const a11y = await new AxeBuilder({ page })
    .exclude("#overview-youtube")
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  report.accessibility = a11y.violations.map((v) => ({
    id: v.id,
    impact: v.impact,
    description: v.description,
    nodes: v.nodes.map((n) => ({
      target: n.target,
      summary: n.failureSummary,
    })),
  }));
  assert.equal(a11y.violations.length, 0);
  record(
    "Accessibility audit",
    "No WCAG A/AA violations in page-owned content; the external YouTube player is outside this audit",
  );
  const mobile = await browser.newPage({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
  });
  await mobile.goto(base);
  await mobile.waitForSelector(".result-card");
  await mobile.evaluate(() => document.fonts.ready);
  assert(
    await mobile.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  );
  await mobile.screenshot({
    path: path.join(output, "mobile.png"),
    fullPage: true,
  });
  await mobile.click("[data-demo=robot]");
  await mobile
    .locator("#lab-panel")
    .screenshot({ path: path.join(output, "mobile-playground.png") });
  assert(
    await mobile.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  );
  record(
    "Mobile layout",
    "390px viewport, touch controls, no horizontal overflow",
  );
  await mobile.close();
  const fallback = await browser.newPage();
  await fallback.addInitScript(() =>
    Object.defineProperty(navigator, "gpu", {
      value: undefined,
      configurable: true,
    }),
  );
  await fallback.goto(base);
  await fallback.click("#launch-demo");
  await fallback.waitForSelector(".gpu-error");
  assert.equal(
    await fallback.locator("#gpu-status").innerText(),
    "WebGPU unavailable",
  );
  assert(await fallback.locator('.gpu-error a[href="#motion"]').isVisible());
  record(
    "Unavailable WebGPU fallback",
    "Recorded-video link and understandable support message",
  );
  await fallback.close();
  assert.deepEqual(report.errors, []);
  record("Browser runtime", "No uncaught exceptions or console errors");
  report.status = "PASS";
} catch (error) {
  report.status = "FAIL";
  report.failure = error.stack;
  console.error(error);
  process.exitCode = 1;
  await page
    .screenshot({ path: path.join(output, "failure.png"), fullPage: true })
    .catch(() => {});
} finally {
  await fs.writeFile(
    path.join(output, "validation.json"),
    JSON.stringify(report, null, 2) + "\n",
  );
  await browser.close();
}
