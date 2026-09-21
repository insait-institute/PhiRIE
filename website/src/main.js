const $ = (s) => document.querySelector(s);
const dialog = $("#lightbox");
let previousFocus;
function showImage(
  src,
  title,
  description = "Preserved from the project’s paper figure collection. Source details are available in the asset manifest.",
) {
  previousFocus = document.activeElement;
  $("#lightbox-image").src = src;
  $("#lightbox-image").alt = title;
  $("#lightbox-title").textContent = title;
  $("#lightbox-description").textContent = description;
  dialog.showModal();
  document.body.style.overflow = "hidden";
}
dialog.querySelector("button").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (e) => {
  if (e.target === dialog) dialog.close();
});
dialog.addEventListener("close", () => {
  document.body.style.overflow = "";
  previousFocus?.focus();
});
document
  .querySelectorAll("[data-image]")
  .forEach((b) =>
    b.addEventListener("click", () =>
      showImage(b.dataset.image, b.dataset.title),
    ),
  );

let gallery = [],
  filter = "All",
  query = "",
  limit = 12;
function renderGallery() {
  const matches = gallery.filter(
    (item) =>
      (filter === "All" || item.category === filter) &&
      `${item.title} ${item.scene} ${item.category}`
        .toLowerCase()
        .includes(query),
  );
  $("#gallery").replaceChildren();
  for (const item of matches.slice(0, limit)) {
    const card = document.createElement("button");
    card.className = "result-card";
    const img = document.createElement("img");
    img.src = item.src;
    img.alt = item.title;
    img.loading = "lazy";
    img.decoding = "async";
    const info = document.createElement("span");
    info.className = "card-info";
    for (const [tag, cls, value] of [
      ["span", "card-category", item.category],
      ["strong", "", item.title],
      ["small", "", item.scene],
    ]) {
      const el = document.createElement(tag);
      el.className = cls;
      el.textContent = value;
      info.append(el);
    }
    card.append(img, info);
    card.addEventListener("click", () =>
      showImage(item.src, item.title, item.caption),
    );
    $("#gallery").append(card);
  }
  if (!matches.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent =
      "No results match this search. Try a scene, cup, bottle, or keyboard.";
    $("#gallery").append(empty);
  }
  $("#gallery-count").textContent =
    `${Math.min(limit, matches.length)} of ${matches.length} views · ${gallery.length} visualizations in the collection`;
  $("#load-more").hidden = limit >= matches.length;
}
fetch("gallery.json")
  .then((r) => {
    if (!r.ok) throw new Error("Gallery unavailable");
    return r.json();
  })
  .then((data) => {
    gallery = data;
    renderGallery();
  })
  .catch(() => {
    $("#gallery-count").textContent =
      "The gallery could not load. Please refresh the page.";
    $("#load-more").hidden = true;
  });
document.querySelectorAll("[data-filter]").forEach((b) =>
  b.addEventListener("click", () => {
    filter = b.dataset.filter;
    limit = 12;
    document
      .querySelectorAll("[data-filter]")
      .forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderGallery();
  }),
);
$("#gallery-search").addEventListener("input", (e) => {
  query = e.target.value.toLowerCase().trim();
  limit = 12;
  renderGallery();
});
$("#load-more").addEventListener("click", () => {
  limit += 12;
  renderGallery();
});

const objects = ["headphone", "cup", "keyboard"];
const videoList = objects.flatMap((object) =>
  ["shooting", "robot"].map((family) => ({ object, family })),
);
videoList.forEach(({ object, family }, i) => {
  const label =
    object === "headphone"
      ? "Headphones"
      : object[0].toUpperCase() + object.slice(1);
  const button = document.createElement("button");
  button.setAttribute("aria-pressed", String(i === 0));
  const img = document.createElement("img");
  img.src = `media/${object}-${family === "shooting" ? "06_shooting_1" : "10_robot_2"}.webp`;
  img.alt = "";
  img.loading = "lazy";
  const text = document.createElement("span");
  text.textContent = label;
  const small = document.createElement("small");
  small.textContent =
    family === "shooting" ? "Two impacts · 4 s" : "Robot interaction · 7 s";
  text.append(small);
  button.append(img, text);
  button.addEventListener("click", () => {
    const video = $("#result-video");
    video.pause();
    video.src = `media/${object}_${family}.mp4`;
    video.poster = img.src;
    video.setAttribute("aria-label", `Recorded ${label} ${family} sequence`);
    video.load();
    $("#video-title").textContent =
      `${label} / ${family === "shooting" ? "two impacts" : "robot interaction"}`;
    $("#video-counter").textContent = `0${i + 1} / 06`;
    $("#video-description").textContent =
      family === "shooting"
        ? "Two scripted projectile launches with SAM3D collision geometry. Estimated parameters; a qualitative physics demonstration."
        : object === "cup"
          ? "Recorded scripted push and retract with SAM3D cup collision geometry. This sequence does not establish a successful grasp or learned-policy performance."
          : "Recorded scripted robot motion with SAM3D appearance replacement and the original collision dynamics. No new SAM3D robot dynamics or grasp success is claimed.";
    document
      .querySelectorAll(".video-picker button")
      .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
  });
  $(".video-picker").append(button);
});
$("#compare-slider").addEventListener("input", (e) =>
  $("#comparison").style.setProperty("--split", `${e.target.value}%`),
);
$("#compare-object").addEventListener("change", (e) => {
  $("#compare-before").src = `media/harmony-${e.target.value}-without.webp`;
  $("#compare-after").src = `media/harmony-${e.target.value}-with.webp`;
  $("#compare-before").alt =
    `${e.target.selectedOptions[0].textContent} without harmonization`;
  $("#compare-after").alt =
    `${e.target.selectedOptions[0].textContent} with harmonization`;
});

// Load the rendering stack only after an explicit launch.
let labModule;
const loadLab = () => (labModule ??= import("./lab.js"));
$("#launch-demo").addEventListener("click", async () =>
  (await loadLab()).launch(),
);
document.querySelectorAll("[data-demo]").forEach((button) => {
  button.addEventListener("click", async () =>
    (await loadLab()).selectDemo(button.dataset.demo),
  );
  button.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    const tabs = [...document.querySelectorAll("[data-demo]")];
    let index = tabs.indexOf(button);
    index =
      e.key === "Home"
        ? 0
        : e.key === "End"
          ? tabs.length - 1
          : (index + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) %
            tabs.length;
    tabs[index].focus();
    tabs[index].click();
  });
});
