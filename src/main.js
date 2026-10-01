const $ = (s) => document.querySelector(s);
const notice = $(".code-notice");
const noticeButton = $(".release-dot");
const noticeText = $("#code-release-note");
let noticePinned = false;
function showNotice(show) {
  noticeText.hidden = !show;
  noticeButton.setAttribute("aria-expanded", String(show));
}
notice.addEventListener("mouseenter", () => showNotice(true));
notice.addEventListener("mouseleave", () => {
  if (!noticePinned && !notice.contains(document.activeElement))
    showNotice(false);
});
notice.addEventListener("focusin", () => showNotice(true));
notice.addEventListener("focusout", (event) => {
  if (!notice.contains(event.relatedTarget)) {
    noticePinned = false;
    showNotice(false);
  }
});
noticeButton.addEventListener("click", () => {
  noticePinned = !noticePinned;
  showNotice(noticePinned);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    noticePinned = false;
    showNotice(false);
  }
});
document.addEventListener("click", (event) => {
  if (!notice.contains(event.target)) {
    noticePinned = false;
    showNotice(false);
  }
});
$("#copy-citation").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("#bibtex").textContent);
    $("#citation-status").textContent = "BibTeX copied to clipboard.";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents($("#bibtex"));
    selection.removeAllRanges();
    selection.addRange(range);
    $("#citation-status").textContent =
      "Citation selected. Copy the selection or download the .bib file.";
  }
});
const dialog = $("#lightbox");
let previousFocus;
function showImage(
  src,
  title,
  description = "PhiRIE reconstruction and interaction results.",
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

fetch("demos.json")
  .then((response) => {
    if (!response.ok) throw new Error("Demos unavailable");
    return response.json();
  })
  .then((videos) => {
    videos.forEach((item, i) => {
      const button = document.createElement("button");
      button.dataset.videoId = item.id;
      button.setAttribute("aria-pressed", String(i === 0));
      const img = document.createElement("img");
      img.src = item.poster;
      img.alt = "";
      img.loading = "lazy";
      const text = document.createElement("span");
      text.textContent = `${item.id} / ${item.title}`;
      const small = document.createElement("small");
      small.textContent = item.subtitle;
      text.append(small);
      button.append(img, text);
      button.addEventListener("click", () => {
        const video = $("#result-video");
        video.pause();
        video.src = item.src;
        video.poster = item.poster;
        video.setAttribute("aria-label", `PhiView: ${item.title}`);
        video.load();
        $("#video-title").textContent = item.title;
        $("#video-type").textContent = `PHIVIEW · ${item.subtitle}`;
        $("#video-counter").textContent = `${item.id} / 09`;
        $("#video-description").textContent = item.description;
        $("#video-download").href = item.src;
        document
          .querySelectorAll(".video-picker button")
          .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      });
      $(".video-picker").append(button);
    });
  })
  .catch(() => {
    $(".video-picker").textContent =
      "Please refresh to load the demo collection.";
  });

document.querySelectorAll("[data-film-source]").forEach((button) =>
  button.addEventListener("click", () => {
    const local = button.dataset.filmSource === "local";
    const frame = $("#overview-youtube");
    $("#overview-local").pause();
    $("#overview-local").hidden = !local;
    frame.hidden = local;
    if (local) {
      frame.removeAttribute("src");
      $("#overview-local").load();
    } else {
      frame.src =
        "https://www.youtube-nocookie.com/embed/3-YdcBh6Tbw?rel=0&playsinline=1";
    }
    document
      .querySelectorAll("[data-film-source]")
      .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
  }),
);

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
