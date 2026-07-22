const pageCopy = {
  zh: {
    title: "基于稠密三维几何的多视图 Gaussian 初始化",
    description: "基于像素对齐稠密三维几何的区域级各向异性 Gaussian 初始化方法。",
  },
  en: {
    title: "Multi-view Gaussian Initialization from Dense 3D Geometry",
    description: "A region-level method for anisotropic Gaussian initialization from pixel-aligned dense 3D geometry.",
  },
};

function currentLanguage() {
  return document.documentElement.dataset.language === "en" ? "en" : "zh";
}

function setLanguage(language, persist = true) {
  const selected = language === "en" ? "en" : "zh";
  document.documentElement.dataset.language = selected;
  document.documentElement.lang = selected === "zh" ? "zh-CN" : "en";
  document.title = pageCopy[selected].title;
  document
    .querySelector('meta[name="description"]')
    ?.setAttribute("content", pageCopy[selected].description);

  document.querySelectorAll("[data-set-language]").forEach((button) => {
    const active = button.dataset.setLanguage === selected;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });

  if (persist) {
    try {
      localStorage.setItem("gaussian-init-language", selected);
    } catch (_) {}
  }
}

document.querySelectorAll("[data-set-language]").forEach((button) => {
  button.addEventListener("click", () => setLanguage(button.dataset.setLanguage));
});

setLanguage(currentLanguage(), false);

if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  document.querySelectorAll("video[autoplay]").forEach((video) => video.pause());
}
