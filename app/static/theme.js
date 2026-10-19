// Light/dark switch shared by every page. The choice is saved per browser.
(() => {
  const root = document.documentElement;
  const btn = document.getElementById("theme-toggle");
  const current = () =>
    root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  btn?.addEventListener("click", () => {
    const next = current() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked: still switches */ }
    window.dispatchEvent(new CustomEvent("themechange", { detail: next }));
  });
})();
