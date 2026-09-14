document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("navToggle");
  const nav = document.getElementById("siteNav");
  if (toggle && nav) {
    toggle.addEventListener("click", () => nav.classList.toggle("open"));
  }

  document.querySelectorAll(".card-hover").forEach((card) => {
    card.addEventListener("mouseenter", () => card.classList.add("is-hovered"));
    card.addEventListener("mouseleave", () => card.classList.remove("is-hovered"));
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
  );

  document.querySelectorAll(".fade-up:not(.in-view)").forEach((el) => observer.observe(el));
});
