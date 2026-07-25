(function () {
  "use strict";

  var FAVORITES_KEY = "robotOverlordFavorites";

  function domain(rawURL) {
    try {
      var host = new URL(rawURL).hostname;
      return host.replace(/^www\./, "");
    } catch (e) {
      return "";
    }
  }

  function formatDate(s) {
    if (!s) return "";
    var d = new Date(s);
    if (isNaN(d.getTime())) return s;
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  }

  function loadFavorites() {
    try {
      var raw = localStorage.getItem(FAVORITES_KEY);
      var map = raw ? JSON.parse(raw) : {};
      return Object.keys(map).map(function (url) { return map[url]; });
    } catch (e) {
      return [];
    }
  }

  function savedViewActive() {
    var savedView = document.getElementById("saved-view");
    return !!savedView && !savedView.hidden;
  }

  function fetchNormalArticles() {
    return fetch("/api/present" + window.location.search)
      .then(function (res) {
        if (!res.ok) throw new Error("bad response");
        return res.json();
      })
      .catch(function () {
        return [];
      });
  }

  function getArticlesForPresentation() {
    if (savedViewActive()) {
      return Promise.resolve(loadFavorites());
    }
    return fetchNormalArticles();
  }

  var AUTOPLAY_INTERVAL_MS = 8000;

  var overlay = null;
  var articles = [];
  var index = 0;
  var playing = true;
  var timer = null;
  var progressRAF = null;

  function metaLine(article) {
    var parts = [];
    if (article.Topics && article.Topics.length) parts.push(article.Topics.join(", "));
    var dom = domain(article.URL || "");
    if (dom) parts.push(dom);
    var when = formatDate(article.PublishedOn || article.FetchedAt);
    if (when) parts.push(when);
    return parts.join(" · ");
  }

  function updatePlayPauseButton() {
    var btn = overlay.querySelector(".presentation-playpause");
    if (!btn) return;
    btn.hidden = articles.length < 2;
    btn.textContent = playing ? "⏸" : "▶";
    btn.setAttribute("aria-label", playing ? "Pause autoplay" : "Resume autoplay");
    btn.title = playing ? "Pause autoplay" : "Resume autoplay";
  }

  function getProgressBar() {
    return overlay ? overlay.querySelector(".presentation-progress-bar") : null;
  }

  function stopProgressAnimation() {
    if (progressRAF) {
      cancelAnimationFrame(progressRAF);
      progressRAF = null;
    }
  }

  function resetProgressBar() {
    stopProgressAnimation();
    var bar = getProgressBar();
    if (!bar) return;
    bar.style.width = "0%";
  }

  function tickProgress(startTime) {
    var bar = getProgressBar();
    if (!bar) return;
    var pct = Math.min(100, ((Date.now() - startTime) / AUTOPLAY_INTERVAL_MS) * 100);
    bar.style.width = pct + "%";
    if (pct < 100) {
      progressRAF = requestAnimationFrame(function () { tickProgress(startTime); });
    }
  }

  function animateProgressBar() {
    stopProgressAnimation();
    var bar = getProgressBar();
    if (!bar) return;
    bar.style.width = "0%";
    progressRAF = requestAnimationFrame(function () { tickProgress(Date.now()); });
  }

  function renderSlide() {
    var titleEl = overlay.querySelector(".presentation-title");
    var summaryEl = overlay.querySelector(".presentation-summary");
    var metaEl = overlay.querySelector(".presentation-meta");
    var counterEl = overlay.querySelector(".presentation-counter");
    var prevBtn = overlay.querySelector(".presentation-prev");
    var nextBtn = overlay.querySelector(".presentation-next");

    resetProgressBar();

    if (articles.length === 0) {
      titleEl.textContent = "Nothing to present";
      summaryEl.textContent = "No articles match the current view.";
      metaEl.textContent = "";
      counterEl.textContent = "";
      prevBtn.hidden = true;
      nextBtn.hidden = true;
      updatePlayPauseButton();
      return;
    }

    var a = articles[index];
    titleEl.textContent = a.Title || "(no title)";
    summaryEl.textContent = a.Summary || "";
    metaEl.textContent = metaLine(a);
    counterEl.textContent = (index + 1) + " / " + articles.length;
    prevBtn.hidden = false;
    nextBtn.hidden = false;
    updatePlayPauseButton();
  }

  function stopAutoplay() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    stopProgressAnimation();
  }

  function startAutoplay() {
    stopAutoplay();
    if (!playing || articles.length < 2) return;
    timer = setInterval(showNext, AUTOPLAY_INTERVAL_MS);
    animateProgressBar();
  }

  function showNext() {
    if (articles.length === 0) return;
    index = (index + 1) % articles.length;
    renderSlide();
    startAutoplay();
  }

  function showPrev() {
    if (articles.length === 0) return;
    index = (index - 1 + articles.length) % articles.length;
    renderSlide();
    startAutoplay();
  }

  function togglePlayPause() {
    playing = !playing;
    updatePlayPauseButton();
    startAutoplay();
  }

  function onKeydown(e) {
    if (e.key === "ArrowRight" || e.key === " ") {
      e.preventDefault();
      showNext();
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      showPrev();
    } else if (e.key === "Escape") {
      closePresentation();
    } else if (e.key === "p" || e.key === "P") {
      togglePlayPause();
    }
  }

  function closePresentation() {
    if (!overlay) return;
    stopAutoplay();
    document.removeEventListener("keydown", onKeydown);
    overlay.remove();
    overlay = null;
  }

  function buildOverlay() {
    var el = document.createElement("div");
    el.className = "presentation-overlay";
    el.innerHTML =
      '<button class="presentation-close" type="button" aria-label="Close presentation">✕</button>' +
      '<div class="presentation-progress"><div class="presentation-progress-bar"></div></div>' +
      '<div class="presentation-meta"></div>' +
      '<div class="presentation-title"></div>' +
      '<div class="presentation-summary"></div>' +
      '<div class="presentation-controls">' +
      '<button class="presentation-prev" type="button" aria-label="Previous article">&larr;</button>' +
      '<button class="presentation-playpause" type="button" aria-label="Pause autoplay">⏸</button>' +
      '<span class="presentation-counter"></span>' +
      '<button class="presentation-next" type="button" aria-label="Next article">&rarr;</button>' +
      "</div>";

    el.querySelector(".presentation-close").addEventListener("click", closePresentation);
    el.querySelector(".presentation-prev").addEventListener("click", showPrev);
    el.querySelector(".presentation-next").addEventListener("click", showNext);
    el.querySelector(".presentation-playpause").addEventListener("click", togglePlayPause);

    return el;
  }

  function openPresentation() {
    if (overlay) return;
    overlay = buildOverlay();
    document.body.appendChild(overlay);
    document.addEventListener("keydown", onKeydown);

    index = 0;
    articles = [];
    playing = true;
    renderSlide();

    getArticlesForPresentation().then(function (list) {
      articles = list || [];
      index = 0;
      if (!overlay) return;
      renderSlide();
      startAutoplay();
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("presentation-toggle");
    if (!btn) return;
    btn.addEventListener("click", openPresentation);
  });
})();
