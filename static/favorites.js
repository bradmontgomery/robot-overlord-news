(function () {
  "use strict";

  var STORAGE_KEY = "robotOverlordFavorites";

  var TOPIC_COLORS = [
    "#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444",
    "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
  ];

  function topicColor(topic) {
    // Mirrors main.go's topicColor: Go's `int` is 64-bit and wraps on
    // overflow, so BigInt + asIntN(64, ...) is needed to match exactly.
    var h = 0n;
    var lower = topic.toLowerCase();
    for (var i = 0; i < lower.length; i++) {
      h = BigInt.asIntN(64, h * 31n + BigInt(lower.charCodeAt(i)));
    }
    if (h < 0n) h = -h;
    return TOPIC_COLORS[Number(h % BigInt(TOPIC_COLORS.length))];
  }

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
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  }

  function loadFavorites() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (e) {
      return {};
    }
  }

  function saveFavorites(map) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  }

  var favorites = loadFavorites();

  function updateCount() {
    var countEl = document.getElementById("saved-count");
    if (countEl) countEl.textContent = String(Object.keys(favorites).length);
  }

  function setButtonState(btn, isFavorited) {
    btn.textContent = isFavorited ? "♥" : "♡";
    btn.classList.toggle("favorited", isFavorited);
    btn.setAttribute("aria-label", isFavorited ? "Remove from saved" : "Save article");
    btn.title = isFavorited ? "Remove from saved" : "Save article";
  }

  function initButtons() {
    var buttons = document.querySelectorAll("#normal-view .favorite-btn");
    buttons.forEach(function (btn) {
      var article;
      try {
        article = JSON.parse(btn.getAttribute("data-article"));
      } catch (e) {
        return;
      }
      setButtonState(btn, !!favorites[article.URL]);
      btn.addEventListener("click", function () {
        toggleFavorite(article, btn);
      });
    });
  }

  function applyToggle(article) {
    var isFavorited = !!favorites[article.URL];
    if (isFavorited) {
      delete favorites[article.URL];
    } else {
      favorites[article.URL] = article;
    }
    saveFavorites(favorites);
    syncNormalViewButton(article.URL, !isFavorited);
    updateCount();
    if (!document.getElementById("saved-view").hidden) {
      renderSavedView();
    }
    return !isFavorited;
  }

  function toggleFavorite(article, btn) {
    var nowFavorited = applyToggle(article);
    if (btn) setButtonState(btn, nowFavorited);
  }

  function syncNormalViewButton(url, isFavorited) {
    var buttons = document.querySelectorAll("#normal-view .favorite-btn");
    buttons.forEach(function (btn) {
      var article;
      try {
        article = JSON.parse(btn.getAttribute("data-article"));
      } catch (e) {
        return;
      }
      if (article.URL === url) setButtonState(btn, isFavorited);
    });
  }

  function articleCardHTML(article) {
    var topics = article.Topics || [];
    var topicHTML = topics
      .map(function (t) {
        return (
          '<span class="topic-pill" style="background:' +
          topicColor(t) +
          '">' +
          escapeHTML(t) +
          "</span>"
        );
      })
      .join("");

    var dom = domain(article.URL || "");
    var fetched = formatDate(article.FetchedAt);
    var published = article.PublishedOn ? formatDate(article.PublishedOn) : "";

    return (
      '<div class="article">' +
      '<button class="favorite-btn favorited" type="button" aria-label="Remove from saved" title="Remove from saved">♥</button>' +
      '<div class="article-meta">' +
      topicHTML +
      (published ? '<span class="published-on" title="Published on ' + escapeHTML(published) + '">' + escapeHTML(published) + "</span>" : "") +
      (fetched ? '<span class="fetched-at" title="Article fetched on ' + escapeHTML(fetched) + '">' + escapeHTML(fetched) + "</span>" : "") +
      "</div>" +
      "<h3>" +
      (article.URL
        ? '<a href="' + escapeAttr(article.URL) + '" target="_blank" rel="noopener noreferrer">' + escapeHTML(article.Title || "(no title)") + "</a>"
        : escapeHTML(article.Title || "(no title)")) +
      "</h3>" +
      (dom ? '<span class="article-domain">' + escapeHTML(dom) + "</span>" : "") +
      (article.Summary ? "<p>" + escapeHTML(article.Summary) + "</p>" : "") +
      "</div>"
    );
  }

  function escapeHTML(s) {
    var div = document.createElement("div");
    div.textContent = s == null ? "" : String(s);
    return div.innerHTML;
  }

  function escapeAttr(s) {
    return escapeHTML(s).replace(/"/g, "&quot;");
  }

  function renderSavedView() {
    var savedView = document.getElementById("saved-view");
    var urls = Object.keys(favorites);

    if (urls.length === 0) {
      savedView.innerHTML = '<div class="no-articles"><p>No saved articles yet.</p></div>';
      attachSavedViewHandlers();
      return;
    }

    savedView.innerHTML = urls.map(function (url) {
      return articleCardHTML(favorites[url]);
    }).join("");

    attachSavedViewHandlers();
  }

  function attachSavedViewHandlers() {
    var savedView = document.getElementById("saved-view");
    var buttons = savedView.querySelectorAll(".favorite-btn");
    var urls = Object.keys(favorites);
    buttons.forEach(function (btn, i) {
      var url = urls[i];
      btn.addEventListener("click", function () {
        toggleFavorite(favorites[url], null);
      });
    });
  }

  function showSavedView() {
    document.getElementById("normal-view").hidden = true;
    document.getElementById("saved-view").hidden = false;
    renderSavedView();
  }

  function showNormalView() {
    document.getElementById("saved-view").hidden = true;
    document.getElementById("normal-view").hidden = false;
  }

  window.RobotOverlordFavorites = {
    isFavorite: function (url) {
      return !!favorites[url];
    },
    toggle: function (article) {
      return applyToggle(article);
    },
    list: function () {
      return Object.keys(favorites).map(function (url) {
        return favorites[url];
      });
    },
  };

  document.addEventListener("DOMContentLoaded", function () {
    initButtons();
    updateCount();

    var toggle = document.getElementById("saved-view-toggle");
    if (toggle) {
      toggle.addEventListener("click", function (e) {
        e.preventDefault();
        var savedView = document.getElementById("saved-view");
        if (savedView.hidden) {
          showSavedView();
        } else {
          showNormalView();
        }
      });
    }
  });
})();
