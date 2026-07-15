(function () {
  "use strict";

  function escapeHTML(s) {
    var div = document.createElement("div");
    div.textContent = s == null ? "" : String(s);
    return div.innerHTML;
  }

  // Case-insensitive substring highlighter. Wraps each match of `query` in
  // `text` with <mark>, escaping everything else so it's safe to set as innerHTML.
  function highlight(text, query) {
    if (!query) return escapeHTML(text);
    var lowerText = text.toLowerCase();
    var lowerQuery = query.toLowerCase();
    var out = "";
    var start = 0;
    var i;
    while ((i = lowerText.indexOf(lowerQuery, start)) !== -1) {
      out += escapeHTML(text.slice(start, i));
      out += "<mark>" + escapeHTML(text.slice(i, i + lowerQuery.length)) + "</mark>";
      start = i + lowerQuery.length;
    }
    out += escapeHTML(text.slice(start));
    return out;
  }

  function getArticles() {
    var normalView = document.getElementById("normal-view");
    if (!normalView) return [];
    var els = normalView.querySelectorAll(".article");
    return Array.prototype.map.call(els, function (el) {
      var titleEl = el.querySelector(".search-title");
      var summaryEl = el.querySelector(".search-summary");
      return {
        el: el,
        titleEl: titleEl,
        summaryEl: summaryEl,
        titleText: titleEl ? titleEl.textContent : "",
        summaryText: summaryEl ? summaryEl.textContent : "",
      };
    });
  }

  function ensureNoResultsEl() {
    var el = document.getElementById("search-no-results");
    if (el) return el;
    el = document.createElement("div");
    el.id = "search-no-results";
    el.className = "no-articles";
    el.hidden = true;
    el.innerHTML = "<p>No articles on this page match your search.</p>";
    var articlesContainer = document.querySelector("#normal-view .articles");
    if (articlesContainer && articlesContainer.parentNode) {
      articlesContainer.parentNode.insertBefore(el, articlesContainer.nextSibling);
    }
    return el;
  }

  document.addEventListener("DOMContentLoaded", function () {
    var input = document.getElementById("article-search");
    if (!input) return;

    var articles = getArticles();
    var countEl = document.getElementById("search-count");
    var noResultsEl = ensureNoResultsEl();
    var paginationEl = document.querySelector("#normal-view .pagination");

    function applySearch() {
      var query = input.value.trim();

      if (!query) {
        articles.forEach(function (a) {
          a.el.hidden = false;
          if (a.titleEl) a.titleEl.innerHTML = escapeHTML(a.titleText);
          if (a.summaryEl) a.summaryEl.innerHTML = escapeHTML(a.summaryText);
        });
        if (countEl) countEl.hidden = true;
        noResultsEl.hidden = true;
        if (paginationEl) paginationEl.classList.remove("search-hidden");
        return;
      }

      var shown = 0;
      articles.forEach(function (a) {
        var matches =
          a.titleText.toLowerCase().indexOf(query.toLowerCase()) !== -1 ||
          a.summaryText.toLowerCase().indexOf(query.toLowerCase()) !== -1;
        a.el.hidden = !matches;
        if (matches) {
          shown++;
          if (a.titleEl) a.titleEl.innerHTML = highlight(a.titleText, query);
          if (a.summaryEl) a.summaryEl.innerHTML = highlight(a.summaryText, query);
        }
      });

      if (countEl) {
        countEl.hidden = false;
        countEl.textContent = shown + " of " + articles.length + " on this page";
      }
      noResultsEl.hidden = shown !== 0;
      if (paginationEl) paginationEl.classList.add("search-hidden");
    }

    input.addEventListener("input", applySearch);
  });
})();
