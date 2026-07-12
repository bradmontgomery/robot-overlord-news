(function () {
  "use strict";

  var STORAGE_KEY = "robotOverlordTheme";

  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  }

  function updateButton(btn, theme) {
    if (theme === "light") {
      btn.textContent = "☀️";
      btn.setAttribute("aria-label", "Switch to dark theme");
      btn.title = "Switch to dark theme";
    } else {
      btn.textContent = "🌙";
      btn.setAttribute("aria-label", "Switch to light theme");
      btn.title = "Switch to light theme";
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;

    updateButton(btn, currentTheme());

    btn.addEventListener("click", function () {
      var next = currentTheme() === "light" ? "dark" : "light";
      if (next === "light") {
        document.documentElement.setAttribute("data-theme", "light");
      } else {
        document.documentElement.removeAttribute("data-theme");
      }
      localStorage.setItem(STORAGE_KEY, next);
      updateButton(btn, next);
    });
  });
})();
