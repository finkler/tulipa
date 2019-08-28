function initApp() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/scripts/service-worker.js").then(reg => {
      console.log("Service worker registered.", reg);
    });
  }

  var acc = document.getElementsByClassName("accordion");
  var i;

  for (i = 0; i < acc.length; i++) {
    acc[i].addEventListener("click", function() {
      icon = this.getElementsByClassName("fas")[0];
      icon.classList.toggle("fa-caret-down");
      icon.classList.toggle("fa-caret-up");

      var panel = this.nextElementSibling;
      if (panel.style.maxHeight) {
        panel.style.maxHeight = null;
      } else {
        panel.style.maxHeight = panel.scrollHeight + "px";
      }
    });
  }
  acc[1].click();
}

function check(s) {
  var cb = document.querySelectorAll("input[type=checkbox]:not([disabled])");
  var i;

  for (i = 0; i < cb.length; i++) {
    cb[i].checked = s === "all" || (!cb[i].checked && s === "invert");
  }
}

function validateForm() {
  var fc = document.getElementById("data-file");
  if (fc.value === "") {
    alert("No file chosen");
    return false;
  }
  var cb = document.querySelectorAll("input[type=checkbox]:checked");
  if (cb.length === 0) {
    alert("No models selected");
    return false;
  }
}

window.onload = initApp;
