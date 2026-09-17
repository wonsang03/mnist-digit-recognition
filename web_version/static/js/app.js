/* Created: 2025-09-15
 * Canvas drawing plus calls to the /predict endpoint.
 */

(function () {
  "use strict";

  var canvas = document.getElementById("canvas");
  var context = canvas.getContext("2d");
  var digitEl = document.getElementById("digit");
  var confidenceEl = document.getElementById("confidence");
  var previewEl = document.getElementById("preview");
  var scoresEl = document.getElementById("scores");
  var statusEl = document.getElementById("status");
  var clearButton = document.getElementById("clear");
  var recognizeButton = document.getElementById("recognize");

  var BRUSH_WIDTH = 22;
  var isDrawing = false;
  var hasDrawing = false;

  // The canvas starts white so it matches the desktop version exactly.
  function resetCanvas() {
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
  }

  context.lineWidth = BRUSH_WIDTH;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "#000000";
  resetCanvas();

  // Translate a mouse or touch event into canvas coordinates. The canvas can be
  // scaled down by CSS on small screens, so the ratio has to be applied.
  function pointFromEvent(event) {
    var rect = canvas.getBoundingClientRect();
    var source = event.touches && event.touches.length ? event.touches[0] : event;
    return {
      x: (source.clientX - rect.left) * (canvas.width / rect.width),
      y: (source.clientY - rect.top) * (canvas.height / rect.height)
    };
  }

  function startStroke(event) {
    event.preventDefault();
    isDrawing = true;
    hasDrawing = true;

    var point = pointFromEvent(event);
    context.beginPath();
    context.moveTo(point.x, point.y);

    // A single tap should leave a dot, not nothing.
    context.lineTo(point.x + 0.01, point.y + 0.01);
    context.stroke();
  }

  function extendStroke(event) {
    if (!isDrawing) {
      return;
    }
    event.preventDefault();

    var point = pointFromEvent(event);
    context.lineTo(point.x, point.y);
    context.stroke();
  }

  function endStroke() {
    isDrawing = false;
  }

  canvas.addEventListener("mousedown", startStroke);
  canvas.addEventListener("mousemove", extendStroke);
  window.addEventListener("mouseup", endStroke);
  canvas.addEventListener("mouseleave", endStroke);

  canvas.addEventListener("touchstart", startStroke, { passive: false });
  canvas.addEventListener("touchmove", extendStroke, { passive: false });
  canvas.addEventListener("touchend", endStroke);

  function setStatus(message, isError) {
    statusEl.textContent = message || "";
    statusEl.classList.toggle("is-error", Boolean(isError));
  }

  function clearAll() {
    resetCanvas();
    hasDrawing = false;

    digitEl.textContent = "-";
    confidenceEl.textContent = "Confidence: -";
    previewEl.classList.remove("is-visible");
    previewEl.removeAttribute("src");
    scoresEl.innerHTML = "";
    setStatus("");
  }

  function renderScores(scores, predicted) {
    scoresEl.innerHTML = "";

    scores.forEach(function (value, label) {
      var item = document.createElement("li");
      if (label === predicted) {
        item.classList.add("is-top");
      }

      var name = document.createElement("span");
      name.textContent = String(label);

      var bar = document.createElement("span");
      bar.className = "bar";
      var fill = document.createElement("span");
      fill.style.width = Math.max(value, 0.5) + "%";
      bar.appendChild(fill);

      var percent = document.createElement("span");
      percent.textContent = value.toFixed(1) + "%";

      item.appendChild(name);
      item.appendChild(bar);
      item.appendChild(percent);
      scoresEl.appendChild(item);
    });
  }

  function recognize() {
    // Enter also triggers a click on the focused Recognize button, so the
    // disabled flag is checked here too and not only on the button.
    if (recognizeButton.disabled) {
      return;
    }

    if (!hasDrawing) {
      setStatus("Draw a digit first.", true);
      return;
    }

    recognizeButton.disabled = true;
    setStatus("Recognizing...");

    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvas.toDataURL("image/png") })
    })
      .then(function (response) {
        return response.json().then(function (data) {
          if (!response.ok) {
            throw new Error(data.error || "The server returned an error.");
          }
          return data;
        });
      })
      .then(function (data) {
        digitEl.textContent = String(data.digit);
        confidenceEl.textContent = "Confidence: " + data.confidence.toFixed(1) + "%";

        previewEl.src = data.preview;
        previewEl.classList.add("is-visible");

        renderScores(data.scores, data.digit);
        setStatus("");
      })
      .catch(function (error) {
        setStatus(error.message, true);
      })
      .finally(function () {
        recognizeButton.disabled = false;
      });
  }

  clearButton.addEventListener("click", clearAll);
  recognizeButton.addEventListener("click", recognize);

  document.addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
      recognize();
    } else if (event.key === "Escape") {
      clearAll();
    }
  });
})();
