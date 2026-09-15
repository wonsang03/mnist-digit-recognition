"""Desktop handwritten digit recognizer (Tkinter).

Created: 2025-09-15
Draw a digit from 0 to 9 with the mouse, press [Recognize], and the trained CNN
reports the predicted digit together with its confidence.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageDraw

# Allow running this file directly from the desktop_version folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.model import MODEL_PATH, load_model  # noqa: E402
from common.preprocess import predict_digit  # noqa: E402

# Canvas geometry and drawing settings.
CANVAS_SIZE = 280
BRUSH_RADIUS = 11

# Colour palette.
BG_COLOR = "#1e1e2e"
PANEL_COLOR = "#282a36"
CANVAS_BG = "#ffffff"
INK_COLOR = "#000000"
TEXT_COLOR = "#f8f8f2"
ACCENT_COLOR = "#8be9fd"
MUTED_COLOR = "#6272a4"


class DigitRecognizerApp:
    """Tkinter window holding the drawing canvas and the prediction panel."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Handwritten Digit Recognition - MNIST CNN")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        self.model = self._load_model_or_exit()

        # Off-screen image that mirrors every stroke drawn on the canvas.
        # Reading pixels back from the canvas widget is not portable, so each
        # stroke is recorded twice: once for the user, once for the model.
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.last_point: tuple[int, int] | None = None
        self.has_drawing = False

        self._build_widgets()

    # ------------------------------------------------------------------
    # Set-up
    # ------------------------------------------------------------------
    def _load_model_or_exit(self):
        """Load the checkpoint, or show an error dialog and quit."""
        try:
            return load_model()
        except FileNotFoundError:
            messagebox.showerror(
                "Model not found",
                f"The trained model was not found at:\n{MODEL_PATH}\n\n"
                "Run train_model.py in the project root first, or start the app\n"
                "with run_desktop.bat, which trains the model for you.",
            )
            self.root.destroy()
            sys.exit(1)

    def _build_widgets(self) -> None:
        """Create every widget in the window."""
        title = tk.Label(
            self.root,
            text="Handwritten Digit Recognition",
            font=("Segoe UI", 18, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
        )
        title.grid(row=0, column=0, columnspan=2, pady=(18, 2))

        subtitle = tk.Label(
            self.root,
            text="Draw a digit (0-9) in the white box, then press Recognize",
            font=("Segoe UI", 10),
            bg=BG_COLOR,
            fg=MUTED_COLOR,
        )
        subtitle.grid(row=1, column=0, columnspan=2, pady=(0, 14))

        # Drawing canvas.
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=CANVAS_BG,
            highlightthickness=2,
            highlightbackground=ACCENT_COLOR,
            cursor="crosshair",
        )
        self.canvas.grid(row=2, column=0, padx=(22, 11), pady=6)

        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Result panel on the right.
        panel = tk.Frame(self.root, bg=PANEL_COLOR, width=210, height=CANVAS_SIZE)
        panel.grid(row=2, column=1, padx=(11, 22), pady=6, sticky="nsew")
        panel.grid_propagate(False)

        tk.Label(
            panel,
            text="PREDICTION",
            font=("Segoe UI", 9, "bold"),
            bg=PANEL_COLOR,
            fg=MUTED_COLOR,
        ).pack(pady=(20, 4))

        self.result_label = tk.Label(
            panel,
            text="-",
            font=("Segoe UI", 76, "bold"),
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        )
        self.result_label.pack()

        self.confidence_label = tk.Label(
            panel,
            text="Confidence: -",
            font=("Segoe UI", 11),
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
        )
        self.confidence_label.pack(pady=(2, 12))

        tk.Label(
            panel,
            text="TOP 3",
            font=("Segoe UI", 9, "bold"),
            bg=PANEL_COLOR,
            fg=MUTED_COLOR,
        ).pack()

        self.top3_label = tk.Label(
            panel,
            text="-",
            font=("Consolas", 11),
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            justify="left",
        )
        self.top3_label.pack(pady=(4, 0))

        # Buttons.
        button_row = tk.Frame(self.root, bg=BG_COLOR)
        button_row.grid(row=3, column=0, columnspan=2, pady=(12, 20))

        tk.Button(
            button_row,
            text="Clear",
            font=("Segoe UI", 12, "bold"),
            width=12,
            bg="#ff5555",
            fg="white",
            activebackground="#ff7777",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            command=self.clear_canvas,
        ).pack(side="left", padx=8)

        tk.Button(
            button_row,
            text="Recognize",
            font=("Segoe UI", 12, "bold"),
            width=12,
            bg="#50fa7b",
            fg="#1e1e2e",
            activebackground="#69ff94",
            activeforeground="#1e1e2e",
            relief="flat",
            cursor="hand2",
            command=self.recognize,
        ).pack(side="left", padx=8)

        # Keyboard shortcuts: Enter recognizes, Escape clears.
        self.root.bind("<Return>", lambda event: self.recognize())
        self.root.bind("<Escape>", lambda event: self.clear_canvas())

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _on_press(self, event: tk.Event) -> None:
        self.last_point = (event.x, event.y)
        self._paint(event.x, event.y, event.x, event.y)

    def _on_drag(self, event: tk.Event) -> None:
        if self.last_point is None:
            self.last_point = (event.x, event.y)
        start_x, start_y = self.last_point
        self._paint(start_x, start_y, event.x, event.y)
        self.last_point = (event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        self.last_point = None

    def _paint(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Draw one stroke segment on both the canvas and the shadow image."""
        self.canvas.create_line(
            x1,
            y1,
            x2,
            y2,
            fill=INK_COLOR,
            width=BRUSH_RADIUS * 2,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.draw.line([(x1, y1), (x2, y2)], fill=0, width=BRUSH_RADIUS * 2)

        # Round off the joints so fast strokes do not look segmented.
        self.draw.ellipse(
            [
                x2 - BRUSH_RADIUS,
                y2 - BRUSH_RADIUS,
                x2 + BRUSH_RADIUS,
                y2 + BRUSH_RADIUS,
            ],
            fill=0,
        )

        self.has_drawing = True

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def clear_canvas(self) -> None:
        """Erase the drawing and reset the result panel."""
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)
        self.has_drawing = False
        self.last_point = None

        self.result_label.config(text="-")
        self.confidence_label.config(text="Confidence: -")
        self.top3_label.config(text="-")

    def recognize(self) -> None:
        """Run the model on the current drawing and show the result."""
        if not self.has_drawing:
            messagebox.showinfo("Empty canvas", "Draw a digit first.")
            return

        digit, confidence, scores, _ = predict_digit(self.model, self.image)

        self.result_label.config(text=str(digit))
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")

        ranking = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        self.top3_label.config(
            text="\n".join(
                f"{value:>5.1f}%  ->  {label}" for label, value in ranking[:3]
            )
        )


def main() -> None:
    root = tk.Tk()
    DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
