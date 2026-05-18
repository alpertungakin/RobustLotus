"""
CityJSON LOD2 Building Exporter
--------------------------------
Exports each LOD2 building in a CityJSON file as a separate .obj file,
with a matching .txt world-info file containing BBox, EPSG, and transform.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox

try:
    from cjio import cityjson
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cjio"])
    from cjio import cityjson


# ─────────────────────────────────────────────────────────────
#  Core export logic
# ─────────────────────────────────────────────────────────────

def extract_epsg(cm):
    """
    Parse the EPSG code from CityJSON metadata.referenceSystem.
    Handles common URI and URN formats:
      - https://www.opengis.net/def/crs/EPSG/0/7415
      - http://www.opengis.net/def/crs/EPSG/0/7415
      - urn:ogc:def:crs:EPSG::7415
      - urn:ogc:def:crs:EPSG:6.6:7415
      - EPSG:7415
    Returns the code as a string, or None if not found / unrecognised.
    """
    import re
    ref = (
        cm.j.get("metadata", {}).get("referenceSystem")
        or cm.j.get("metadata", {}).get("crs", {}).get("epsg")
    )
    if ref is None:
        return None
    ref = str(ref).strip()
    # plain integer stored directly
    if ref.isdigit():
        return ref
    # any URI/URN: grab the last numeric segment
    m = re.search(r"[:/](\d{4,6})$", ref)
    if m:
        return m.group(1)
    # "EPSG:XXXXX" case-insensitive
    m = re.match(r"epsg:(\d+)", ref, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def get_lod2_building_ids(cm):
    """Return IDs of CityObjects that have at least one LOD 2 geometry."""
    ids = []
    for obj_id, city_obj in cm.j.get("CityObjects", {}).items():
        # Accept building types only (Building, BuildingPart, etc.)
        obj_type = city_obj.get("type", "")
        if "Building" not in obj_type:
            continue
        for geom in city_obj.get("geometry", []):
            lod = str(geom.get("lod", "")).strip()
            if lod in ("2", "2.0", "2.2", "2.3"):
                ids.append(obj_id)
                break
    return ids


def export_one_building(cm, obj_id, output_dir, epsg, log):
    """
    Export a single building to <obj_id>.obj + <obj_id>.txt.
    Returns True on success.
    """
    s = cm.transform["scale"]
    t = cm.transform["translate"]

    try:
        # ── Subset & export ────────────────────────────────────
        subset = cm.get_subset_ids([obj_id])
        subset.remove_orphan_vertices()

        if "transform" not in subset.j:
            subset.compress(3)

        obj_data = subset.export2obj(sloppy=True)

        safe_name = "".join(
            c for c in obj_id if c.isalnum() or c in ("-", "_")
        ).strip()
        obj_path = os.path.join(output_dir, f"{safe_name}.obj")

        with open(obj_path, "w") as fh:
            fh.write(obj_data.getvalue())

        # ── Post-process: apply real-world coordinates ─────────
        INF = float("inf")
        bbox_min = [INF, INF, INF]
        bbox_max = [-INF, -INF, -INF]

        fixed_lines = []
        with open(obj_path, "r") as fh:
            for line in fh:
                if line.startswith("v "):
                    parts = line.split()
                    x = float(parts[1]) * s[0] + t[0]
                    y = float(parts[2]) * s[1] + t[1]
                    z = float(parts[3]) * s[2] + t[2]

                    bbox_min[0] = min(bbox_min[0], x)
                    bbox_min[1] = min(bbox_min[1], y)
                    bbox_min[2] = min(bbox_min[2], z)
                    bbox_max[0] = max(bbox_max[0], x)
                    bbox_max[1] = max(bbox_max[1], y)
                    bbox_max[2] = max(bbox_max[2], z)

                    fixed_lines.append(f"v {x:.4f} {y:.4f} {z:.4f}\n")
                else:
                    fixed_lines.append(line)

        with open(obj_path, "w") as fh:
            fh.writelines(fixed_lines)

        # ── World / metadata text file ─────────────────────────
        txt_path = os.path.join(output_dir, f"{safe_name}.txt")
        with open(txt_path, "w") as fh:
            fh.write(f"Building ID: {obj_id}\n")
            fh.write("-" * 30 + "\n")
            fh.write(f"EPSG: {epsg}\n")
            fh.write(f"BBox Min (X, Y, Z): {bbox_min}\n")
            fh.write(f"BBox Max (X, Y, Z): {bbox_max}\n")
            transform_str = (
                f"{{'scale': {list(s)}, 'translate': {list(t)}}}"
            )
            fh.write(f"CityJSON Transformation: {transform_str}\n")

        log(f"  ✓  {safe_name}")
        return True

    except Exception as exc:
        log(f"  ✗  {obj_id}  →  {exc}")
        return False


# ─────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CityJSON LOD2 Building Exporter")
        self.resizable(True, True)
        self.minsize(640, 540)
        self._cm_cache = None
        self._build_ui()
        self._center()

    # ── layout ────────────────────────────────────────────────

    def _center(self):
        self.update_idletasks()
        w, h = 700, 580
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _build_ui(self):
        DARK   = "#1a1a2e"
        MID    = "#16213e"
        ACCENT = "#0f3460"
        BLUE   = "#4dabf7"
        GREEN  = "#69db7c"
        RED    = "#ff6b6b"
        TEXT   = "#e8eaf6"
        MUTED  = "#8892b0"

        self.configure(bg=DARK)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame",       background=DARK)
        style.configure("Card.TFrame",  background=MID,   relief="flat")
        style.configure("TLabel",       background=DARK,  foreground=TEXT,
                        font=("Consolas", 10))
        style.configure("Header.TLabel", background=DARK, foreground=BLUE,
                        font=("Consolas", 13, "bold"))
        style.configure("Sub.TLabel",   background=MID,   foreground=MUTED,
                        font=("Consolas", 9))
        style.configure("TEntry",       fieldbackground=ACCENT, foreground=TEXT,
                        insertcolor=BLUE, font=("Consolas", 10))
        style.configure("TButton",      background=ACCENT, foreground=TEXT,
                        font=("Consolas", 10, "bold"), borderwidth=0, padding=6)
        style.map("TButton",
                  background=[("active", BLUE)],
                  foreground=[("active", DARK)])
        style.configure("Run.TButton",  background=BLUE,  foreground=DARK,
                        font=("Consolas", 11, "bold"), padding=8)
        style.map("Run.TButton",
                  background=[("active", GREEN), ("disabled", MUTED)],
                  foreground=[("active", DARK),  ("disabled", DARK)])
        style.configure("TProgressbar", troughcolor=ACCENT,
                        background=BLUE, thickness=6)

        # ── Header ────────────────────────────────────────────
        hdr = ttk.Frame(self, padding=(20, 18, 20, 8))
        hdr.pack(fill="x")
        ttk.Label(hdr, text="⬡  CityJSON  LOD2  Exporter",
                  style="Header.TLabel").pack(side="left")

        # ── Card: file paths ───────────────────────────────────
        card1 = ttk.Frame(self, style="Card.TFrame", padding=14)
        card1.pack(fill="x", padx=20, pady=4)

        self._json_var  = tk.StringVar()
        self._outdir_var = tk.StringVar()

        self._path_row(card1, "CityJSON file", self._json_var,
                       self._browse_json, 0)
        self._path_row(card1, "Output folder", self._outdir_var,
                       self._browse_outdir, 1)

        # ── Card: options ──────────────────────────────────────
        card2 = ttk.Frame(self, style="Card.TFrame", padding=14)
        card2.pack(fill="x", padx=20, pady=4)

        self._epsg_detected = tk.StringVar(value="—")

        r = ttk.Frame(card2, style="Card.TFrame")
        r.pack(fill="x")

        # EPSG: read-only display, populated after file load
        ttk.Label(r, text="EPSG  (auto-detected)", style="Sub.TLabel",
                  background=MID).grid(row=0, column=0, sticky="w", padx=(0,16))
        ttk.Label(r, textvariable=self._epsg_detected,
                  background=MID, foreground=BLUE,
                  font=("Consolas", 10, "bold")).grid(
                      row=1, column=0, sticky="w", padx=(0,16))

        # ── Progress / status ──────────────────────────────────
        prog_frame = ttk.Frame(self, padding=(20, 6, 20, 0))
        prog_frame.pack(fill="x")

        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(prog_frame, textvariable=self._status_var,
                  foreground=MUTED, background=DARK,
                  font=("Consolas", 9)).pack(anchor="w")

        self._progress = ttk.Progressbar(prog_frame, mode="determinate",
                                          style="TProgressbar")
        self._progress.pack(fill="x", pady=(4, 0))

        # ── Log area ───────────────────────────────────────────
        log_frame = ttk.Frame(self, padding=(20, 8, 20, 0))
        log_frame.pack(fill="both", expand=True)

        self._log = scrolledtext.ScrolledText(
            log_frame, state="disabled", height=12, wrap="none",
            bg="#0d1117", fg=TEXT, insertbackground=BLUE,
            font=("Consolas", 9), relief="flat", bd=0,
            selectbackground=ACCENT
        )
        self._log.pack(fill="both", expand=True)

        # colour tags
        self._log.tag_config("ok",    foreground=GREEN)
        self._log.tag_config("err",   foreground=RED)
        self._log.tag_config("info",  foreground=BLUE)
        self._log.tag_config("muted", foreground=MUTED)

        # ── Buttons ────────────────────────────────────────────
        btn_frame = ttk.Frame(self, padding=(20, 10, 20, 16))
        btn_frame.pack(fill="x")

        self._run_btn = ttk.Button(
            btn_frame, text="▶  Export LOD2 Buildings",
            style="Run.TButton", command=self._start_export
        )
        self._run_btn.pack(side="left")

        ttk.Button(btn_frame, text="Clear log",
                   command=self._clear_log).pack(side="right")

    def _path_row(self, parent, label, var, cmd, row):
        f = ttk.Frame(parent, style="Card.TFrame")
        f.pack(fill="x", pady=3)
        ttk.Label(f, text=label, style="Sub.TLabel",
                  background="#16213e").pack(anchor="w")
        inner = ttk.Frame(f, style="Card.TFrame")
        inner.pack(fill="x")
        ttk.Entry(inner, textvariable=var).pack(side="left", fill="x",
                                                expand=True, padx=(0, 6))
        ttk.Button(inner, text="Browse…", command=cmd).pack(side="right")

    # ── Dialogs ────────────────────────────────────────────────

    def _browse_json(self):
        path = filedialog.askopenfilename(
            title="Select CityJSON file",
            filetypes=[("CityJSON", "*.json *.city.json"), ("All files", "*.*")]
        )
        if path:
            self._json_var.set(path)
            if not self._outdir_var.get():
                self._outdir_var.set(
                    os.path.join(os.path.dirname(path), "obj_output")
                )
            # Detect EPSG in background so UI stays responsive
            threading.Thread(target=self._detect_epsg,
                             args=(path,), daemon=True).start()

    def _detect_epsg(self, path):
        """Load file header and extract EPSG; update label on main thread."""
        try:
            cm   = cityjson.load(path)
            code = extract_epsg(cm)
            self._cm_cache = cm          # cache so export doesn't reload
            label = code if code else "not found in file"
            self.after(0, self._epsg_detected.set, label)
            self.after(0, self._status_var.set,
                       f"EPSG {label} detected — ready.")
        except Exception as exc:
            self._cm_cache = None
            self.after(0, self._epsg_detected.set, "error reading file")
            self.after(0, self._status_var.set, f"Load error: {exc}")

    def _browse_outdir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self._outdir_var.set(path)

    # ── Logging helpers ────────────────────────────────────────

    def _append_log(self, text, tag=""):
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n", tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _log_ok(self,   msg): self.after(0, self._append_log, msg, "ok")
    def _log_err(self,  msg): self.after(0, self._append_log, msg, "err")
    def _log_info(self, msg): self.after(0, self._append_log, msg, "info")

    def _log_any(self, msg):
        """Dispatcher used as callback from worker thread."""
        if msg.startswith("  ✓"):
            self._log_ok(msg)
        elif msg.startswith("  ✗"):
            self._log_err(msg)
        else:
            self._log_info(msg)

    # ── Export worker ──────────────────────────────────────────

    def _start_export(self):
        json_path  = self._json_var.get().strip()
        output_dir = self._outdir_var.get().strip()
        epsg       = self._epsg_detected.get().strip()

        if not json_path or not os.path.isfile(json_path):
            messagebox.showerror("Error", "Please select a valid CityJSON file.")
            return
        if not output_dir:
            messagebox.showerror("Error", "Please select an output folder.")
            return
        if epsg in ("—", "not found in file", "error reading file", ""):
            if not messagebox.askyesno(
                "EPSG not detected",
                "Could not read an EPSG code from the file.\n"
                "The world files will show 'unknown'.\nContinue?"
            ):
                return
            epsg = "unknown"
            return

        self._run_btn.configure(state="disabled")
        self._progress["value"] = 0
        self._status_var.set("Starting export…")

        # Use cached cm if available (file already loaded during EPSG detect)
        cm = getattr(self, "_cm_cache", None)

        thread = threading.Thread(
            target=self._run_export,
            args=(json_path, output_dir, epsg, cm),
            daemon=True
        )
        thread.start()

    def _run_export(self, json_path, output_dir, epsg, cm=None):
        try:
            if cm is None:
                self._log_info(f"Loading  →  {os.path.basename(json_path)}")
                cm = cityjson.load(json_path)
            else:
                self._log_info(f"Using cached  →  {os.path.basename(json_path)}")
            self._log_info(
                f"CityObjects total: "
                f"{len(cm.j.get('CityObjects', {}))}"
            )

            ids = get_lod2_building_ids(cm)
            if not ids:
                self.after(0, self._status_var.set,
                           "No LOD2 buildings found.")
                self._log_err("No LOD2 building objects found in this file.")
                self.after(0, self._run_btn.configure, {"state": "normal"})
                return

            self._log_info(f"LOD2 buildings found: {len(ids)}")
            self.after(0, self._status_var.set,
                       f"Exporting {len(ids)} buildings…")

            os.makedirs(output_dir, exist_ok=True)

            ok_count = 0
            for i, obj_id in enumerate(ids, 1):
                ok = export_one_building(
                    cm, obj_id, output_dir,
                    epsg,
                    self._log_any
                )
                if ok:
                    ok_count += 1
                pct = int(i / len(ids) * 100)
                self.after(0, self._progress.configure, {"value": pct})
                self.after(0, self._status_var.set,
                           f"{i}/{len(ids)}  —  {ok_count} ok")

            self._log_info(
                f"\nDone.  {ok_count}/{len(ids)} buildings exported"
                f"  →  {output_dir}"
            )
            self.after(0, self._status_var.set,
                       f"Complete — {ok_count}/{len(ids)} exported.")

        except Exception as exc:
            self._log_err(f"Fatal error: {exc}")
            self.after(0, self._status_var.set, f"Error: {exc}")

        finally:
            self.after(0, self._run_btn.configure, {"state": "normal"})


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()