import os 
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def check(label, condition, detail=""):
    mark = "OK " if condition else "FAIL "
    print(f"[{mark}] {label}" + (f" ({detail})" if detail else ""))
    if not condition:
        sys.exit(1)

# 1 imports
try:
    import pyspedas
    from pyspedas import tplot, options, tplot_options, get_data
    from pyspedas import tplot_names
    import matplotlib
except Exception as e:
    print(f"[FAIL] import: {e}")
    sys.exit(1)
check("imports", True, f"pyspedas {pyspedas.version()}")

# 2 env var
data_dir = os.environ.get("THM_DATA_DIR")
check("THM_DATA_DIR set", data_dir is not None, data_dir or "missing")
check("data dir exists", Path(data_dir).exists(), data_dir)

# 3 download + load
probe = "a"
trange = ["2019-05-01", "2019-05-02"]
print(f"\nLoading THEMIS-{probe.upper()} ESA L2 for {trange[0]}...")
loaded = pyspedas.projects.themis.esa(probe=probe, trange=trange, level="l2")
check("esa() returned variables", bool(loaded), f"{len(loaded)} vars")

# 4 target variable present
target = "tha_peif_en_eflux"
check(f"{target} in tplot store", target in tplot_names(quiet=True))

d = get_data(target)
check(f"{target} has data", d is not None and len(d.times) > 0,
      f"{len(d.times)} time samples" if d is not None else "no data")

# 5 render png
out = Path("test_plot_output.png")
if out.exists():
    out.unlink()

options(target, "spec", 1)
options(target, "ylog", 1)
options(target, "zlog", 1)
options(target, "yrange", [5, 30000])
options(target, "zrange", [1e3, 1e8])
options(target, "Colormap", "spedas")
tplot_options("title", f"TEST  THEMIS-{probe.upper()}  {target}  {trange[0]}")
tplot(target, save_png=str(out.with_suffix("")), xsize=12, ysize=3, display=False)

check("PNG written", out.exists(), str(out.resolve()))
check("PNG nonempty", out.stat().st_size > 1000, f"{out.stat().st_size} bytes")

print("\nAll checks passed.")