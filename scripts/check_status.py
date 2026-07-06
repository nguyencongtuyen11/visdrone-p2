"""KIEM TRA TRANG THAI moi truong (Lightning hoac local) — tra loi "co dong bo khong?".
Chay:  python scripts/check_status.py
In ra: git commit, GPU, data, weights, RL checkpoint, phien ban lib — kem PASS/FAIL tung muc.
"""
import subprocess, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
OK, BAD, WARN = "[OK]  ", "[LOI] ", "[CHU Y]"

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT, timeout=15).stdout.strip()
    except Exception:
        return ""

print("=" * 62)
print("  TRANG THAI REPO / MOI TRUONG  —  visdrone-p2")
print("=" * 62)

# --- 1. GIT: dang o commit nao, co file sua tay chua commit khong ---
head = sh("git rev-parse --short HEAD")
msg = sh("git log -1 --format=%s")
dirty = sh("git status --porcelain")
print(f"\n[1] GIT")
print(f"{OK}commit hien tai : {head}  \"{msg[:56]}\"")
if dirty:
    print(f"{WARN}co {len(dirty.splitlines())} file sua tay chua commit (co the lech voi local):")
    for line in dirty.splitlines()[:6]: print(f"        {line}")
else:
    print(f"{OK}working tree sach — khop 100% voi commit tren")

# --- 2. GPU ---
print(f"\n[2] GPU")
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{OK}{name} · {vram:.1f} GB · torch {torch.__version__}")
    else:
        print(f"{BAD}KHONG co GPU! (torch {torch.__version__}) — gan lai GPU cho Studio roi chay lai")
except Exception as e:
    print(f"{BAD}khong import duoc torch: {e}")

# --- 3. LIBS ---
print(f"\n[3] THU VIEN")
try:
    import ultralytics
    print(f"{OK}ultralytics {ultralytics.__version__}")
except Exception as e:
    print(f"{BAD}ultralytics: {e}")

# --- 4. DATA ---
print(f"\n[4] DATA (data/raw)")
expect = {"train": 6471, "val": 548, "test": 1610}
for split, exp in expect.items():
    d = ROOT / "data" / "raw" / "images" / split
    n = len(list(d.glob("*.jpg"))) if d.exists() else 0
    tag = OK if n == exp else (BAD if n == 0 else WARN)
    print(f"{tag}{split:5s}: {n} anh (chuan: {exp})" + ("  -> chay: python scripts/download_visdrone.py" if n == 0 else ""))

# --- 5. WEIGHTS + RL CHECKPOINT ---
print(f"\n[5] WEIGHTS")
w = ROOT / "best_visdrone.pt"
if w.exists():
    print(f"{OK}best_visdrone.pt · {w.stat().st_size/1e6:.1f} MB")
else:
    print(f"{BAD}THIEU best_visdrone.pt — git pull lai")
ckpt = ROOT / "runs" / "ft_rl" / "dqn" / "best.pt"
if ckpt.exists():
    try:
        import torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ec = ck.get("env_cfg", {})
        print(f"{OK}RL checkpoint · state_dim {ck.get('state_dim')} · max_steps {ec.get('max_steps')} · max_slices {ec.get('max_slices')}")
        good = ck.get("state_dim") == 5660 and ec.get("max_steps") == 10 and ec.get("max_slices") == 14
        print((OK + "khop chuan (5660 / 10 / 14) — dung checkpoint dang dung") if good
              else (WARN + "KHAC chuan 5660/10/14 — kiem tra co pull nham checkpoint khac khong"))
    except Exception as e:
        print(f"{WARN}doc checkpoint loi: {e}")
else:
    print(f"{BAD}THIEU runs/ft_rl/dqn/best.pt — git pull lai (da force-add vao repo)")

# --- 6. SCRIPTS can co ---
print(f"\n[6] SCRIPTS")
for s in ["benchmark_oneshot.py", "benchmark_speed.py", "download_visdrone.py", "train_p2.py"]:
    p = ROOT / "scripts" / s
    print((OK if p.exists() else BAD) + s)

print("\n" + "=" * 62)
print("  Muon so voi may local: local chay cung lenh nay roi doi chieu")
print("  dong [1] GIT commit — 2 ben cung hash = dong bo code 100%.")
print("=" * 62)
