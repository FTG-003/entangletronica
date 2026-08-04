"""In-flight animation of the flying electron forming the interference figure.

Produces:
  assets/iframe_flight.gif    (animated image, referenced by the README)

Run: python scripts/make_animation.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entangletronica import potential as P
from entangletronica import electron

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
ASSETS = os.path.join(HERE, "assets")

NX, NY, DX = 140, 80, 2.0
X = np.arange(NX) * DX - 40.0
Y = np.arange(NY) * DX - 80.0
xx, yy = np.meshgrid(X, Y, indexing="ij")
K0, S = 0.2, 10.0
DT = 0.30
NT = 1300
psi0 = electron.gaussian_packet(xx, yy, k0=K0, s=S)


def young(x, y, Vg=0.0, phase_k=0.0, barrier_k=1.0):
    V = np.zeros_like(x)
    V += P.wall(x, y, xx=0.0, w=6.0, a=15.0)
    V += barrier_k * 12.0 * P.gauss(x, 60.0, 6.0) * \
         (1.0 - P.gauss(y, -12.0, 4.0)) * (1.0 - P.gauss(y, 12.0, 4.0))
    V += P.phase_shifter(x, y, x0=68.0, y0=12.0, s=6.0, a=-15.0, k=phase_k)
    V += P.wall(x, y, xx=160.0, w=6.0, a=15.0)
    return V


def main():
    os.makedirs(FIG, exist_ok=True)
    Vmev = young(xx, yy, phase_k=1.0)
    V = Vmev * P.MEV_TO_NAT
    psi = psi0.copy()

    # accumulate snapshots at chosen flight times
    snap_steps = [0, 250, 500, 750, 1000, 1300]
    frames = []
    for n in range(1, NT + 1):
        psi = electron.step(psi, V, DT, X, Y)
        if n in snap_steps:
            frames.append((n, np.abs(psi) ** 2))
            print(f"frame at step {n}")

    # export GIF via PIL (matplotlib has no gif backend here)
    from PIL import Image
    frame_pngs = []
    for n, p in frames:
        f, a = plt.subplots(figsize=(8.5, 4))
        a.imshow(p.T, extent=(X[0], X[-1], Y[0], Y[-1]), origin="lower",
                 cmap="magma", aspect="auto", vmax=1e-3)
        a.set_title(f"step {n}/1300")
        a.set_xlabel("x [nm]"); a.set_ylabel("y [nm]")
        fp = os.path.join(FIG, "_frame.png")
        f.savefig(fp, dpi=110)
        plt.close(f)
        frame_pngs.append(Image.open(fp).convert("RGB"))

    os.makedirs(ASSETS, exist_ok=True)
    gif_path = os.path.join(ASSETS, "iframe_flight.gif")
    frame_pngs[0].save(gif_path, save_all=True, append_images=frame_pngs[1:],
                       duration=500, loop=0)
    print("wrote", gif_path, os.path.getsize(gif_path), "bytes")
    os.remove(os.path.join(FIG, "_frame.png"))

    plt.close("all")


if __name__ == "__main__":
    main()