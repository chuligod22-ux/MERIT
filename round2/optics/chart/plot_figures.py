"""Draw the three figures from one set of image arrays and measured scores."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_all(reference_image, images, rows, reference_mtf, dest):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    records = {r["method"]: r for r in rows}
    fig = plt.figure(figsize=(8.0, 8.4))
    grid = fig.add_gridspec(3, 10, left=.02, right=.98, bottom=.035, top=.97,
                            wspace=.35, hspace=.43)
    groups = [
        ("Source reference", "Degraded input", "NAFNet-big", "MERIT"),
        ("Wiener (oracle PSF)", "Richardson–Lucy", "Unsupervised Wiener",
         "Wiener (wrong PSF)", "Unsharp mask"),
        ("TV denoise", "NL-means", "SD-x4", "DiffBIR v2.1"),
    ]
    lo, hi = np.percentile(reference_image, (2, 98))
    for row, names in enumerate(groups):
        for i, name in enumerate(names):
            start = (5-len(names)) + 2*i
            ax = fig.add_subplot(grid[row, start:start+2])
            im = reference_image if name == "Source reference" else images[name]
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name, size=8.2, weight="bold" if name == "MERIT" else "normal", pad=4)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#767676")
                spine.set_linewidth(.6)
            if name == "Source reference":
                subtitle = f"MTF50 {reference_mtf:.3f} cy/px"
            else:
                item = records[name]
                subtitle = (f"MTF50 {item['mtf50']:.3f} ({item['reference_percent']:.0f}%)"
                            f"\nPSNR {item['psnr_db']:.1f} dB")
            ax.set_xlabel(subtitle, fontsize=8, labelpad=4)
    fig.savefig(dest/"fig5.pdf", bbox_inches="tight", pad_inches=.06)
    plt.close(fig)

    for perception, figure in ((False, 6), (True, 7)):
        scatter(rows, perception, dest/f"fig{figure}.pdf")


def scatter(rows, perception, outfile):
    fig, (ax, key) = plt.subplots(1, 2, figsize=(8.0, 4.0),
                                  gridspec_kw={"width_ratios": [2.15, 1.35]})
    fig.subplots_adjust(left=.095, right=.99, top=.95, bottom=.16, wspace=.085)
    xkey = "lpips_alex" if perception else "psnr_db"
    x = np.array([r[xkey] for r in rows])
    y = np.array([r["absolute_mtf_error"] for r in rows])
    ax.set_xlim(x.min()-.07*np.ptp(x), x.max()+.1*np.ptp(x))
    ax.set_ylim(-.002, max(y)*1.12)
    ax.set_xlabel("LPIPS to source reference" if perception else "PSNR to source reference (dB)")
    ax.set_ylabel("Absolute MTF50 error (cy/px)")
    ax.grid(color="#dde2e7", linewidth=.55)
    ax.set_axisbelow(True)
    key.axis("off")
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.text(.06, .97, "Method", weight="bold")
    key.text(.99, .97, "PSNR (dB)", ha="right", weight="bold", fontsize=8.3)
    fig.canvas.draw()
    used = []
    points = ax.transData.transform(np.column_stack([x, y]))
    offsets = [(6, 7), (7, -7), (-7, 7), (-7, -7), (0, 12), (15, 2), (-13, 1), (0, -13)]
    for i, (r, pos) in enumerate(zip(rows, points)):
        number = str(i+1)
        name = r["method"]
        color = "#204f83" if name == "NAFNet-big" else "#a13c12" if name == "MERIT" else "#505a63"
        marker = "D" if name == "NAFNet-big" else "*" if name == "MERIT" else "o"
        ax.scatter(r[xkey], r["absolute_mtf_error"], c=color, marker=marker,
                   s=36, zorder=3, edgecolors="white", linewidths=.4)
        def clearance(offset):
            candidate = pos + np.asarray(offset) * fig.dpi/72
            others = [q for j,q in enumerate(points) if j != i] + used
            return min(np.linalg.norm(candidate-q) for q in others)
        offset = next((off for off in offsets if clearance(off)>16), max(offsets,key=clearance))
        used.append(pos+np.asarray(offset)*fig.dpi/72)
        ax.annotate(number, (r[xkey], r["absolute_mtf_error"]), xytext=offset,
                    textcoords="offset points", ha="center", va="center", color=color,
                    size=8.3, weight="bold")
        yrow = .896 - i*.071
        key.text(.06, yrow, number, color=color, weight="bold", fontsize=8.3)
        key.text(.17, yrow, name, color=color, fontsize=8.3)
        key.text(.99, yrow, f"{r['psnr_db']:.2f}", ha="right", color=color, fontsize=8.3)
    fig.savefig(outfile, bbox_inches="tight", pad_inches=.07)
    plt.close(fig)
