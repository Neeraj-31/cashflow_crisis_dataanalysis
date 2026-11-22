"""Shared matplotlib style for framework figures (use with plt.rc_context(STYLE))."""
C = dict(navy="#1d3557", blue="#457b9d", sky="#a8dadc", coral="#e63946", orange="#f4a261", amber="#e9c46a",
         teal="#2a9d8f", slate="#6c757d", green="#52b788", purple="#7b2cbf", ink="#212529",
         nodata="#e9ecef", ok="#b7e4c7", band="#a8dadc")
EVENT = dict(color=C["coral"], ls="--", lw=1.6, zorder=5)
STYLE = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
         "grid.alpha": .25, "axes.titleweight": "bold", "axes.titlesize": 10.5, "axes.titlelocation": "left",
         "axes.edgecolor": "#adb5bd", "figure.facecolor": "white", "legend.frameon": False,
         "font.family": "DejaVu Sans", "savefig.dpi": 150, "savefig.bbox": "tight"}


def fy_label(y) -> str:
    return f"FY{str(int(y))[2:]}"
