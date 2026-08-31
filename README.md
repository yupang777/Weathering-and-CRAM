Code and data for *Carbonate weathering drives the global accumulation of riverine recalcitrant dissolved organic carbon*.

Run all commands from the repository root.

```bash
pip install -r requirements.txt
```

| Figure | Command |
|--------|---------|
| Fig. 1 | `python fig1/plot_map.py` · `plot_bc.py` · `plot_def.py` |
| Fig. 2 | `python fig2/plot_a.py` · `plot_b.py` · `pre_model_c.py` · `plot_c.py` |
| Fig. 3 | `python fig3/plot_fig3.py` |
| Fig. 4 | `python fig4/plot_ab.py` · `plot_cd.py` |
| Fig. 5 | `python fig5_different_flux/flux_global.py` · `flux_weathering.py` · `flux_continued.py` |

Site chemistry is in `global_pearl_puding data.xlsx` (Fig. 1 and Fig. 3a–b). Each other figure has its own table next to the scripts (`fig2.xlsx`, `fig3_data.xlsx`, `fig4.xlsx`, `puding.xlsx`).

Figures are written to `figures/`. GIS layers and saturation rasters are not included; Fig. 4–5 plots use the precomputed sheets in `fig4.xlsx`.
