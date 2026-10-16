# Smart Factory Stochastic Simulator — v3

Streamlit discrete-event simulation of a multi-product smart-factory flow.

## Professor-feedback additions in v3
- Explicit live **WIP** display during replay, defined as admitted products minus completed products.
- Peak WIP KPI and a WIP-over-time chart.
- Editable reliability assumptions in the sidebar:
  - mean time to failure (MTTF),
  - Weibull shape parameter,
  - mean repair time,
  - repair-time coefficient of variation.
- Reliability inputs are grouped by CNC (MC1–MC4), feeder/press (MC5–MC9), and inspection (MC10).
- Existing product-route visualization and compressed event-log replay are retained.

## Model scope
Routing and known 25 s / 5 s nominal route times come from the supplied workbook. Missing cycle-time, variability, reliability, and repair parameters are synthetic engineering assumptions intended for scenario demonstration and can later be replaced with plant measurements.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
