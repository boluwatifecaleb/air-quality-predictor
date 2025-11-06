# air-quality-predictor
ML Early Warning System for critical $\text{CO}$ pollution ($\ge 4.7$ ppm) 24 hours ahead. Uses a Random Forest model trained on lagged sensor data (current, 1H, 24H history) to capture momentum and daily cycles. Prioritizes high Recall (catching danger) for safety. Deployed via Streamlit on GitHub
