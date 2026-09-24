# 🎈 Blank app template

A simple Streamlit app template for you to modify!

## Finnhub GitHub Secret

To enable the limited Finnhub news, financials, and company-profile fetches in
GitHub Actions:

1. Open the repository on GitHub and go to **Settings > Secrets and variables > Actions**.
2. Select **New repository secret**.
3. Set the name to `FINNHUB_API_KEY` and paste the API key as the value.
4. Update the workflow step that runs StockOracle to expose the secret:

    ```yaml
    env:
       FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
    ```

Never commit the key or put it in `finnhub_keys.py`. The application reads the
environment variable, and the daily SMA pipeline limits news requests to the
top 50 trend-strength tickers.

To inspect fetch freshness locally, run `python print_data_health.py`.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

Prerequisite: install `uv` if you don't already have it.

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Sync the dependencies

   ```
   $ uv sync
   ```

2. Run the app

   ```
   $ uv run streamlit run streamlit_app.py
   ```
