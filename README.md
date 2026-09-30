# StockOracle

StockOracle is a Streamlit dashboard for stock research, signals, and portfolio
monitoring. The dashboard is read-only with respect to brokerage accounts; it
does not place trades.

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

## Run locally

Install the dependencies listed in `requirements.txt`, then launch Streamlit:

```sh
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The `streamlit_app.py` entry point delegates to `dashboard.py`. API credentials
are optional; configure them in the environment or a local `.env` file. Never
commit API keys.
