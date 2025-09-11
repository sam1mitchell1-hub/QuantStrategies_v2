import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    from quant_strategies.data import YFinanceClient
    import matplotlib.pyplot as plt
    return YFinanceClient, plt


@app.cell
def __(datetime, timedelta):
    # %%
    # Set your ticker and date range
    TICKER = "AAPL"
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5 * 365)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    return TICKER, end_date, end_str, start_date, start_str


@app.cell
def __(start_str):
    start_str
    return


@app.cell
def __(MarketstackClient, TICKER, end_str, start_str):
    # %%
    # Create the client and fetch data
    client = MarketstackClient()
    df = client.get_daily_prices(TICKER , start_str, end_str)

    # %%
    # Display results
    print(f"Fetched {len(df)} rows for {TICKER}")
    df.head()
    return client, df


@app.cell
def __(df):
    df1 = df #.sort_values('date')
    return (df1,)


@app.cell
def __(TICKER, df1, plt):
    plt.figure(figsize=(12, 6))
    plt.plot(df1["date"], df1["open"], label=f"{TICKER} Opening Price")
    plt.xlabel("Date")
    plt.ylabel("Opening Price (USD)")
    plt.title(f"{TICKER} Opening Price Over Last 5 Years")
    plt.legend()
    plt.tight_layout()
    plt.show() 
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
