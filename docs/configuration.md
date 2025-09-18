## Configuration & Environment

### YAML: `config/tickers.yaml`

- `demo_tickers`: sample list used in examples/tests
- `tech_tickers`, `sp500_tickers`: curated groups
- `defaults`: `start_date`, `data_dir`, `output_format`
- `settings`: `retry_attempts`, `delay_between_requests`, `batch_size`

This file is an example catalog; scripts accept CLI flags to override.

### Environment Variables

- `MARKETSTACK_API_KEY`: required for `MarketstackClient`

### Scheduling

- `crontab` runs daily at 07:05 UTC:

```cron
5 7 * * * /bin/sh -c 'python /app/scripts/fetch_options.py --tickers SPY QQQ --num-expiries 12 >> /app/logs/fetch_options.log 2>&1'
```

Use the provided `Dockerfile` to run this in a container with `supercronic`.

