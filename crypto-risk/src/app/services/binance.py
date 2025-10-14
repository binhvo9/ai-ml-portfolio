import asyncio
import random
import httpx

BASE = "https://api.binance.com"

async def _with_retry(url: str, params: dict | None = None, tries: int = 3, timeout: int = 10):
    last = None
    for i in range(tries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            last = e
            await asyncio.sleep(0.6 * (2 ** i) + random.random() * 0.2)
    raise last

async def get_simple_prices(symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in symbols:
        j = await _with_retry(f"{BASE}/api/v3/ticker/price", params={"symbol": s})
        out[s] = float(j["price"])
    return out
