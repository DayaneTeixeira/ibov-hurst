import requests
import numpy as np
import pandas as pd
import json
from datetime import datetime
import os

# ── Janelas idênticas à planilha ──────────────────────────────────────────────
WINDOWS = [11, 21, 63, 252]
LAMBDA_EWMA = 0.94

# ── Thresholds de regime (da planilha) ───────────────────────────────────────
VOL_THRESHOLDS = {
    "comprimida": 14.7,
    "normal":     18.2,
    "stress":     19.8,
}

# ─────────────────────────────────────────────────────────────────────────────
def fetch_brapi(token=None):
    """Puxa histórico do IBOV via brapi.dev (gratuito, sem token)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    params  = {"range": "2y", "interval": "1d", "fundamental": "false"}
    if token:
        params["token"] = token

    url = "https://brapi.dev/api/quote/%5EBVSP"
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()

    hist = data["results"][0]["historicalDataPrice"]
    df = pd.DataFrame(hist)[["date", "close"]].dropna()
    df["date"]  = pd.to_datetime(df["date"], unit="s")
    df = df.rename(columns={"close": "Close"})
    df = df.sort_values("date").set_index("date")
    return df

def fetch_yfinance_fallback():
    """Fallback: yfinance caso brapi falhe."""
    import yfinance as yf
    tk = yf.Ticker("^BVSP")
    df = tk.history(period="2y", interval="1d")[["Close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()

# ─────────────────────────────────────────────────────────────────────────────
def hurst_rs(series, min_n=5):
    """Expoente de Hurst pelo método R/S (igual à planilha)."""
    series = np.array(series, dtype=float)
    N = len(series)
    if N < min_n * 2:
        return np.nan
    ns, rs_vals = [], []
    for k in range(2, int(np.log2(N)) + 1):
        n = int(N / k)
        if n < min_n:
            continue
        rs_list = []
        for i in range(N // n):
            seg = series[i*n:(i+1)*n]
            mean = np.mean(seg)
            dev  = np.cumsum(seg - mean)
            R = dev.max() - dev.min()
            S = np.std(seg, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            ns.append(np.log(n))
            rs_vals.append(np.log(np.mean(rs_list)))
    if len(ns) < 2:
        return np.nan
    return round(float(np.polyfit(ns, rs_vals, 1)[0]), 4)

def ewma_vol(log_rets, lam=LAMBDA_EWMA):
    """Volatilidade EWMA anualizada (%) — igual ao modelo da planilha."""
    var = 0.0
    for r in log_rets:
        var = lam * var + (1 - lam) * r**2
    return round(float(np.sqrt(var * 252) * 100), 2)

def vol_regime(vol):
    if vol is None:
        return "N/A"
    if vol < VOL_THRESHOLDS["comprimida"]:
        return "Comprimida"
    if vol < VOL_THRESHOLDS["normal"]:
        return "Normal"
    if vol < VOL_THRESHOLDS["stress"]:
        return "Stress Elevado"
    return "Stress Extremo"

def classify_hurst(h):
    if h is None or np.isnan(h):
        return "N/A"
    if h > 0.55:
        return "Tendência"
    if h < 0.45:
        return "Reversão"
    return "Aleatório"

def sinal(regime_h, regime_v):
    combos = {
        ("Tendência",  "Comprimida"):     ("Tendência + Comprimida",   "Aguardar expansão de vol antes de entrar"),
        ("Tendência",  "Normal"):         ("Tendência + Normal",        "Melhor cenário para seguir tendência"),
        ("Tendência",  "Stress Elevado"): ("Tendência + Stress",        "Tendência válida, gerenciar tamanho"),
        ("Tendência",  "Stress Extremo"): ("Tendência + Extremo",       "Reduzir exposição, vol muito alta"),
        ("Reversão",   "Comprimida"):     ("Reversão + Comprimida",     "Operar contra extremos · alvo menor"),
        ("Reversão",   "Normal"):         ("Reversão + Normal",         "Operar contra extremos · alvo menor"),
        ("Reversão",   "Stress Elevado"): ("Reversão + Stress",         "Cautela · vol pode ampliar movimento"),
        ("Reversão",   "Stress Extremo"): ("Reversão + Extremo",        "Evitar · risco de gap e virada brusca"),
        ("Aleatório",  "Comprimida"):     ("Aleatório + Comprimida",    "Aguardar definição de regime"),
        ("Aleatório",  "Normal"):         ("Aleatório",                 "Sem vantagem direcional · operar menor"),
        ("Aleatório",  "Stress Elevado"): ("Aleatório + Stress",        "Sem edge · melhor ficar fora"),
        ("Aleatório",  "Stress Extremo"): ("Aleatório + Extremo",       "Sem edge · ficar fora"),
    }
    return combos.get((regime_h, regime_v), (f"{regime_h}", "—"))

def range_pts(close, vol_pct):
    """Range diário estimado em pontos."""
    if vol_pct is None:
        return None
    daily_vol = close * (vol_pct / 100) / np.sqrt(252)
    return {
        "meio":   round(daily_vol * 0.5),
        "um_sig": round(daily_vol),
        "um5":    round(daily_vol * 1.5),
    }

# ─────────────────────────────────────────────────────────────────────────────
def main():
    token = os.environ.get("BRAPI_TOKEN")  # opcional

    print("📡 Buscando dados via brapi.dev...")
    try:
        df = fetch_brapi(token)
        print(f"   ✅ brapi.dev OK — {len(df)} pregões")
    except Exception as e:
        print(f"   ⚠️  brapi falhou ({e}), tentando yfinance...")
        df = fetch_yfinance_fallback()
        print(f"   ✅ yfinance OK — {len(df)} pregões")

    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()

    min_periods = max(WINDOWS) + 10
    results = []

    for i in range(min_periods, len(df)):
        close    = round(float(df["Close"].iloc[i]), 2)
        date_str = df.index[i].strftime("%Y-%m-%d")
        log_rets = df["log_ret"].iloc[:i+1].values

        row = {"date": date_str, "close": close, "janelas": {}}

        for w in WINDOWS:
            seg   = log_rets[max(0, i-w):i]
            h     = hurst_rs(seg)
            v     = ewma_vol(seg)
            rh    = classify_hurst(h)
            rv    = vol_regime(v)
            s, sd = sinal(rh, rv)
            rng   = range_pts(close, v)

            row["janelas"][str(w)] = {
                "hurst":       h,
                "vol_ewma":    v,
                "regime_h":    rh,
                "regime_v":    rv,
                "sinal":       s,
                "sinal_desc":  sd,
                "range":       rng,
            }

        results.append(row)

    results = results[-500:]

    output = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ticker": "^BVSP",
        "name":   "IBOVESPA",
        "latest": results[-1] if results else {},
        "data":   results,
    }

    os.makedirs("docs", exist_ok=True)
    with open("docs/data.json", "w") as f:
        json.dump(output, f, separators=(",", ":"))

    l = results[-1]
    print(f"\n✅ {len(results)} pregões salvos em docs/data.json")
    print(f"   Último: {l['date']} | Fechamento: {l['close']}")
    for w in WINDOWS:
        j = l["janelas"][str(w)]
        print(f"   {w:>3}d → H={j['hurst']}  Vol={j['vol_ewma']}%  [{j['regime_h']} + {j['regime_v']}]  → {j['sinal']}")

if __name__ == "__main__":
    main()
