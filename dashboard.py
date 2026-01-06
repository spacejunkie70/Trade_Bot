import time
import json
import os
import random
import threading
import sys
import pytz
from datetime import datetime
from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- FILE CONFIG ---
FILES = {
    "stock": "aggressive_stock_portfolio.json",
    "crypto": "crypto_portfolio.json",
    "config": "bot_config.json",
    "sectors": "sector_cache.json"
}

DEFAULT_CONFIG = {
    "stock_symbols": ["TSLA", "NVDA", "VLO", "SLB", "AMD", "PLTR", "CVX", "GEV", "INTC", "PSX"],
    "crypto_symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD", "RENDER-USD", "FET-USD", "AVAX-USD"],
    "news_keywords": ["Venezuela", "Maduro", "Oil", "Chevron", "Nvidia", "Intel", "CES"],
    "stock_strategy": {"risk_per_trade": 125, "stop_loss_mult": 1.7, "sma_period": 50, "cooldown_minutes": 60},
    "crypto_strategy": {"risk_per_trade": 75, "stop_loss_mult": 4.5, "sma_period": 25}
}

# --- GLOBALS ---
NEWS_FEED = []
LAST_HEARTBEAT = "Waiting..."
MARKET_DATA_CACHE = {} 
ALERT_HISTORY = {} 
SECTOR_CACHE = {} 

# --- CALCULATOR (THE FIX FOR NET WORTH) ---
def calculate_total_net_worth():
    """
    Calculates the TRUE Net Worth:
    Cash (from both files) + The Current Value of all Stocks/Crypto you own.
    """
    total_value = 0.0
    
    # 1. Add Stock Cash & Equity
    try:
        s_port = load_json(FILES["stock"])
        total_value += s_port.get('balance', 0)
        for sym, pos in s_port.get('positions', {}).items():
            # Value = Shares * Current Price (or Entry Price if current is missing)
            price = pos.get('current_price', pos.get('entry', 0))
            total_value += (price * pos.get('shares', 0))
    except: pass

    # 2. Add Crypto Cash & Equity
    try:
        c_port = load_json(FILES["crypto"])
        total_value += c_port.get('balance', 0)
        for sym, pos in c_port.get('positions', {}).items():
            price = pos.get('current_price', pos.get('entry', 0))
            total_value += (price * pos.get('shares', 0))
    except: pass

    return total_value

# --- CORE HELPERS ---
def load_json(filename, default=None):
    if default is None:
        default = {"balance": 10000, "positions": {}, "history": [], "cooldowns": {}}
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                if not isinstance(data, dict): data = default
                return data
        except:
            return default
    return default

def save_json(filename, data):
    tmp_filename = filename + ".tmp"
    try:
        with open(tmp_filename, "w") as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno()) 
        os.replace(tmp_filename, filename)
    except Exception as e:
        print(f"❌ Save Error: {e}")

def get_config():
    cfg = load_json(FILES["config"], DEFAULT_CONFIG)
    for k, v in DEFAULT_CONFIG.items():
        if k not in cfg: cfg[k] = v
    return cfg

def get_market_status():
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    if now.weekday() > 4: return "WEEKEND", False
    t = now.time()
    if datetime.strptime("08:00", "%H:%M").time() <= t < datetime.strptime("09:30", "%H:%M").time():
        return "PRE-MARKET", True
    elif datetime.strptime("09:30", "%H:%M").time() <= t <= datetime.strptime("16:00", "%H:%M").time():
        return "MARKET OPEN", True
    else:
        return "CLOSED", False

# --- SECTOR HELPERS ---
def load_sector_cache():
    global SECTOR_CACHE
    SECTOR_CACHE = load_json(FILES["sectors"], {})

def get_sector(symbol):
    if "-" in symbol and ("USD" in symbol or "BTC" in symbol):
        return "Crypto"
    if symbol in SECTOR_CACHE:
        return SECTOR_CACHE[symbol]
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info 
        sector = info.get('sector', 'Uncategorized')
        SECTOR_CACHE[symbol] = sector
        save_json(FILES["sectors"], SECTOR_CACHE)
        return sector
    except:
        return "Uncategorized"

# --- TECHNICAL ENGINE ---
def calculate_indicators(prices, sma_period=50):
    if len(prices) < sma_period: return None, None
    try:
        sma = prices.rolling(window=sma_period).mean().iloc[-1]
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return float(sma), float(rsi.iloc[-1])
    except:
        return None, None

def process_exits(portfolio, file_name, strategy_settings):
    to_sell = []
    positions = portfolio.get('positions', {})
    for sym, pos in positions.items():
        if pos.get('manual_hold', False): continue
        current_price = pos.get('current_price', pos['entry'])
        
        # Stop Loss & Take Profit Logic
        if current_price <= pos.get('stop', 0):
            to_sell.append((sym, current_price, "LOSS (Stop Hit)"))
        elif current_price >= pos.get('target', 999999):
            to_sell.append((sym, current_price, "WIN (Target Hit)"))
            
    for sym, price, reason in to_sell:
        execute_sell(portfolio, sym, price, positions[sym]['shares'], reason, strategy_settings)
    
    if to_sell: save_json(file_name, portfolio)

def execute_sell(portfolio, sym, price, shares_to_sell, reason, strategy_snapshot=None):
    if sym not in portfolio['positions']: return
    pos = portfolio['positions'][sym]
    sell_value = shares_to_sell * price
    cost_basis = shares_to_sell * pos['entry']
    profit = sell_value - cost_basis
    
    portfolio['balance'] += sell_value
    portfolio['history'].append({
        "date": str(datetime.now()), "type": "SELL", "symbol": sym, 
        "profit": profit, "reason": reason, "price": price, "shares": shares_to_sell,
        "strategy_snapshot": strategy_snapshot
    })
    
    remaining = pos['shares'] - shares_to_sell
    if remaining < 0.0001: 
        del portfolio['positions'][sym]
        # --- FIX: RECORD THE COOLDOWN TIME ---
        # This prevents the bot from buying the stock back immediately
        if 'cooldowns' not in portfolio: portfolio['cooldowns'] = {}
        portfolio['cooldowns'][sym] = time.time()
        # -------------------------------------
    else: 
        portfolio['positions'][sym]['shares'] = remaining

def bulk_scanner(portfolio, symbols, strategy, file_name, asset_type):
    global LAST_HEARTBEAT, MARKET_DATA_CACHE
    LAST_HEARTBEAT = datetime.now().strftime("%H:%M:%S")
    if not symbols: return
    
    owned_symbols = list(portfolio.get('positions', {}).keys())
    full_list = list(set(symbols + owned_symbols))

    try:
        data = yf.download(full_list, period="5d", interval="1m", group_by='ticker', progress=False)
        if data is None or data.empty: return

        for sym in full_list:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    df = data[sym].dropna()
                else:
                    if len(full_list) == 1 and full_list[0] == sym:
                         df = data.dropna()
                    else:
                         continue 
            except Exception: 
                continue 

            if df.empty or len(df) < 2: continue
            
            current_price = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[0]) 
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            if sym in portfolio['positions']:
                portfolio['positions'][sym]['current_price'] = current_price

            sma_val, rsi_val = calculate_indicators(df['Close'], strategy['sma_period'])
            trend_str = "NEUTRAL"; status_str = "WAIT"

            if sma_val is not None:
                trend_str = "BULL [+]" if current_price > sma_val else "BEAR [-]"
                if trend_str == "BULL [+]" and 40 < rsi_val < 65: status_str = "BUY (Trend)"
                elif rsi_val < 30: status_str = "BUY (Oversold)"
                elif rsi_val > 70: status_str = "SELL (High RSI)"

            sector_name = get_sector(sym)

            MARKET_DATA_CACHE[sym] = {
                "symbol": sym, "price": current_price, "change": change_pct,
                "rsi": rsi_val if rsi_val is not None else 50, "trend": trend_str, "status": status_str,
                "sector": sector_name
            }

            # BUY LOGIC (With Cooldown Check)
            last_sold_time = portfolio.get('cooldowns', {}).get(sym, 0)
            is_cooled_down = (time.time() - last_sold_time) > (strategy.get('cooldown_minutes', 0) * 60)

            if sym not in portfolio['positions'] and portfolio['balance'] > 500 and is_cooled_down:
                if "BUY" in status_str:
                    risk_target = strategy['risk_per_trade']
                    stop_pct = (strategy['stop_loss_mult'] * 2) / 100.0
                    stop_price = current_price * (1 - stop_pct)
                    risk_per_share = current_price - stop_price
                    
                    if risk_per_share > 0:
                        ideal_cost = (risk_target / risk_per_share) * current_price
                        avail_cash = portfolio['balance'] - 100
                        final_cost = min(ideal_cost, avail_cash)
                        
                        if final_cost > 100:
                            shares = final_cost / current_price
                            target = current_price + (risk_per_share * 2.5)
                            portfolio['balance'] -= final_cost
                            portfolio['positions'][sym] = {
                                "entry": current_price, "current_price": current_price,
                                "shares": shares, "stop": stop_price, "target": target, 
                                "setup": status_str, "manual_hold": False 
                            }
                            portfolio['history'].append({
                                "date": str(datetime.now()), "type": "BUY", "symbol": sym, 
                                "price": current_price, "cost": final_cost, "strategy_snapshot": strategy
                            })
        save_json(file_name, portfolio)
    except Exception as e: print(f"❌ Scanner Error ({asset_type}): {e}")

def trading_loop():
    load_sector_cache()
    while True:
        try:
            cfg = get_config()
            s_port = load_json(FILES["stock"])
            c_port = load_json(FILES["crypto"])
            
            market_status, is_active = get_market_status()
            current_dt = datetime.now(pytz.timezone('US/Eastern'))
            
            # STOCK: TRADE 9:30 AM - 4:00 PM
            if is_active and current_dt.hour >= 9 and (current_dt.hour > 9 or current_dt.minute >= 30):
                bulk_scanner(s_port, cfg['stock_symbols'], cfg['stock_strategy'], FILES["stock"], "STOCK")
                process_exits(s_port, FILES["stock"], cfg['stock_strategy'])
            
            # CRYPTO: 24/7
            bulk_scanner(c_port, cfg['crypto_symbols'], cfg['crypto_strategy'], FILES["crypto"], "CRYPTO")
            process_exits(c_port, FILES["crypto"], cfg['crypto_strategy'])
            
            time.sleep(15)
        except Exception as e:
            print(f"Loop Error: {e}")
            time.sleep(10)
            
def news_loop():
    while True:
        try:
            current_time = time.time()
            for s in list(ALERT_HISTORY.keys()):
                if current_time - ALERT_HISTORY[s] > 3600: del ALERT_HISTORY[s]
            
            for sym, data in MARKET_DATA_CACHE.items():
                change = data.get('change', 0)
                if abs(change) >= 2.0 and sym not in ALERT_HISTORY:
                    direction = "SURGING" if change > 0 else "DROPPING"
                    arrow = "⬆️" if change > 0 else "⬇️"
                    txt = f"{arrow} VOLATILITY ALERT: {sym} is {direction} {change:+.2f}% today."
                    t_str = datetime.now().strftime("%H:%M")
                    NEWS_FEED.insert(0, {"time": t_str, "text": txt})
                    if len(NEWS_FEED) > 50: NEWS_FEED.pop()
                    ALERT_HISTORY[sym] = current_time
        except Exception as e: print(f"News Error: {e}")
        time.sleep(10)

threading.Thread(target=trading_loop, daemon=True).start()
threading.Thread(target=news_loop, daemon=True).start()

# --- API ROUTES ---
@app.route('/api/ai_command', methods=['POST'])
def ai_command():
    payload = request.json
    cfg = get_config()
    logs = []

    def log_system_event(message):
        try:
            port = load_json(FILES["stock"])
            port['history'].append({
                "date": str(datetime.now()), "type": "SYSTEM", "symbol": "CONFIG", 
                "price": 0, "profit": 0, "reason": message, "strategy_snapshot": "N/A"
            })
            save_json(FILES["stock"], port)
        except: pass

    try:
        if payload.get('reset_balance') or payload.get('reset_logs'):
            for asset in ["stock", "crypto"]:
                port = load_json(FILES[asset])
                if payload.get('reset_balance'): 
                    port['balance'] = payload.get('reset_balance')
                    port['positions'] = {}
                if payload.get('reset_logs'): port['history'] = []
                save_json(FILES[asset], port)
            msg = "SYSTEM RESET: Accounts restored."; logs.append(msg); log_system_event(msg)

        if 'add_stocks' in payload:
            for s in payload['add_stocks']:
                if s not in cfg['stock_symbols']: cfg['stock_symbols'].append(s); msg=f"Added Stock: {s}"; logs.append(msg); log_system_event(msg)
        if 'add_crypto' in payload:
            for c in payload['add_crypto']:
                ticker = c if '-' in c else f"{c}-USD"
                if ticker not in cfg['crypto_symbols']: cfg['crypto_symbols'].append(ticker); msg=f"Added Crypto: {ticker}"; logs.append(msg); log_system_event(msg)

        if 'remove_stocks' in payload:
            for s in payload['remove_stocks']:
                if s in cfg['stock_symbols']: 
                    cfg['stock_symbols'].remove(s)
                    if s in MARKET_DATA_CACHE: del MARKET_DATA_CACHE[s]
                    msg=f"Removed Stock: {s}"; logs.append(msg); log_system_event(msg)
        if 'remove_crypto' in payload:
            for c in payload['remove_crypto']:
                ticker = c if '-' in c else f"{c}-USD"
                if ticker in cfg['crypto_symbols']: 
                    cfg['crypto_symbols'].remove(ticker)
                    if ticker in MARKET_DATA_CACHE: del MARKET_DATA_CACHE[ticker]
                    msg=f"Removed Crypto: {ticker}"; logs.append(msg); log_system_event(msg)

        if 'sell_now' in payload:
            for s in payload['sell_now']:
                s_port = load_json(FILES["stock"])
                if s in s_port['positions']:
                    price = s_port['positions'][s].get('current_price', s_port['positions'][s]['entry'])
                    execute_sell(s_port, s, price, s_port['positions'][s]['shares'], "USER_COMMAND")
                    save_json(FILES["stock"], s_port)
                    msg = f"Liquidated Stock: {s}"; logs.append(msg); log_system_event(msg)
                
                c_port = load_json(FILES["crypto"])
                ticker = s if '-' in s else f"{s}-USD"
                if ticker in c_port['positions']:
                    price = c_port['positions'][ticker].get('current_price', c_port['positions'][ticker]['entry'])
                    execute_sell(c_port, ticker, price, c_port['positions'][ticker]['shares'], "USER_COMMAND")
                    save_json(FILES["crypto"], c_port)
                    msg = f"Liquidated Crypto: {ticker}"; logs.append(msg); log_system_event(msg)

        if 'buy_now' in payload:
            for s in payload['buy_now']:
                is_crypto = '-' in s and ('USD' in s or 'BTC' in s)
                ftype = "crypto" if is_crypto else "stock"
                port = load_json(FILES[ftype])
                strategy = cfg['crypto_strategy'] if is_crypto else cfg['stock_strategy']
                
                try:
                    data = yf.download(s, period="1d", progress=False)
                    if not data.empty:
                        price = float(data['Close'].iloc[-1])
                        risk_target = strategy.get('risk_per_trade', 50)
                        stop_mult = strategy.get('stop_loss_mult', 2.0)
                        stop_pct = (stop_mult * 2) / 100.0
                        stop_price = price * (1 - stop_pct)
                        risk_per_share = price - stop_price
                        
                        if risk_per_share > 0:
                            ideal_cost = (risk_target / risk_per_share) * price
                            final_cost = min(ideal_cost, port['balance'], 2000)
                            
                            if final_cost > 10: 
                                shares = final_cost / price
                                target = price + (risk_per_share * 2.5)
                                
                                port['balance'] -= final_cost
                                port['positions'][s] = {
                                    "entry": price, "current_price": price, "shares": shares,
                                    "stop": stop_price, "target": target, "setup": "USER_FORCE",
                                    "manual_hold": True 
                                }
                                port['history'].append({
                                    "date": str(datetime.now()), "type": "BUY", "symbol": s, 
                                    "price": price, "cost": final_cost, "reason": "USER_FORCE"
                                })
                                save_json(FILES[ftype], port)
                                msg = f"Force Buy: {s} @ ${price:.2f} (Inv: ${final_cost:.2f})"; logs.append(msg); log_system_event(msg)
                except Exception as e:
                    logs.append(f"Failed to buy {s}: {e}")

        if 'set_targets' in payload:
            for sym, target_price in payload['set_targets'].items():
                s_port = load_json(FILES["stock"])
                if sym in s_port['positions']:
                    s_port['positions'][sym]['target'] = float(target_price)
                    save_json(FILES["stock"], s_port)
                    msg = f"Set Target for {sym}: ${target_price}"; logs.append(msg)
                c_port = load_json(FILES["crypto"])
                ticker = sym if '-' in sym else f"{sym}-USD"
                if ticker in c_port['positions']:
                    c_port['positions'][ticker]['target'] = float(target_price)
                    save_json(FILES["crypto"], c_port)
                    msg = f"Set Target for {ticker}: ${target_price}"; logs.append(msg)

        if 'stock_strategy' in payload: 
            cfg['stock_strategy'].update(payload['stock_strategy'])
            msg=f"Updated Stock Strategy: {payload['stock_strategy']}"; logs.append(msg); log_system_event(msg)
        if 'crypto_strategy' in payload: 
            cfg['crypto_strategy'].update(payload['crypto_strategy'])
            msg=f"Updated Crypto Strategy: {payload['crypto_strategy']}"; logs.append(msg); log_system_event(msg)

        save_json(FILES["config"], cfg)
        return jsonify({"status": "Success", "actions": logs})
    except Exception as e: return jsonify({"status": "Error", "msg": str(e)})
    
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/data')
def get_data():
    s_port = load_json(FILES["stock"])
    c_port = load_json(FILES["crypto"])
    cfg = get_config()
    st, _ = get_market_status()
    
    s_pos = s_port.get('positions', {})
    c_pos = c_port.get('positions', {})

    for sym, pos in s_pos.items(): 
        cache = MARKET_DATA_CACHE.get(sym, {})
        pos['analysis'] = cache
        pos['sector'] = cache.get('sector', 'Uncategorized')

    for sym, pos in c_pos.items(): 
        cache = MARKET_DATA_CACHE.get(sym, {})
        pos['analysis'] = cache
        pos['sector'] = 'Crypto'

    # --- FIX: Use the Central Calculator ---
    true_net_worth = calculate_total_net_worth()
    
    s_watch = []
    for s in cfg.get('stock_symbols', []):
        if s in MARKET_DATA_CACHE: s_watch.append(MARKET_DATA_CACHE[s])
            
    c_watch = []
    for c in cfg.get('crypto_symbols', []):
        if c in MARKET_DATA_CACHE: c_watch.append(MARKET_DATA_CACHE[c])

    return jsonify({
        "stock": s_port, "crypto": c_port, "config": cfg, 
        "market_status": st, "last_updated": LAST_HEARTBEAT,
        "net_worth": true_net_worth, # CORRECTED VALUE
        "cash_stock": s_port.get('balance', 10000),
        "cash_crypto": c_port.get('balance', 10000),
        "news": NEWS_FEED, 
        "watchlist_data": {"stock": s_watch, "crypto": c_watch}
    })

@app.route('/api/get_strategy_log')
def get_strategy_log():
    s_p = load_json(FILES["stock"]); c_p = load_json(FILES["crypto"])
    lines = ["TYPE,SYMBOL,ACTION,DATE,PRICE,PROFIT,REASON,STRATEGY_CONTEXT"]
    all_h = []
    for h in s_p.get('history', []): h['asset'] = 'STOCK'; all_h.append(h)
    for h in c_p.get('history', []): h['asset'] = 'CRYPTO'; all_h.append(h)
    all_h.sort(key=lambda x: x['date'])
    for h in all_h:
        ctx = str(h.get('strategy_snapshot', "N/A")).replace(",", "|")
        lines.append(f"{h['asset']},{h['symbol']},{h['type']},{h['date']},{h.get('price',0):.2f},{h.get('profit',0):.2f},{h.get('reason','N/A')},{ctx}")
    return jsonify({"log": "\n".join(lines)})

@app.route('/api/update_config', methods=['POST'])
def update_config():
    save_json(FILES["config"], request.json)
    return jsonify({"status": "ok"})

@app.route('/api/sell_manual', methods=['POST'])
def sell_manual():
    d = request.json
    fname = FILES["stock"] if d['type'] == 'stock' else FILES["crypto"]
    port = load_json(fname)
    if d['symbol'] in port['positions']:
        execute_sell(port, d['symbol'], port['positions'][d['symbol']].get('current_price', 0), float(d['qty']), "MANUAL")
        save_json(fname, port)
        return jsonify({"status": "ok"})
    return jsonify({"status": "error"})

@app.route('/api/toggle_hold', methods=['POST'])
def toggle_hold():
    d = request.json
    fname = FILES["stock"] if d['type'] == 'stock' else FILES["crypto"]
    port = load_json(fname)
    if d['symbol'] in port['positions']:
        port['positions'][d['symbol']]['manual_hold'] = not port['positions'][d['symbol']].get('manual_hold', False)
        save_json(fname, port)
        return jsonify({"status": "ok"})
    return jsonify({"status": "error"})

@app.route('/api/status', methods=['GET'])
def get_status():
    s_port = load_json(FILES["stock"])
    c_port = load_json(FILES["crypto"])
    
    # --- FIX: Use the Central Calculator ---
    true_net_worth = calculate_total_net_worth()
    total_cash = s_port.get('balance', 0) + c_port.get('balance', 0)

    return jsonify({
        "net_worth": true_net_worth,
        "cash_available": total_cash,
        "status": "Online"
    })

if __name__ == '__main__':
    if not os.path.exists(FILES["config"]): save_json(FILES["config"], DEFAULT_CONFIG)
    print("AI Command Center Started...")
    app.run(debug=True, port=5000)