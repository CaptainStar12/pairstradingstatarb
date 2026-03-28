from ib_insync import *
from engine import NSAIEngine
import time

# --- CONFIG ---
TICKER1, TICKER2 = 'KO', 'PEP'
QTY = 100
PORT = 4002 # 4001 for Live, 4002 for Paper

def run_live():
    ib = IB()
    engine = NSAIEngine()
    
    try:
        ib.connect('127.0.0.1', PORT, clientId=1)
        c1, c2 = Stock(TICKER1, 'SMART', 'USD'), Stock(TICKER2, 'SMART', 'USD')
        ib.qualifyContracts(c1, c2)
        
        print(f"📡 NSAI Interceptor Live: Tracking {TICKER1}/{TICKER2}")
        
        while ib.waitOnUpdate():
            tickers = ib.reqTickers(c1, c2)
            p1, p2 = tickers[0].marketPrice(), tickers[1].marketPrice()
            
            if p1 > 0 and p2 > 0:
                res = engine.get_signal(p1, p2)
                print(f"Z: {res['z']:.2f} | Hurst: {res['hurst']:.2f} | Action: {res['action']}")
                
                # Execution Logic would go here (ib.placeOrder)
            
            ib.sleep(1)
            
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    run_live()
