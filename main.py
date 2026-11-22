"""SIADS 699: ETF-only WRDS collection, price factors, and simple backtests.

Run: python siads699_wrds_pipeline.py
Requires: pip install wrds pandas numpy pyarrow
Does not collect Compustat/IBES, historical ETF valuation, or fundamental factors.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import wrds

TICKERS = ['XLB','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY','XLC','SPY']
SECTORS = [x for x in TICKERS if x != 'SPY']
PERMNO = {'SPY':84398,'XLB':86449,'XLC':17940,'XLE':86454,'XLF':86455,
          'XLI':86456,'XLK':86457,'XLP':86452,'XLRE':15732,'XLU':86458,
          'XLV':86451,'XLY':86453}
START = '2018-01-01'
END = '2026-11-21'  # Query is inclusive; WRDS will return available observations only.
FEE_BPS = 10.0  # Applied to one-way weight turnover at each rebalance
OUT = Path('data')
RAW, PROCESSED, RESULTS = (OUT/'raw', OUT/'processed', OUT/'results')
for p in (RAW,PROCESSED,RESULTS): p.mkdir(parents=True,exist_ok=True)

def write(df, name, folder=PROCESSED):
    df.to_csv(folder/name)
    print('Saved', folder/name, 'shape:', df.shape)

def table_columns(db,lib,table):
    d = db.describe_table(lib,table)
    return set(d['name'].str.lower())

def choose_table(db, libraries, candidates):
    for lib,table in candidates:
        if lib not in libraries: continue
        try:
            if table in db.list_tables(library=lib):
                return lib,table,table_columns(db,lib,table)
        except Exception as e:
            print('Skipping',lib,table, str(e)[:150])
    return None

def get_prices(db,libraries,frequency):
    if frequency == 'm':
        possibilities = [('crsp','msf_v2'),('crsp_m_stock','msf_v2'),('crsp','msf'),('crspq','msf')]
        new = {'date':['mthcaldt','date'],'ret':['mthret','ret'],'price':['mthprc','prc'],
               'volume':['mthvol','vol'],'retx':['mthretx','retx'],'cap':['mthcap']}
    else:
        possibilities = [('crsp','dsf_v2'),('crsp_d_stock','dsf_v2'),('crsp','dsf'),('crspq','dsf')]
        new = {'date':['dlycaldt','date'],'ret':['dlyret','ret'],'price':['dlyprc','prc'],
               'volume':['dlyvol','vol'],'retx':['dlyretx','retx'],'cap':['dlycap']}
    found = choose_table(db,libraries,possibilities)
    if not found:
        print('No accessible CRSP',frequency,'security-return table found')
        return None
    lib,tab,cols = found
    mapping = {key:next((c for c in opts if c in cols),None) for key,opts in new.items()}
    if not {'permno',mapping['date'],mapping['ret']}.issubset(cols):
        raise RuntimeError(f'Missing required CRSP columns in {lib}.{tab}: {mapping}')
    selected = ['permno'] + [f'{col} AS {key}' for key,col in mapping.items() if col]
    print('Downloading',frequency,'from',f'{lib}.{tab}',mapping)
    sql = f"SELECT {', '.join(selected)} FROM {lib}.{tab} WHERE permno IN ({','.join(map(str,PERMNO.values()))}) AND {mapping['date']} >= %(start)s AND {mapping['date']} <= %(end)s"
    df = db.raw_sql(sql,params={'start':START,'end':END},date_cols=['date'])
    if df.empty: raise RuntimeError(f'CRSP {frequency} returned no data')
    rev = {v:k for k,v in PERMNO.items()}
    df['ticker'] = df['permno'].astype(int).map(rev)
    df['date'] = pd.to_datetime(df['date'])
    for c in ['ret','retx','price','volume','cap']:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce')
    if frequency=='m': df['date']=df['date'].dt.to_period('M').dt.to_timestamp('M')
    dup = df.duplicated(['date','ticker']).sum()
    if dup: raise RuntimeError(f'{dup} duplicate {frequency} observations: investigate before continuing')
    df = df.sort_values(['ticker','date'])
    df.to_csv(RAW/f'crsp_{frequency}_etfs_long.csv',index=False)
    rets = df.pivot(index='date',columns='ticker',values='ret').sort_index().reindex(columns=TICKERS)
    write(rets,f'crsp_{frequency}_returns.csv',RAW)
    for col in ['price','volume','retx','cap']:
        if col in df:
            wide = df.pivot(index='date',columns='ticker',values=col).sort_index().reindex(columns=TICKERS)
            write(wide,f'crsp_{frequency}_{col}.csv',RAW)
    return rets

def get_riskfree(db,libraries):
    # Only supported exact table/field combinations. No speculative bond yields treated as returns.
    choices = [('crsp_a_treasuries','tfz_mth_rf2'),('crsp_a_treasuries','tfz_mth_rf'),
               ('crsp_m_treasuries','tfz_mth_rf2'),('crsp_m_treasuries','tfz_mth_rf')]
    found=choose_table(db,libraries,choices)
    if not found:
        print('Risk-free table not found; risk-free-adjusted performance will use 0 and be labeled as proxy')
        return None
    lib,table,cols=found
    print('Treasury candidate',f'{lib}.{table}','columns:',sorted(cols))
    # We cannot automatically infer which fields are yields vs returns safely.
    # Export raw monthly table limited to requested dates, for inspection.
    date_col=next((x for x in ['mcaldt','caldt','date','mthcaldt'] if x in cols),None)
    if not date_col:
        print('Treasury table has no recognized date column; skipping export')
        return None
    sql=f'SELECT * FROM {lib}.{table} WHERE {date_col} >= %(start)s AND {date_col} <= %(end)s'
    try:
        df=db.raw_sql(sql,params={'start':START,'end':END})
        df.to_csv(RAW/'treasury_monthly_raw_review.csv',index=False)
        print('Treasury raw table saved for manual rate definition; not used in Sharpe calculations')
    except Exception as e:
        print('Treasury table export skipped:',str(e)[:200])
    return None

def momentum(r, months):
    return (1+r).rolling(months,min_periods=months).apply(np.prod,raw=True)-1

def factor_panel(monthly,daily):
    r=monthly[SECTORS].copy()
    bench=monthly['SPY']
    d={}
    for n in [1,3,6,9,12]: d[f'momentum_{n}m']=momentum(r,n)
    d['momentum_12m_skip_1m']=(1+r.shift(1)).rolling(11,min_periods=11).apply(np.prod,raw=True)-1
    for n in [3,6,12]:
        d[f'vol_{n}m']=r.rolling(n,min_periods=n).std()*np.sqrt(12)
        d[f'downside_dev_{n}m']=r.clip(upper=0).pow(2).rolling(n,min_periods=n).mean().pow(.5)*np.sqrt(12)
    for n in [3,6,12]: d[f'rel_strength_{n}m']=d[f'momentum_{n}m'].sub(momentum(bench.to_frame('SPY'),n)['SPY'],axis=0)
    d['beta_12m']=r.rolling(12,min_periods=12).cov(bench).div(bench.rolling(12,min_periods=12).var(),axis=0)
    nav=(1+r.fillna(0)).cumprod()
    d['drawdown_12m']=nav.div(nav.rolling(12,min_periods=12).max())-1
    if daily is not None:
        dr=daily[SECTORS]
        # 63-day daily realized vol / downside volatility; month-end observations
        d['realized_vol_63d']= (dr.rolling(63,min_periods=63).std()*np.sqrt(252)).resample('ME').last().reindex(monthly.index)
        d['downside_63d']= (dr.clip(upper=0).pow(2).rolling(63,min_periods=63).mean().pow(.5)*np.sqrt(252)).resample('ME').last().reindex(monthly.index)
    factors=pd.concat(d,axis=1,names=['factor','ticker']).stack(level='ticker',future_stack=True)
    factors.index.names=['date','ticker']
    factors=factors.reset_index().sort_values(['date','ticker'])
    factors.to_csv(PROCESSED/'factor_panel.csv',index=False)
    print('Saved factor panel',factors.shape)
    return factors

def ranks_and_backtest(monthly,factors):
    # Use a small, understandable, non-redundant multi-factor recipe.
    dates=monthly.index
    def wide(col):
        return factors.pivot(index='date',columns='ticker',values=col).reindex(index=dates,columns=SECTORS)
    mom=wide('momentum_12m_skip_1m')
    vol=wide('vol_6m')
    lowdown=wide('downside_dev_6m')
    # higher is better; percentile scores across 11 sectors on each date
    mom_score=mom.rank(axis=1,pct=True,ascending=True)
    vol_score=vol.rank(axis=1,pct=True,ascending=False)
    down_score=lowdown.rank(axis=1,pct=True,ascending=False)
    composite=(mom_score+vol_score+down_score)/3
    ranks=pd.concat({'momentum':mom_score,'low_vol':vol_score,'downside':down_score,'composite':composite},axis=1)
    write(ranks,'monthly_sector_factor_scores.csv')
    def top_three(score):
        valid=score.notna().sum(axis=1).eq(len(SECTORS))
        winner=score.rank(axis=1,ascending=False,method='first').le(3).astype(float)
        winner=winner.div(winner.sum(axis=1),axis=0).where(valid)
        return winner
    def same_weight():
        valid=monthly[SECTORS].notna().all(axis=1)
        return pd.DataFrame(1/len(SECTORS),index=dates,columns=SECTORS).where(valid,other=np.nan)
    target={'equal_weight':same_weight(),'momentum_top3':top_three(mom_score),
            'low_vol_top3':top_three(vol_score),'composite_top3':top_three(composite)}
    # Month t signals become holdings for month t+1 (no lookahead).
    sector_ret=monthly[SECTORS]
    portfolios={'SPY':monthly['SPY']}
    weights_long=[]
    turnover_all={}
    for name,w in target.items():
        held=w.shift(1)
        # Previous month end target is actually implemented for this month.
        # Simple monthly rebalance approximation; turnover uses targets, not drift-adjusted weights.
        valid=held.notna().all(axis=1)&sector_ret.notna().all(axis=1)
        gross=(held*sector_ret).sum(axis=1,min_count=len(SECTORS)).where(valid)
        turn=w.diff().abs().sum(axis=1,min_count=len(SECTORS))/2
        # Costs paid when newly rebalanced holdings enter; use last month target turnover.
        turn_used=turn.shift(1).where(valid)
        net=gross-turn_used.fillna(0)*FEE_BPS/10000
        net=net.where(valid)
        portfolios[name]=net
        turnover_all[name]=turn_used
        z=w.stack(dropna=True).rename('target_weight').reset_index()
        z.columns=['date','ticker','target_weight']
        z['strategy']=name
        weights_long.append(z)
    pr=pd.DataFrame(portfolios).dropna(how='any')
    write(pr,'portfolio_returns_monthly.csv',RESULTS)
    (1+pr).cumprod().to_csv(RESULTS/'portfolio_growth_1dollar.csv')
    pd.concat(weights_long).to_csv(RESULTS/'target_weights_monthly.csv',index=False)
    to=pd.DataFrame(turnover_all).reindex(pr.index)
    write(to,'turnover_monthly_proxy.csv',RESULTS)
    # Since monthly riskfree was not verified, Sharpe here is mean/std of total monthly returns,
    # NOT true excess-return Sharpe. This is labeled clearly.
    metrics=[]
    for col in pr:
        x=pr[col].dropna()
        wealth=(1+x).cumprod()
        years=len(x)/12
        ann=(wealth.iloc[-1]**(1/years)-1) if len(x)>0 and wealth.iloc[-1]>0 else np.nan
        vol=x.std()*np.sqrt(12)
        metrics.append({'strategy':col,'months':len(x),'annualized_return':ann,
                        'annualized_vol':vol,'return_to_vol_proxy_not_sharpe':x.mean()*12/vol if vol>0 else np.nan,
                        'max_drawdown':(wealth/wealth.cummax()-1).min(),
                        'average_one_way_turnover_proxy':to[col].mean() if col in to else np.nan})
    pd.DataFrame(metrics).to_csv(RESULTS/'performance_summary.csv',index=False)
    print('\nPERFORMANCE SUMMARY (returns are net of the stated approximate transaction costs)')
    print(pd.DataFrame(metrics).to_string(index=False))
    print('\nNOTE: These portfolios are illustrative. Monthly return data assumes month-end trade at next holding period; risk-free series unverified.')

def main():
    db=wrds.Connection()
    try:
        libs=set(db.list_libraries())
        monthly=get_prices(db,libs,'m')
        if monthly is None: raise RuntimeError('Monthly data is required')
        daily=get_prices(db,libs,'d')
        get_riskfree(db,libs)
    finally:
        db.close()
    print('\nMonthly data coverage:\n',monthly.count().to_string())
    print('Actual monthly observations:',str(monthly.index.min().date()),'through',str(monthly.index.max().date()))
    if monthly.index.max() < pd.Timestamp('2026-06-01'):
        print('WARNING: WRDS monthly table appears to be stale; check latest release and table before claiming 2026 coverage.')
    if any(monthly[t].count()<70 for t in TICKERS):
        print('WARNING: Unexpected coverage; review before interpreting results')
    factors=factor_panel(monthly,daily)
    ranks_and_backtest(monthly,factors)
    manifest={'source':'WRDS CRSP CIZ or legacy, verified at runtime','start':START,'end':END,
              'permno':PERMNO,'fee_bps_one_way':FEE_BPS,
              'factor_groups':['momentum','relative_strength','volatility','downside','drawdown','beta'],
              'not_included':['valuation','earnings revisions','accounting quality','verified riskfree Sharpe'],
              'limitations':['monthly holdings approximation','target-weight turnover proxy','no point-in-time fundamental data']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nFinished. Outputs in',OUT.resolve())

if __name__=='__main__': main()
