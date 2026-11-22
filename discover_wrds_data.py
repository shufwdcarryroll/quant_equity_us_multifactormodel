"""SIADS 699 - WRDS Step 2: locate value & EPS revision sources.
Run from project root after the ETF pipeline. Safe metadata-only discovery:
no massive downloads, no company-level joins, no passwords printed.
"""
from pathlib import Path
import json
import wrds

ROOT=Path('data/diagnostics')
ROOT.mkdir(parents=True,exist_ok=True)
OUT=ROOT/'wrds_fundamentals_discovery.txt'
TARGET_LIB_KEYS=('finratio','ibes','comp','spdji','gics','factset')
TABLE_KEYS=('ratio','sector','industry','gics','statsum','summary','identifier','id','link','annual','quarter')
MAX_COLS=75
lines=[]
def log(message=''):
    msg=str(message)
    print(msg)
    lines.append(msg)

def columns(db,lib,table):
    desc=db.describe_table(lib,table)
    return list(desc['name'].astype(str))

log('SIADS 699 | WRDS value/revisions source check')
log('No credentials or raw company records will be saved.')
db=wrds.Connection()
try:
    libraries=sorted(db.list_libraries())
    candidates=[lib for lib in libraries if any(x in lib.lower() for x in TARGET_LIB_KEYS)]
    log('\nAVAILABLE CANDIDATE LIBRARIES:')
    log(', '.join(candidates) if candidates else '(none)')
    for lib in candidates:
        try:
            tables=sorted(db.list_tables(library=lib))
        except Exception as exc:
            log(f'\n{lib}: table listing unavailable ({type(exc).__name__})')
            continue
        matched=[t for t in tables if any(k in t.lower() for k in TABLE_KEYS)]
        log(f'\n[{lib}] {len(tables)} tables; {len(matched)} relevant matches')
        log('Matching tables: '+(', '.join(matched[:70]) if matched else '(none)'))
        # Examine only especially useful schema candidates, never large table data.
        inspect=[t for t in matched if any(k in t.lower() for k in ('industry','sector','gics','statsumu_epsus','statsum_epsus','finratio'))][:8]
        for t in inspect:
            try:
                cols=columns(db,lib,t)
                log('  SCHEMA '+lib+'.'+t+': '+', '.join(cols[:MAX_COLS]))
            except Exception as exc:
                log('  Could not inspect '+lib+'.'+t+': '+str(exc)[:160])
    log('\nINTERPRETATION:')
    log('- An available firm-level ratios table is NOT sector-level ratios.')
    log('- WRDS Financial Ratios: Industry Level may be web-only; confirm before assuming direct Python access.')
    log('- IBES summary EPS is company-level. Creating sector revisions requires constituent mapping, not just a ticker filter.')
    log('- If no preaggregated sector series exists, use price/risk factor model as primary project.')
finally:
    db.close()
    OUT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    log('\nSaved diagnostics to '+str(OUT))
