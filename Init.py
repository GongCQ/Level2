import gongcq.Market as Market
import gongcq.CodeSymbol as CodeSymbol
import Strategy
import datetime as dt

root = 'E:\\TEMP\\l2_sample_day'
connStr = 'reader/reader@172.16.8.20:1521/dbcenter'
sqlPrc = "SELECT TRADE_DATE, STK_UNI_CODE, CLOSE_PRICE, CLOSE_PRICE_RE, RISE_DROP_RANGE_RE / 100, OPEN_PRICE, OPEN_PRICE_RE, STK_TOT_VALUE,  TRADE_AMUT, STK_CIR_VALUE, TURNOVER_RATE " \
         "FROM UPCENTER.STK_BASIC_PRICE_MID " \
         "WHERE ISVALID = 1 AND TRADE_VOL > 0 AND TRADE_DATE = TO_DATE('{TRADE_DATE}', 'YYYY-MM-DD') "
sqlCld = "SELECT MIN(C.END_DATE) " \
         "FROM UPCENTER.PUB_EXCH_CALE C " \
         "WHERE C.IS_TRADE_DATE = 1 AND C.SEC_MAR_PAR = 1 AND " \
         "      C.END_DATE > TO_DATE('{LAST_DATE}', 'YYYY-MM-DD') " \
         "ORDER BY END_DATE"
csMap = CodeSymbol.CodeSymbol(connStr)
codeList, symbolList, nameList, mktList = CodeSymbol.GetAllCode(connStr)
mkt = Market.Market(connStr, sqlCld, dt.datetime(2017, 4, 4))
mkt.CreateDataSource(connStr, sqlPrc, codeList, csMap, 11)
mkt.CreateDataSource(root, None, codeList, csMap, 198)
mkt.CreateAccount(0, 1000000)
stg = Strategy.Strategy(codeList, csMap, 50)
mkt.AddAfterCloseReceiver(stg.da.NewDayHandler)
mkt.AddAfterCloseReceiver(stg.NewDayHandler)