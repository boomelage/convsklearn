import pandas as pd

def relativize(data):
	pricename = [f for f in data.columns if f.find('_price')!=-1 and f.find('spot_')==-1 and f.find('strike_')==-1][0]
	relative_pricename = 'relative_'+pricename
	data = data[data[pricename]<=data['spot_price']]

	data_strikes = data['strike_price']
	data['relative_spot'] = data['spot_price']/data_strikes
	data[relative_pricename] = data[pricename]/data_strikes
	try:
	    data['relative_barrier'] = data['barrier']/data_strikes
	    data['relative_rebate'] = data['rebate']/data_strikes
	except Exception:
	    pass

	data['calculation_date'] = pd.to_datetime(data['calculation_date'],format='mixed')
	data['date'] = pd.to_datetime(data['date'],format='mixed')
	return data