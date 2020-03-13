import pandas as pd

def _filter_df(df, countries, provinces=None):

    if provinces is None:
        return df.loc[df['Country/Region'].isin(countries)]
    else:
        dat = df.loc[df['Country/Region'].isin(countries)]
        return dat.loc[dat['Province/State'].isin(provinces)]
    
def get_country_data(countries, provinces=None):
    if isinstance(countries, str):
        countries = [countries]

    if provinces is not None and isinstance(provinces, str):
        provinces = [provinces]
        
    covid_files = ['./COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv',
                   './COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv',
                   './COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv']
    dat = []
    for cf in covid_files:
        df = pd.read_csv(cf)
        dat.append(_filter_df(df, countries, provinces))
    return dat


if __name__ == '__main__':
    #test
    us_data = get_country_data('US')
    print("US data: \n", us_data)
    wy_data = get_country_data('US','Wyoming')
    print("WY data: \n", wy_data)
        
