import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
from datetime import datetime
import multiprocessing
import matplotlib.transforms as transforms

import tiny
import os

def cell_format(num):
    if -1000 < num < 1000:
        return num
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def fireplot(df, country, start_time=None, most_recent=0, save=True, show=False, fontsize=30, labelsize=30, titlesize=56, Title=None, show_sum=True, grouped_by_week=False, xlabel=None, caption=None, adjustment_factor_y=1.0, legend=False, per_capita=False, compress=True):
    # trimming the df
    if caption:
        plt.rc('text', usetex=True)
    else:
        plt.rc('text', usetex=False)

    if start_time:
        df = df.iloc[df.index.get_loc(start_time):]
    else:
        df = df.iloc[np.argmax(df.sum(axis=1) >= 3):]
        if most_recent: # so that most_recent and start_time trimming cannot be activated simultaneouly
            df = df.iloc[-most_recent:,:]
    if type(df.index[-1]) != str:
        date = df.index[-1].strftime('%Y-%m-%d')
    else:
        date = df.index[-1]
    if show_sum and not per_capita:
        df.insert(loc=0, column='SUM', value=df.sum(axis=1))

    # main part, define color boundaries and colors
    try:
        w_ratio = 1.5 if df.max().max() < 1000 else 2.5
    except TypeError: #For when we are using per capita numbers
        w_ratio = 1.5
    w_ratio = 1.5
    
    y_size = int(df.shape[0]*adjustment_factor_y)
    fig, ax = plt.subplots(figsize=(int(df.shape[1]*w_ratio), y_size)) # add some extra height if needed
    #
    #cmap = colors.ListedColormap(['#38d738','#ffff00','#ff9900','#b45f06','#ff0000','#741b47'])
    #bounds=[-100000,0.5,9.5,29.5,99.5,999.5,100000]
    if per_capita:
        #bounds=[-1,0.001,0.4,0.75,1.5,2.5,3.5,10]
        bounds=[-1, 0.001, 0.25, 0.5, 1.15, 2.25, 3.25, 10]
        bounds=[-1, 0.01, 2.5, 5, 11.5, 22.5, 32.5, 100]
    else:
        bounds=[-100000,0.5,9.5,29.5,99.5,999.5,9999.5,100000]
    cmap = colors.ListedColormap(['#2DFC00','#EBDB00','#FFB500','#B45F06', '#F40204','#BD0C21','#85183D'])
    
    if per_capita:
        norm = colors.BoundaryNorm(bounds, cmap.N, clip=True)
    else:
        norm = colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(df.fillna(0).values, cmap=cmap, norm=norm)
    #
    # tick labels
    # if grouped by week change date to week number
    if grouped_by_week:
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d').strftime('%Y-W%V')
    #
    #    
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right', va='center', rotation_mode="anchor")
    #
    # annotate, font color by cell color
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            cell = df.fillna(0).values[i, j]
            if cell:
                cell_c = 'w' if (cell >= 1000 or cell < 0) else 'k'
                cell_s = fontsize if (cell < 1000) else int(fontsize * 0.8)
                text = ax.text(j, i, cell_format(cell),
                               ha='center', va='center', color=cell_c, fontsize=cell_s)
    #
    ''' now doing this with title and suptitle
    if xlabel and caption:
        textext=r'{ \begin{center}\fontsize{'+str(int(titlesize*0.8))+r'}{3em}\selectfont{}{'+str(xlabel)+r'} \\ \fontsize{'+str(int(labelsize))+r'}{3em}\textit{\selectfont{}{'+str(caption)+r'}}\end{center}}'
        #print(textext)
        ax.set_xlabel(textext, fontsize=int(titlesize*0.8), wrap=True)
    elif caption:
        ax.set_xlabel(caption, fontstyle='italic', fontsize=labelsize, wrap=True)
    elif xlabel:
        ax.set_xlabel(xlabel, fontsize=int(0.8*titlesize), wrap=True)
    '''
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=int(0.8*titlesize), wrap=True)

    if grouped_by_week:
        ax.set_ylabel('Week Number', fontsize=int(0.8*titlesize), wrap=True)
    else:
        ax.set_ylabel('Date', fontsize=int(0.8*titlesize), wrap=True)

    if Title and caption:
        textext = r'{ \begin{center} \fontsize{'+str(int(labelsize))+r'}{3em} \textit{\selectfont{}{'+str(caption)+r'}}\end{center}'
        ax.set_title(textext, wrap=True)
        fig.suptitle(Title, fontsize=titlesize, ha='center', y=1.0, x = 0.5, transform=transforms.blended_transform_factory(ax.transAxes, fig.transFigure))
    elif caption:
        #plt.title(caption, fontsize=int(labelsize), fontstyle='italic')
        textext = r'{ \begin{center} \fontsize{'+str(int(labelsize))+r'}{3em} \textit{\selectfont{}{'+str(caption)+r'}}\end{center}'
        ax.set_title(textext, wrap=True)
        fig.suptitle('%s Fireplot (Cases)'%country, fontsize=titlesize, y=1.0, x = 0.5, transform=transforms.blended_transform_factory(ax.transAxes, fig.transFigure))
    else:
        ax.sup_title('%s Fireplot (Cases)'%country, fontsize=titlesize)

    ax.tick_params(labelsize=labelsize)


    if legend:
        cbar=fig.colorbar(im, shrink=0.45, pad=0.01)
        cbar.ax.tick_params(labelsize=labelsize)
    #
    #
    # output
    plt.tight_layout()
    if save:
        fname = ('Figures/Fire_%s_PC.png'%(country)).replace(' ','_') if per_capita  else ('Figures/Fire_%s.png'%(country)).replace(' ','_')
        plt.savefig(fname, dpi=40)
        print(f'Saving to {fname}')

        if compress:
            tiny.compress_image(uncompressed_image=fname, compressed_image=fname)
    if not show:
        plt.close()

    print(f'Finished with {country}. Date: {date}')




def Brazil():
    print('Creating Fireplot for Brazil')
    df = pd.read_csv('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv')
    df = df.pivot_table(index='date',columns='state',values='newCases').fillna(0).drop('TOTAL',axis=1).fillna(0).astype(int)
    df = df[df.sum().sort_values(ascending=False).index]

    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df_c=df.groupby(['WOY']).sum()
    #print(df_c)
    plt.rc('text', usetex=True)
    #fireplot(df_c, country='Brazil', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='States', adjustment_factor_y=3)
    fireplot(df_c, country='Brazil', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='States', adjustment_factor_y=1.1, legend=True)


def USA(partitions=None):
    print(f'Creating Fireplot for USA: Partitions={partitions}')
    df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv')
    df = df.pivot(index='date',columns='state',values='cases').iloc[32:]
    df = df[df.iloc[-1].sort_values(ascending=False).index].diff()
    df.index.rename('US', inplace=True)
    df = df.fillna(0).astype(int)
    
    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    df = df[df.iloc[-1].sort_values(ascending=False).index]
    if partitions and partitions>0:
        num_cols = len(df.columns)
        partition_size = int(num_cols/partitions)
        for s in range(partitions):
            start = partition_size*s
            end = partition_size*(s+1) + 1
            fireplot(df.iloc[:,start:end], country=f'USA Partition {s+1}', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel=f'States (Partition {s+1})', legend=True)
    else:
        #fireplot(df.iloc[:,:20], country='USA', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='States', adjustment_factor_y=16, caption_location_y=0.01)
        fireplot(df, country='USA', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='States', legend=True)


def USA_by_region():
    print('Creating Fireplot for US Regions')
    df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv')
    
    ne_data = {'New England':['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 'Connecticut']}
    ma_data = {'Middle Atlantic':['New York', 'New Jersey','Pennsylvania']}
    en_data = {'East North Central':['Ohio', 'Michigan', 'Indiana', 'Wisconsin', 'Illinois']}
    wn_data = {'West North Central':['Minnesota', 'Iowa', 'Missouri', 'North Dakota', 'South Dakota', 'Nebraska', 'Kansas']}
    sa_data = {'South Atlantic':['Delaware', 'Maryland', 'West Virginia', 'Virginia', 'North Carolina', 'South Carolina', 'Georgia', 'Florida', 'Puerto Rico', 'District of Columbia', 'Virgin Islands', 'U.S. Virgin Islands', 'Northern Mariana Islands']}
    es_data = {'East South Central':['Kentucky', 'Tennessee', 'Alabama', 'Mississippi']}
    ws_data = {'West South Central':['Arkansas', 'Louisiana', 'Oklahoma', 'Texas']}
    mo_data = {'Mountain':['Montana', 'Idaho', 'Wyoming', 'Colorado', 'New Mexico', 'Arizona', 'Utah', 'Nevada']}
    pa_data = {'Pacific':['California', 'Oregon', 'Washington', 'Alaska', 'Hawaii', 'Guam']}

    Regions = [ne_data, ma_data, en_data, wn_data, sa_data, es_data, ws_data, mo_data, pa_data]
    reg_dict = {i:k for d in Regions for k,v in d.items() for i in v}
    df['Region'] = df['state'].map(lambda s: reg_dict[s])
    df=df.groupby(by=['Region', 'date']).sum().reset_index()

    pop = pd.DataFrame(pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states_by_population', attrs={'class': 'wikitable'})[0])[['State', 'Population estimate, July 1, 2019[2]']]
    pop.rename(columns={'Population estimate, July 1, 2019[2]':'Population'}, inplace=True) 
    relevant_places = [k for k in reg_dict]
    pop = pop[pop['State'].isin(relevant_places)]
    pop['Region'] = pop['State'].map(lambda s: reg_dict[s])  
    pop= pop.groupby(by='Region').sum().reset_index()

    pop_dict_list=pop.to_dict(orient='records')
    pop_dict={}
    for d in pop_dict_list:
        state = d['Region']
        pop_dict[state]=d['Population']  

    add_pop = lambda s: pop_dict[s]
    df['Population'] = df['Region'].map(add_pop)
    df['Per Capita Cases'] = df['cases']/df['Population'] * 10000

    df = df.pivot(index='date',columns='Region',values='Per Capita Cases').iloc[32:]
    df = df[df.iloc[-1].sort_values(ascending=False).index].diff().fillna(0)
    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    df[df.columns] = df[df.columns].applymap('{:.1f}'.format).applymap(float)
    fireplot(df, country='USA Regions', Title='Fireplot US Regions (Cases per 10k persons)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Regions', legend=True, per_capita=True)



def USA_per_capita(partitions=None):
    print('Creating Fireplot for USA (Per Capita)')
    df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv')
    pop = pd.DataFrame(pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states_by_population', attrs={'class': 'wikitable'})[0])[['State', 'Population estimate, July 1, 2019[2]']]
    pop.rename(columns={'Population estimate, July 1, 2019[2]':'Population'}, inplace=True)
    
    pop_dict_list=pop.to_dict(orient='records')
    pop_dict={}
    for d in pop_dict_list:
        state = d['State']
        if state=='U.S. Virgin Islands':
            state = 'Virgin Islands'  # Add exception for virgin islands
        pop_dict[state]=d['Population']

    add_pop = lambda s: pop_dict[s]
    df['Population'] = df['state'].map(add_pop)
    df['Per Capita Cases'] = df['cases']/df['Population'] * 10000

    df = df.pivot(index='date', columns='state', values='Per Capita Cases')
    df = df[df.iloc[-1].sort_values(ascending=False).index].diff().fillna(0)
    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    df[df.columns] = df[df.columns].applymap('{:.1f}'.format).applymap(float)
    df = df[df.tail(7).sum().sort_values(ascending=False).index]
    #df = df[df.sum().sort_values(ascending=False).index]

    
    if partitions and partitions>0:
        num_cols = len(df.columns)
        partition_size = int(num_cols/partitions)
        for s in range(partitions):
            start = partition_size*s
            end = partition_size*(s+1) + 1
            fireplot(df.iloc[:,start:end], country=f'USA Partition {s+1}', Title=f'Fireplot USA (Cases per 10k persons) Partition {s+1}', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel=f'States (Partition {s+1})', legend=True, per_capita=True, start_time='2020-03-16')
    else:
        #fireplot(df.iloc[:,:20], country='USA', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='States', adjustment_factor_y=16, caption_location_y=0.01)
        fireplot(df, country='USA', Title='Fireplot USA (Cases per 10k persons)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='States', legend=True, per_capita=True, start_time='2020-03-16')


def Russia():
    #This is wierd, after the 10th of september the dataset goes from YYYY-MM-DD to YYYY-DD-MM, thanks russia
    print('Creating Fireplot for Russia')
    df = pd.read_csv('https://raw.githubusercontent.com/PhtRaveller/covid19-ru/master/data/covid_stats.csv')
    df = df.query('category=="total"').drop(['category',
                                            'Наблюдение (всего)',
                                            'Россия - сумма по регионам',
                                            'Россия',
                                            'Наблюдение'], axis=1)
    df.date = pd.to_datetime(df.date).dt.strftime('%Y-%m-%d')
    df.set_index('date', inplace=True)
    df = df.diff().fillna(0).astype(int)
    df = df[df.sum().sort_values(ascending=False).index]
    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    #fireplot(df, country='Russia')

def Spain():
    #Spain has stopped reporting since the 20th of July
    print('Creating Fireplot for Spain')
    df = pd.read_csv('https://raw.githubusercontent.com/Secuoyas-Experience/covid-19-es/master/datos-comunidades-csv/covid-19-ES-CCAA-DatosCasos.csv')
    df = df.pivot_table(index='fecha', columns='nombreCcaa', values='casosConfirmados').diff().fillna(0).astype(int)
    df = df[df.sum().sort_values(ascending=False).index]
    
    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    #print(df)
    #fireplot(df, country='Spain')

def UK():
    #Currently depreciated
    print('Creating fireplot for UK')
    df = pd.read_csv('https://raw.githubusercontent.com/tomwhite/covid-19-uk-data/master/data/covid-19-cases-uk.csv')
    df['Area_full'] = df['Country'] + '-' + df['Area']
    df = df.pivot_table(index='Date', columns='Area_full', values='TotalCases').diff().fillna(0).astype(int)
    df = df[df.sum().sort_values(ascending=False).index]

    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    #print(df)
    #fireplot(df, country='UK')


def Italy():
    print('Creating Fireplot for Italy')
    df = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
    df.data = df.data.apply(lambda i:i.split('T')[0])
    # df.data = pd.to_datetime(df.data)
    df = df.pivot_table(index='data',columns='denominazione_regione',values='nuovi_positivi')
    df = df[df.sum().sort_values(ascending=False).index]

    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    #print(df)
    #fireplot(df, country='Italy', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='States', adjustment_factor_y=27, caption_location_y=0.01)
    fireplot(df, country='Italy', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Provinces', legend=True)


def Italy_PC():
    print('Creating Fireplot for Italy (Per Capita)')
    pop = pd.DataFrame(pd.read_html('https://it.wikipedia.org/wiki/Province_d%27Italia')[0])[['Regione', 'Popolazione(ab.)', 'Sigla']]
    to_int = lambda s: int(s.replace('\xa0', '').replace(' ', ''))
    pop['Popolazione(ab.)'] = pop['Popolazione(ab.)'].map(to_int)
    trento = pop[pop['Sigla']=='TN'].drop(['Regione', 'Sigla'], axis=1)
    trento.index = ['P.A. Trento']
    bolzano = pop[pop['Sigla']=='BZ'].drop(['Regione', 'Sigla'], axis=1)
    bolzano.index=['P.A. Bolzano']
    pop = pop.groupby(by='Regione').sum().append(trento).append(bolzano).reset_index()

    pop.rename(columns={'Populazione(ab.)':'Population'}, inplace=True)
    pop_dict_list=pop.to_dict(orient='records')

    pop_dict={}
    for d in pop_dict_list:
        state = d['index']
        if state == 'Friuli-Venezia Giulia':
            state = 'Friuli Venezia Giulia'
        pop_dict[state]=d['Popolazione(ab.)']

    df = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
    df.data = df.data.apply(lambda i:i.split('T')[0])
    add_pop = lambda s: pop_dict[s]
    df['Population'] = df['denominazione_regione'].map(add_pop)
    df['Per Capita Cases'] = df['totale_casi']/df['Population'] * 10000

    df = df.pivot(index='data', columns='denominazione_regione', values='Per Capita Cases')
    df = df[df.iloc[-1].sort_values(ascending=False).index].diff().fillna(0)
    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    df[df.columns] = df[df.columns].applymap('{:.1f}'.format).applymap(float)
    #df = df[df.tail(7).sum().sort_values(ascending=False).index]
    df = df[df.sum().sort_values(ascending=False).index]
    
    fireplot(df, country='Italy', Title='Fireplot Italy (Cases per 10k persons)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Provinces', legend=True, per_capita=True)


def India():
    print('Creating Fireplot for India')
    df = pd.read_csv('https://raw.githubusercontent.com/pratik-bose/CoronaTracker/V1/CoronaData.csv')
    df = df.pivot_table(index='Date',columns='Name_1',values='TotalCases').diff().fillna(0).astype(int)
    df = df[df.sum().sort_values(ascending=False).index]

    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    #print(df)
    #fireplot(df, country='India')

def Europe():
    print('Creating Fireplot for Europe')
    EU_Countries = ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom']
    df = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv')
    df = df.pivot_table(index='Date', columns='Country', values='Confirmed')[EU_Countries].iloc[6:]
    df = df[df.sum().sort_values(ascending=False).index].diff().fillna(0.0).astype(int)

    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    #fireplot(df, country='Europe', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Countries', adjustment_factor_y=30, caption_location_y=0.01)
    fireplot(df, country='Europe', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Countries', legend=True)


def Europe_PC():
    return


def World(partitions=0):
    df = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv')
    df = df.pivot_table(index='Date', columns='Country', values='Confirmed')
    df = df[df.sum().sort_values(ascending=False).index]

    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    #print(df)   
    if partitions>0:
        num_cols = len(df.columns)
        partition_size = int(num_cols/partitions)
        for s in range(partitions):
            start = partition_size*s
            end = partition_size*(s+1) + 1
            fireplot(df.iloc[:, start:end], country=f'World Partition {s+1}', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel=f'Countries (Partition {s+1})')
    else:
        fireplot(df, country='World', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Countries', legend=True)


def Zurich():
    df = pd.read_csv('https://raw.githubusercontent.com/openZH/covid_19/master/fallzahlen_kanton_alter_geschlecht_csv/COVID19_Fallzahlen_Kanton_ZH_altersklassen_geschlecht.csv')
    df=df[['Week', 'Year', 'AgeYearCat', 'NewConfCases']]
    df['WOY'] = df['Year'].astype(str)+'-'+df['Week'].astype(str)+'-1'
    
    strptm = lambda s: datetime.strptime(s, '%Y-%W-%w')
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(by=['WOY', 'AgeYearCat']).sum().reset_index()
    df=df.pivot(index='WOY', columns='AgeYearCat', values='NewConfCases').drop(['unbekannt','100+'], axis=1).fillna(0).astype(int)
    #print(df)

    fireplot(df, country='Zürich', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Age Groups', legend=True)

def compression():
    print('Compressing Images')
    images = os.listdir('Figures')
    for img in images:
        tiny.compress_image(uncompressed_image=f'Figures/{img}', compressed_image=f"Figures/{img.replace('.png', '_compressed.png')}")


if __name__ == "__main__":
    PARALLEL = True

    if PARALLEL:
        print('Using Parallel')
        arguments = {USA: [{'partitions':3}, {'partitions':0}], USA_per_capita: [{'partitions':3}]}
        functions = [Brazil, USA, USA_per_capita, Italy, Italy_PC, Europe, Zurich]
        for f in functions:
            if f in arguments:
                for kwarg in arguments[f]:
                    p = multiprocessing.Process(target=f, kwargs=kwarg)
                    p.start()
            else:
                p = multiprocessing.Process(target=f)
                p.start()
        p.join()

    else:
        print('Not using Parallel')
        #Brazil()
        #Russia() # Data wierd
        #USA_per_capita(partitions=0)
        #USA_per_capita(partitions=3)
        #USA(partitions=0)
        USA_by_region()
        #Spain() # Data Incomplete
        #UK() # Data Incomplete
        #Italy()
        #Italy_PC()
        #India() # Data Incomplete
        #Europe()
        #World()
        #Zurich()

    #compression()
        