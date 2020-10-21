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
import requests, json

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


def fireplot(df, country, start_time=None, most_recent=0, save=True, show=False, fontsize=30, labelsize=35, titlesize=56, Title=None, show_sum=True, grouped_by_week=False, xlabel=None, caption=None, adjustment_factor_y=1.0, legend=False, per_capita=False, compress=True, w_rat=1.5, age_groups=False):
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
        col_name = 'All Ages' if age_groups else 'All Regions'
        df.insert(loc=0, column=col_name, value=df.sum(axis=1))

    # main part, define color boundaries and colors
    try:
        w_ratio = 1.5 if df.max().max() < 1000 else 2.5
    except TypeError: #For when we are using per capita numbers
        w_ratio = 1.5
    w_ratio = w_rat # I am aware this is stupid
    
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
        bounds=[-1,0.5,9.5,29.5,99.5,999.5,9999.5,100000]
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
            #tiny.compress_image(uncompressed_image=fname, compressed_image=fname)
            pass
    if not show:
        plt.close()

    print(f'Finished with {country}. Date: {date}')


def negative_to_zero(x):
    return max(x,0)


def Brazil():
    print('Creating Fireplot for Brazil')
    df = pd.read_csv('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv')
    df = df.pivot_table(index='date',columns='state',values='newCases').fillna(0).drop('TOTAL',axis=1).fillna(0).astype(int)
    df = df[df.sum().sort_values(ascending=False).index]
    df[df.columns] = df[df.columns].applymap(negative_to_zero)

    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df_c=df.groupby(['WOY']).sum()
    #print(df_c)
    #fireplot(df_c, country='Brazil', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='States', adjustment_factor_y=3)
    fireplot(df_c, country='Brazil', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='States', adjustment_factor_y=1.1, legend=True)


def Czechia_Age():
    print('Working on Czechia (Age)')
    df = pd.read_csv('https://onemocneni-aktualne.mzcr.cz/api/v2/covid-19/osoby.csv')
    age_groups= [(0,10), (10,15), (15,20), (20,25), (25,30), (30,35), (35,40), (40,45), (45,50), (50,55), (55,60), (60,65), (65,70), (70,75), (75,80), (80,200)]
    translate = lambda s: f'{s[0]}-{s[1]}' if s[1]<200 else f'{s[0]}+'
    def grouping(age):
        for grp in age_groups:
            if grp[0] <= age < grp[1]:
                return translate(grp)
    age_dict = {age:grouping(age) for age in range(0,150)}
    df['Group'] = df['vek'].map(age_dict)
    df['Count'] = 1
    week = lambda x : x.weekofyear
    df['WOY'] = pd.to_datetime(df['datum'], format='%Y/%m/%d').map(week).astype(int)
    df = df.groupby(by=['WOY', 'Group']).sum()[['Count']].reset_index()
    
    strptm = lambda s: datetime.strptime('2020-'+s+'-0', "%Y-%W-%w")
    df['WOY'] = df['WOY'].astype(str).map(strptm)
    df = df.pivot(index='WOY', columns='Group', values='Count').fillna(0).astype(int)
    #print(df)
    fireplot(df, country='Czechia_By_Age', Title='Czechia (Cases by Age Group)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Age Groups', legend=True, age_groups=True)

def G20(partitions=0):
    print('Working on G20 (per Capita)')
    df = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv')
    countries = ['Bulgaria', 'Romania', 'Cyprus', 'Greece', 'Denmark', 'Netherlands', 'Sweden', 'Estonia', 'Hungary', 'Iceland', 'Czechia', 'Indonesia', 'Liechtenstein', 'France', 'Lithuania', 'Mexico', 'Brazil', 'Latvia', 'China', 'Switzerland', 'Ireland', 'Austria', 'Portugal', 'Norway', 'Spain', 'Australia', 'United Kingdom', 'Slovakia', 'Italy', 'Finland', 'Argentina', 'Canada', 'South Africa', 'Saudi Arabia', 'Luxembourg', 'Russia', 'Japan', 'Slovenia', 'India', 'US', 'Belgium', 'Korea, South', 'Turkey', 'Germany', 'Poland', 'Malta']
    df = df.pivot_table(index='Date', columns='Country', values='Confirmed')[countries].diff().fillna(0).astype(int)

    pop = pd.read_csv('https://raw.githubusercontent.com/datasets/population/master/data/population.csv').pivot(index='Year', columns='Country Name', values='Value').rename(columns={'United States':'US', 'Czech Republic':'Czechia', 'Korea, Rep.':'Korea, South', 'Russian Federation':'Russia', 'Slovak Republic':'Slovakia'})
    pop = pop[countries].fillna(method='ffill').T[[2018]].reset_index()

    pop_dict_list=pop.to_dict(orient='records')
    pop_dict={}
    for d in pop_dict_list:
        state = d['Country Name']
        pop_dict[state]=d[2018]
  
    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df = df.groupby(by='WOY').sum()
    df = df.apply(lambda x: (x/pop_dict[x.name])*10000)
    df = df[df.tail(7).sum().sort_values(ascending=False).index]
    df[df.columns] = df[df.columns].applymap('{:.1f}'.format).applymap(float)

    if partitions>0:
        num_cols = len(df.columns)
        partition_size = int(num_cols/partitions)
        for s in range(partitions):
            start = partition_size*s
            end = partition_size*(s+1) + 1
            fireplot(df.iloc[:, start:end], country=f'Key Countries {s+1}', Title=f'G20 + Schengen Countries Partition {s+1} (Cases per 10k Persons)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel=f'Countries (Partition {s+1})', legend=True, per_capita=True)
    else:
        fireplot(df, country='Key Countries', Title='G20 + Schengen Countries (Cases per 10k Persons)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Countries', legend=True, per_capita=True)


def Germany():
    ger=pd.read_csv('https://www.arcgis.com/sharing/rest/content/items/f10774f1c63e40168479a1feb6c7ca74/data')
    
    # By Age
    print('Working on Germany (By Age)')
    df = ger[['Altersgruppe', 'Refdatum', 'AnzahlFall']]
    df.index = df['Refdatum'].map(lambda s: s[:-9])

    df['WOY'] = pd.to_datetime(df.index.copy(), format='%Y/%m/%d').weekofyear.astype(int)
    df = df.groupby(by=['WOY', 'Altersgruppe']).sum().reset_index()
    df['Altersgruppe'] = df['Altersgruppe'].map(lambda s: s.replace('A', ''))
    df = df.pivot(index='WOY', columns='Altersgruppe', values='AnzahlFall').sort_index()
    df.rename(columns={'00-04':'0-4', '05-14':'5-14', 'unbekannt':'unknown'}, inplace=True)
    df.index = df.index.astype(str)
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df.index = df.index.map(strptm)
    df = df.fillna(0).astype(int)
    fireplot(df, country='Germany_By_Age', Title='Germany (Cases by Age Group)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Age Groups', legend=True, w_rat=1.8, age_groups=True)

    # By Province
    print('Working on Germany (By Province)')
    df = ger[['Bundesland', 'Refdatum', 'AnzahlFall']]
    df.index = df['Refdatum'].map(lambda s: s[:-9])

    df['WOY'] = pd.to_datetime(df.index.copy(), format='%Y/%m/%d').weekofyear.astype(int)
    df = df.groupby(by=['WOY', 'Bundesland']).sum().reset_index()
    df = df.pivot(index='WOY', columns='Bundesland', values='AnzahlFall').sort_index()
    df = df[df.tail(7).sum().sort_values(ascending=False).index]
    df.index = df.index.astype(str)
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df.index = df.index.map(strptm)
    df = df.fillna(0).astype(int)
    fireplot(df, country='Germany', Title='Germany (Cases)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Provinces', legend=True,)

    # Province Per Capita
    print('Working on Germany (Per Capita)')
    pop = pd.DataFrame(pd.read_html('https://de.wikipedia.org/wiki/Land_(Deutschland)')[0])[['Land', 'Ein-wohner(Mio.)[12]']]
    pop.rename(columns={'Ein-wohner(Mio.)[12]':'Population', 'Land':'State'}, inplace=True)
    
    pop_dict_list=pop.to_dict(orient='records')
    pop_dict={}
    for d in pop_dict_list:
        state = d['State']
        pop_dict[state]=d['Population'] * 1000 # Times 1 thousand

    #print(pop_dict)
    df = ger[['Bundesland', 'Refdatum', 'AnzahlFall']]
    df['AnzahlFall'] = df['AnzahlFall'].map(negative_to_zero)
    df.index = df['Refdatum'].map(lambda s: s[:-9])

    df['WOY'] = pd.to_datetime(df.index.copy(), format='%Y/%m/%d').weekofyear.astype(int)
    df = df.groupby(by=['WOY', 'Bundesland']).sum().reset_index()
    df = df.pivot(index='WOY', columns='Bundesland', values='AnzahlFall').sort_index()
    df.index = df.index.astype(str)
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df.index = df.index.map(strptm)
    df = df.fillna(0).astype(int)
    add_pop = lambda x: (x/pop_dict[x.name])*10000
    df[df.columns] = df[df.columns].apply(add_pop)
    df[df.columns] = df[df.columns].applymap('{:.1f}'.format).applymap(float)
    df = df[df.tail(7).sum().sort_values(ascending=False).index]
    fireplot(df, country='Germany', Title='Germany (Cases per 10k persons)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Provinces', legend=True, per_capita=True)


def Holland():
    print('Working on Holland')
    df = pd.read_csv('https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv', delimiter=';').drop('Week_of_death', axis=1)
    df['Count'] = 1
    df['WOY'] = pd.to_datetime(df['Date_statistics'].copy(), format='%Y-%m-%d').map(lambda s: s.weekofyear).astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    age = df.groupby(by=['WOY', 'Agegroup']).sum()
    df = df.groupby(by=['WOY', 'Province']).sum()

    df = df.reset_index().pivot(index='WOY', columns='Province', values='Count').fillna(0).astype(int)
    df = df[df.tail(7).sum().sort_values(ascending=False).index]
    age = age.reset_index().pivot(index='WOY', columns='Agegroup', values='Count').fillna(0).astype(int)
    age = age[['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79',
       '80-89', '90+', 'Unknown']]

    print('Working on Holland (Cases)')
    fireplot(df, country='Holland', Title='Netherlands (Cases)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Provinces', legend=True)
    print('Working on Holland (Age Groups)')
    fireplot(age, country='Holland_By_Age', Title='Netherlands (Cases by Age Group)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Age Groups', legend=True, age_groups=True)

    pop = pd.DataFrame(pd.read_html('https://en.wikipedia.org/wiki/Provinces_of_the_Netherlands')[2])[['Dutch name', 'Population[A][1]']]
    pop.columns = pop.columns.get_level_values(0)
    pop_dict_list=pop.to_dict(orient='records')
    pop_dict={}
    for d in pop_dict_list:
        state = d['Dutch name'].replace(' ', '-')
        pop_dict[state]=d['Population[A][1]'] 

    divide_by_pop = lambda x : (x/pop_dict[x.name])*10000
    df[df.columns]=df[df.columns].apply(divide_by_pop)
    df[df.columns] = df[df.columns].applymap('{:.1f}'.format).applymap(float)
    df = df[df.tail(7).sum().sort_values(ascending=False).index]
    
    print('Working on Holland (Per Capita)')
    fireplot(df, country='Holland', Title='Netherland (Cases per 10k persons)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Provinces', legend=True, per_capita=True)



def USA(partitions=None):
    print(f'Creating Fireplot for USA: Partitions={partitions}')
    df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv')
    df = df.pivot(index='date',columns='state',values='cases').iloc[32:]
    df = df[df.iloc[-1].sort_values(ascending=False).index].diff()
    df.index.rename('US', inplace=True)
    df = df.fillna(0).astype(int)
    df[df.columns] = df[df.columns].applymap(negative_to_zero)
    #print(df)
    
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
    df['Per Capita Cases'] = (df['cases']/df['Population'] * 10000)
    df = df.pivot(index='date',columns='Region',values='Per Capita Cases').iloc[32:]
    df = df[df.iloc[-1].sort_values(ascending=False).index].diff().fillna(0)
    df[df.columns] = df[df.columns].applymap(negative_to_zero)
    
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
    df[df.columns] = df[df.columns].applymap(negative_to_zero)
    
    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    df = df[df.tail(7).sum().sort_values(ascending=False).index]
    df[df.columns] = df[df.columns].applymap('{:.1f}'.format).applymap(float)
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


def Florida_Age():
    print('Working on Florida (By Age)')
    df = pd.read_csv('https://www.arcgis.com/sharing/rest/content/items/4cc62b3a510949c7a8167f6baa3e069d/data')
    df['Age_group']=df['Age_group'].map(lambda s: s.replace(' years', ''))
    df['Count'] = 1
    df=df.groupby(['Case_', 'Age_group']).sum()['Count']
    df=df.reset_index().pivot(index='Case_', columns='Age_group', values='Count').fillna(0).astype(int)
    df.index = df.index.map(lambda s: s.split(' ')[0]) # Get rid of timestamp in date

    df['WOY'] = pd.to_datetime(df.index, format='%m/%d/%Y').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()[['0-4', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65-74','75-84', '85+', 'Unknown']]
    #print(df)

    fireplot(df, country='Florida', Title='Florida (Cases by Age Group)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Age Groups', legend=True, age_groups=True)



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
    df[df.columns] = df[df.columns].applymap(negative_to_zero)

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
    df[df.columns] = df[df.columns].applymap(negative_to_zero)
    
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
    if partitions>1:
        num_cols = len(df.columns)
        partition_size = int(num_cols/partitions)
        for s in range(partitions):
            start = partition_size*s
            end = partition_size*(s+1) + 1
            fireplot(df.iloc[:, start:end], country=f'World Partition {s+1}', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel=f'Countries (Partition {s+1})')
    else:
        fireplot(df, country='World', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Countries', legend=True)

def Switzerland():
    print('Creating Fireplot for Switzerland')
    df = pd.read_csv('https://raw.githubusercontent.com/openZH/covid_19/master/COVID19_Fallzahlen_CH_total_v2.csv')
    df = df.pivot(index='date', columns='abbreviation_canton_and_fl', values='ncumul_conf')
    df[df.columns] = df[df.columns].fillna(method='ffill').fillna(0).diff().fillna(0).astype(int)
    df[df.columns] = df[df.columns].applymap(negative_to_zero)

    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    df = df[df.tail(7).sum().sort_values(ascending=False).index]
    #print(df)
    
    fireplot(df, country='Switzerland', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Cantons', legend=True)

def Switzerland_PC():
    print('Creating Fireplot for Switzerland (Per Capita)')
    pop = pd.DataFrame(pd.read_html('https://en.wikipedia.org/wiki/Cantons_of_Switzerland')[1])[['Code', 'Population[Note 3]']]

    pop_dict_list=pop.to_dict(orient='records')
    pop_dict={}
    for d in pop_dict_list:
        state = d['Code'].strip()
        pop_dict[state]=int(d['Population[Note 3]'][:-4].replace(',',''))
    pop_dict['FL'] = 38749

    df = pd.read_csv('https://raw.githubusercontent.com/openZH/covid_19/master/COVID19_Fallzahlen_CH_total_v2.csv')
    df = df.pivot(index='date', columns='abbreviation_canton_and_fl', values='ncumul_conf')
    df[df.columns] = df[df.columns].fillna(method='ffill').fillna(0).diff().fillna(0).astype(int)
    def divide_by_pop(x):
        return (x/pop_dict[x.name]) * 10000
    df = df[df.columns].apply(divide_by_pop, axis=0)
    df[df.columns] = df[df.columns].applymap(negative_to_zero)

    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    df = df[df.tail(7).sum().sort_values(ascending=False).index]
    
    df[df.columns] = df[df.columns].applymap('{:.1f}'.format).applymap(float)
    #print(df)
    fireplot(df, country='Switzerland', Title='Fireplot Switzerland (Cases per 10k persons)',  grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Cantons', legend=True, per_capita=True)


def Zurich():
    print('Creating Fireplot for Zürich (Age Groups)')
    df = pd.read_csv('https://raw.githubusercontent.com/openZH/covid_19/master/fallzahlen_kanton_alter_geschlecht_csv/COVID19_Fallzahlen_Kanton_ZH_altersklassen_geschlecht.csv')
    df=df[['Week', 'Year', 'AgeYearCat', 'NewConfCases']]
    df['WOY'] = df['Year'].astype(str)+'-'+df['Week'].astype(str)+'-1'
    
    strptm = lambda s: datetime.strptime(s, '%Y-%W-%w')
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(by=['WOY', 'AgeYearCat']).sum().reset_index()
    df=df.pivot(index='WOY', columns='AgeYearCat', values='NewConfCases').drop(['unbekannt','100+'], axis=1).fillna(0).astype(int)
    #print(df)

    fireplot(df, country='Zürich', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Age Groups', legend=True, age_groups=True)


def Sweden():    
    print('Creating Fireplot for Sweden')
    df = pd.read_excel('https://fohm.maps.arcgis.com/sharing/rest/content/items/b5e7488e117749c19881cce45db13f7e/data').drop('Totalt_antal_fall', axis=1)
    df[df.columns[1:]] = df[df.columns[1:]].applymap(negative_to_zero)
    
    df['WOY'] = pd.to_datetime(df['Statistikdatum'], format='%Y-%m-%d').map(lambda x: str(x.weekofyear)) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    df=df.rename(columns=lambda s: s.replace('_', ' '))
    df = df[df.sum().sort_values(ascending=False).index]
    
    fireplot(df, country='Sweden', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Provinces', legend=True)


def Sweden_PC():
    print('Creating Fireplot for Sweden (Per Capita)')
    df = pd.read_excel('https://fohm.maps.arcgis.com/sharing/rest/content/items/b5e7488e117749c19881cce45db13f7e/data').drop(['Totalt_antal_fall', 'Sörmland'], axis=1)
    df=df.rename(columns=lambda s: s.replace('_', ' '))

    pop = pd.read_csv('Data/Sweden_Population.csv')[['County', 'Population']]
    pop_dict_list=pop.to_dict(orient='records')

    pop_dict={}
    for d in pop_dict_list:
        state = d['County'].strip()
        pop_dict[state]=int(d['Population'].replace(',',''))

    #print(pop_dict)
    def test(x):
        if x.name=='Statistikdatum':
            return x
        return (x/pop_dict[x.name]) * 10000

    
    df.index = df['Statistikdatum']
    df = df.drop('Statistikdatum', axis=1)
    df = df.apply(test)
    df[df.columns[1:]] = df[df.columns[1:]].applymap(negative_to_zero)
    #print(df)
    df = df[df.iloc[-1].sort_values(ascending=False).index].fillna(0)

    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    df[df.columns] = df[df.columns].applymap('{:.1f}'.format).applymap(float)
    #df = df[df.tail(7).sum().sort_values(ascending=False).index]
    df = df[df.sum().sort_values(ascending=False).index]

    #print(df)
    fireplot(df, country='Sweden', Title='Fireplot Sweden (Cases per 10k persons)',  grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Provinces', legend=True, per_capita=True)

    return   

def Sweden_Age():
    print('Creating Fireplot for Sweden (Age Groups)')

    # Get File List
    r = requests.get('https://api.github.com/repos/adamaltmejd/covid/git/trees/14a910ab0047035d1ad324b7e7a112e1973401f4')
    j = json.loads(r.content.decode())
    file_names = [i['path'] for i in j['tree']][1:85:7] + [i['path'] for i in j['tree']][85::5] # This should give us the totals so far, now we need to get the data from each xlsx file on that day
    # We go by 7 then by 5 because they stop reporting on weekends

    # Get the files and throw em together into one dataframe
    df =pd.read_excel('https://raw.githubusercontent.com/adamaltmejd/covid/master/data/FHM/'+file_names[0], sheet_name='Totalt antal per åldersgrupp').rename(columns={'Totalt_antal_fall':file_names[0][-15:-5]})
    df.index = df['Åldersgrupp']
    df = df[[file_names[0][-15:-5]]]
    the_big_dict= df.to_dict()
    for fname in file_names[1:]:
        # Using dicts here should make this slightly faster
        date=fname[-15:-5]
        df_temp = pd.read_excel('https://raw.githubusercontent.com/adamaltmejd/covid/master/data/FHM/'+fname, sheet_name='Totalt antal per åldersgrupp').rename(columns={'Totalt_antal_fall':date})
        df_temp.index = df_temp['Åldersgrupp']
        the_big_dict[date] = df_temp.to_dict()[date]
        #df = df.join(df_temp[[fname[-15:-5]]])

    df=pd.DataFrame(data=the_big_dict).fillna(0)
    # Some formatting
    df.loc['80+'] = df.loc[['Ålder_80_90', 'Ålder_90_plus', 'Ålder_80_89']].sum()
    df = df.T.drop(['Ålder_80_90', 'Ålder_90_plus', 'Ålder_80_89'], axis=1).diff().fillna(0).astype(int)
    df[df.columns] = df[df.columns].applymap(negative_to_zero)
    df = df[['Ålder_0_9',  'Ålder_10_19',  'Ålder_20_29',  'Ålder_30_39',  'Ålder_40_49',  'Ålder_50_59',  'Ålder_60_69',  'Ålder_70_79', '80+',  'Uppgift saknas']]
    renamer = lambda s: s.replace('Ålder_','').replace('_','-').replace('Uppgift saknas','Unkown')
    df.rename(columns=renamer, inplace=True)

    # Week of Year
    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str)
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    #print(df)
    df.index=df['WOY']
    df.drop('WOY', axis=1, inplace=True)
    #print(df)

    fireplot(df, country='Sweden_By_Age', Title='Fireplot Sweden (Age Groups)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Age Groups', legend=True, age_groups=True)



def Australia():
    print('Creating Fireplot for Australia')
    df = pd.read_csv('https://raw.githubusercontent.com/M3IT/COVID-19_Data/master/Data/COVID_AU_state.csv')
    df=df.pivot_table(index='date', columns='state', values='confirmed')
    df[df.columns] = df[df.columns].applymap(negative_to_zero)
    #print(df)
    #quit()

    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()
    df = df[df.sum().sort_values(ascending=False).index]

    #print(df) 
    fireplot(df, country='Australia', Title='Fireplot Australia (Cases)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Provinces', legend=True)

def Australia_PC():
    print('Creating Fireplot for Australia (Per Capita)')
    pop_temp = pd.DataFrame(pd.read_html('https://en.wikipedia.org/wiki/Demography_of_Australia')[4])[['State/territory', 'Population(2016 census)']]
    pop = pd.DataFrame(pop_temp['State/territory'])
    pop['Population'] = pop_temp['Population(2016 census)']
    pop_dict_list=pop.to_dict(orient='records')
    
    pop_dict={}
    for d in pop_dict_list:
        state = d['State/territory']
        pop_dict[state]=d['Population']
    #print(pop_dict)

    df = pd.read_csv('https://raw.githubusercontent.com/M3IT/COVID-19_Data/master/Data/COVID_AU_state.csv')
    df=df.pivot_table(index='date', columns='state', values='confirmed')
    df[df.columns] = df[df.columns].applymap(negative_to_zero)
    
    df['WOY'] = pd.to_datetime(df.index, format='%Y-%m-%d').weekofyear.astype(str) # week of year column
    strptm = lambda s: datetime.strptime('2020-'+s+'-1', "%Y-%W-%w")
    df['WOY'] = df['WOY'].map(strptm).astype(str)
    df=df.groupby(['WOY']).sum()

    def test(x):
        return (x/pop_dict[x.name]) * 10000
    
    df = df.apply(test)
    df = df[df.sum().sort_values(ascending=False).index]
    df[df.columns] = df[df.columns].applymap('{:.1f}'.format).applymap(float)

    fireplot(df, country='Australia', Title='Fireplot Australia (Cases per 10k persons)', grouped_by_week=True, caption=r'(*) Data from last week is incomplete', xlabel='Provinces', legend=True, per_capita=True, w_rat=2.0)



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
        functions = [Australia, Australia_PC, Brazil, Czechia_Age, Germany, G20, Holland, Florida_Age, USA, USA_per_capita, Italy, Italy_PC, Europe, Sweden, Sweden_PC, Switzerland, Switzerland_PC, Zurich, Sweden_Age]
        Group_Size = 6
        Partitions = int(len(functions)/Group_Size)+1
        for partition in range(Partitions):
            for f in functions[partition*Group_Size: (partition+1)*Group_Size]:
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
        #Czechia_Age()
        #Germany()
        #G20(partitions=0)
        Holland()
        #Russia() # Data wierd
        #USA(partitions=0)
        #USA_by_region()
        #USA_per_capita(partitions=0)
        #USA_per_capita(partitions=3)
        #Florida_Age()
        #Spain() # Data Incomplete
        #UK() # Data Incomplete
        #Italy()
        #Italy_PC()
        #India() # Data Incomplete
        #Europe()
        #World()
        #Switzerland()
        #Switzerland_PC()
        #Zurich()
        #Sweden()
        #Sweden_PC()
        #Sweden_Age()
        #Australia()
        #Australia_PC()

    #compression()
        