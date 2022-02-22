#!/usr/bin/env python
# coding: utf-8

# # VISUALIZING IMMIGRATION DATA USING PYTHON
# <p>The project represents immigration data visually using python libraries. The initial steps include applying data exploration and data cleaning techniques to make the data analysis ready. Python's <code>matplotlib</code> library has been primarily used to depict some basic and specialized visualization. Apart from it, the case study also includes libraries such as <code>seaborn</code> for regression plots and <code>folium</code> for creating maps and geospatial data.</p>

# In[275]:


get_ipython().system('pip install openpyxl==3.0.9')
get_ipython().system('pip3 install folium==0.5.0')


# Importing all necessary packages

# In[276]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.patches as mpatches
import folium
mpl.style.use('ggplot') 


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


file = 'C:\\Users\\lenovo\\Desktop\\Success\\Project\\Projects\\Project5_Immigration\\Canada.xlsx'


# In[5]:


df = pd.read_excel(file, sheet_name = 'Canada by Citizenship', skiprows = range(20), skipfooter = 2)


# ### Data Exploration
# Exploring the contents of the Dataset

# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.index


# In[9]:


df.isna().sum()


# In[10]:


df['Coverage'].value_counts()


# In[11]:


df['AreaName'].value_counts()


# In[12]:


df['RegName'].value_counts()


# In[13]:


df['DevName'].value_counts()


# In[14]:


df.shape


# In[15]:


df.head(2)


# ### Data Cleaning

# Dropping columns that we do not need for further analysis and visualizations

# In[16]:


df.drop(['AREA', 'REG', 'DEV'], axis = 1, inplace = True)


# In[17]:


df.head()


# In[18]:


df.drop(['Unnamed: 43', 'Unnamed: 44', 'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 47', 'Unnamed: 48', 'Unnamed: 49', 'Unnamed: 50'], axis = 1, inplace = True)


# In[19]:


df.head()


# Renaming column names 

# In[20]:


df.rename(columns = {'OdName': 'Country', 'AreaName' : 'Continent', 'RegName' : 'Region'}, inplace = True)
df.head(2)


# In[21]:


df.head()


# In[22]:


df['Total'] = df.sum(axis=1)


# In[23]:


df.head()


# In[24]:


df.describe()


# In[25]:


df.set_index('Country', inplace = True)


# In[26]:


df


# In[27]:


df.loc[['India']]


# In[28]:


df.loc[['Japan'],[2013]]


# In[29]:


df.columns.get_loc(1980)


# To avoid ambuigity, converting the column names into strings: '1980' to '2013'.

# In[30]:


df.columns = list(map(str, df.columns))  # -> easier df.columns = df.columns.astype(str)


# In[31]:


years = list(map(str, range(1980, 2014)))
years


# In[32]:


df.head(1)


# In[33]:


df[df['Continent'] == 'Asia']


# In[34]:


df[(df['Continent'] == 'Asia') & (df['Region'] == 'Southern Asia')]


# In[35]:


df.loc[['India']]


# ### Visualizing Data using Matplotlib

# In[36]:


India = df.loc['India', years].plot(figsize = (10,5))
plt.title('Immigration from India to Canada')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')
plt.show()


# In[37]:


Haiti = df.loc['Haiti', years]
Haiti.index = Haiti.index.map(int) # changing the index values of Haiti to type integer for plotting
Haiti.plot(kind='line', figsize = (10,5))

plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')
plt.text(2003, 6000, '2010 Earthquake')
plt.show()


# In[38]:


Bhutan = df.loc['Bhutan', years]
Bhutan.index = Bhutan.index.map(int)
Bhutan.plot(kind = 'line', figsize = (10,5))

plt.title('Immigration from Bhutan')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')
plt.text(2002,1500, 'Bhutan Refugee Crisis')
plt.show()


# In[39]:


years = list(map(str, range(1980, 2014)))
years
In_Ch= df.loc[['India','China'],years]
In_Ch


# In[40]:


In_Ch = In_Ch.transpose()
In_Ch.head()


# In[41]:


In_Ch.index = In_Ch.index.map(int)
In_Ch.plot(kind = 'line', figsize = (10,5))
plt.title('Immigration from India and China')
plt.xlabel('Years')
plt.ylabel('Number of Immigrants')
plt.show()


# In[42]:


df.head()


# In[43]:


# Checking to know which Countries are at top5 in terms of immigration
df_top5 = df.nlargest(5, 'Total') #Check what nlargest returns
df_top5


# In[44]:


years = list(map(str, range(1980, 2014)))
years

top5 = df.loc[['India','China','United Kingdom of Great Britain and Northern Ireland','Philippines','Pakistan'],years]
#print(top5)
top5 = top5.transpose()
top5.plot(kind = 'area',stacked=False,alpha=0.25, figsize = (20,8))
plt.title('Countries with largest Immigrations to Canada', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Number of Immigrations', fontsize=15)
plt.show()


# In[45]:


df.sort_values(by = ['Total'], axis = 0, inplace = True)
bottom_5 = df.head(5)
bottom_5 


# In[46]:


bottom_five = bottom_5.loc[['Palau','Marshall Islands','Western Sahara','San Marino','New Caledonia'],years]


# In[47]:


bottom_five = bottom_five.transpose()


# In[48]:


bottom_five.plot(kind = 'area',stacked = False,alpha=0.45, figsize = (15,5))
plt.title('Countries with lowest immigration rates to Canada')
plt.xlabel('Years')
plt.ylabel('Number of immigrants')
plt.show()


# In[49]:


df.sort_values(by = ['Total'], ascending = False, inplace = True)


# In[50]:


df.head()


# In[51]:


df['2013'].head()


# In[52]:


count, bin_edges = np.histogram(df['2013'])
print(count)
print(bin_edges)


# In[53]:


df['2013'].plot(kind = 'hist', figsize= (10,5), xticks=bin_edges)
plt.title('Number of immigrants to Canada in 2013')
plt.xlabel('Number of Immigrants')
plt.ylabel('Number of Countries')
plt.show()


# In[54]:


three_countries = df.loc[['Denmark','Norway','Sweden'],years]


# In[55]:


three_countries = three_countries.transpose()


# In[56]:


count, bin_edges = np.histogram(three_countries)


# In[57]:


three_countries.plot(kind = 'hist', stacked = False,bins=15, alpha = 0.6, figsize = (10,5), xticks = bin_edges, color=['coral', 'darkslateblue', 'mediumseagreen'])
plt.title('Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')
plt.show()


# In[58]:


df.head(2)


# In[59]:


countries_3 = df.loc[['Greece', 'Albania', 'Bulgaria'], years]
count,bin_edges = np.histogram(countries_3)
countries_3 =  countries_3.transpose()


# In[60]:


countries_3.plot(kind = 'hist', stacked = False, alpha = 0.5, bins = 15, xticks = bin_edges, figsize = (10,5), color= ['coral', 'darkslateblue', 'mediumseagreen'] )
plt.title('Immigration from Greece, Albania and Bulgaria from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of immigrant')
plt.show()


# In[61]:


df.head()


# In[62]:


df.loc['Iceland',years].plot(kind = 'bar', figsize = (20,5), color = 'mediumseagreen')
plt.title('Immigration from Iceland from 1980-2013')
plt.xlabel('Year')
plt.ylabel('Number of immigrants')
plt.annotate('', xy=(32, 70), xytext=(28, 20), arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', lw=2))
plt.show()


# The 2008 - 2011 Icelandic Financial Crisis was a major economic and political event in Iceland. Relative to the size of its economy, Iceland's systemic banking collapse was the largest experienced by any country in economic history. The crisis led to a severe economic depression in 2008 - 2011 and significant political unrest and an increase in the rate of immigration from the country.

# In[63]:


df.sort_values(by = ['Total'], ascending = True, inplace = True)


# In[64]:


top_fifteen = df['Total'].tail(15)
top_fifteen.plot(kind = 'barh', figsize = (12,12), color = 'steelblue')
plt.xlabel('Number of Immigrants')
plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')
for index, value in enumerate(top_fifteen):
    label = format(int(value), ',')
    
plt.annotate(label, xy=(value - 47000, index - 0.10), color='white')
plt.show()


# In[85]:


df.sort_values('Continent').head()


# In[89]:


df_pie = df.groupby('Continent')['Total'].sum()
df_pie


# In[137]:


colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1]

df_pie.plot(kind = 'pie', autopct='%1.1f%%',figsize = (15,6), colors = colors_list, explode=explode_list, labels = None)
plt.title('Immigration to Canada by Continent [1980 - 2013]')
plt.legend(labels=df_pie.index, loc='upper right', bbox_to_anchor=(2,0.5)) 
plt.show()


# In[151]:


df.head(2)


# In[161]:


df_japan = df.loc[['Japan'], years]
df_japan = df_japan.transpose()
df_japan.plot(kind = 'box', figsize = (10,8), color = 'blue')
plt.title('Immigration from Japan in the years[1980-2013]')
plt.ylabel('Number of Immigrants')
plt.show()


# In[159]:


df_japan.describe()


# In[194]:


df_top15 = df.tail(15)
years_80 = list(map(str,range(1980,1990)))
years_90 = list(map(str, range(1990, 2000))) 
years_00 = list(map(str, range(2000, 2010))) 

df_80 = df_top15.loc[:,years_80].sum(axis = 1)
df_90 = df_top15.loc[:,years_90].sum(axis = 1)
df_00 = df_top15.loc[:,years_00].sum(axis = 1)

new_df = pd.DataFrame({'1980s' : df_80, '1990s' : df_90, '2000s' : df_00})
new_df.head()


# In[192]:


new_df.describe()


# In[190]:


new_df.plot(kind = 'box')
plt.show()


# In[202]:


df.head()


# In[227]:


#df[years].sum(axis = 0).head()
df_total = pd.DataFrame(df[years].sum(axis = 0))
df_total.index = map(int, df_total.index) #Before reseting Years column was the index
df_total.reset_index(inplace = True)
df_total.columns = ['Year','Total']
df_total.head()


# In[236]:


df_total.plot(kind = 'scatter', x = 'Year', y = 'Total', figsize = (10,5), color = 'darkblue')

x = df_total['Year']      # year on x-axis
y = df_total['Total']     # total on y-axis
fit = np.polyfit(x, y, deg=1)

# plot line of best fit
plt.plot(x, fit[0] * x + fit[1], color='red')
plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))

plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.show()


# In[297]:


df.head()


# In[322]:


world_geo = r'custom.geo.json' # geojson file
world_map = folium.Map(location=[0, 0], zoom_start=2)


# In[323]:


world_map.choropleth(
    geo_data=world_geo,
    data=df,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada'
)

world_map


# In[335]:


df.head()


# In[345]:


total = pd.DataFrame(df[years].sum(axis = 0))


# In[347]:


total.index = map(int,total.index)
total.reset_index(inplace = True)


# In[351]:


total.columns = ['Year', 'Number of Immigrants']
total.head(1)


# In[363]:


plt.figure(figsize=(10, 5))
ax = sns.regplot(x = 'Year', y = 'Number of Immigrants', data = total, color = 'blue')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()


# In[ ]:




