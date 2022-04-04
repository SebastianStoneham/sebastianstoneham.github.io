```python
import pandas as pd
import random
import re
import glob 
from bs4 import BeautifulSoup
from near_regex import NEAR_regex 
import seaborn as sns
from numpy.random import default_rng
from tqdm import tqdm
from sec_edgar_downloader import Downloader

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import datetime
import os
import pandas as pd
import numpy as np
import pandas_datareader as pdr 
from datetime import datetime
import matplotlib.pyplot as plt 
```


```python
final_csv = 'output/sp500_accting_plus_textrisks_plus_ret.csv'
final = pd.read_csv(final_csv)
```

## Dataset Explanation

In explore_ugly, I took sp500_accting_plus_textrisks.csv and merged it with the returns for the week of March 9th. The new name of the csv is sp500_accting_plus_textrisks_plus_ret and the only difference is that the last column includes the returns for that week, this made it easier to analyze and create reports. 


```python
final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>Risk 1</th>
      <th>...</th>
      <th>prof_a</th>
      <th>ppe_a</th>
      <th>cash_a</th>
      <th>xrd_a</th>
      <th>dltt_a</th>
      <th>invopps_FG09</th>
      <th>sales_g</th>
      <th>dv_a</th>
      <th>short_debt</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.193936</td>
      <td>0.228196</td>
      <td>0.065407</td>
      <td>0.042791</td>
      <td>0.408339</td>
      <td>2.749554</td>
      <td>NaN</td>
      <td>0.074252</td>
      <td>0.143810</td>
      <td>-0.077905</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>2.0</td>
      <td>...</td>
      <td>0.177698</td>
      <td>0.193689</td>
      <td>0.180314</td>
      <td>0.028744</td>
      <td>0.103303</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.048790</td>
      <td>0.056170</td>
      <td>-0.028109</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.118653</td>
      <td>0.132161</td>
      <td>0.060984</td>
      <td>0.035942</td>
      <td>0.256544</td>
      <td>2.520681</td>
      <td>NaN</td>
      <td>0.033438</td>
      <td>0.088120</td>
      <td>-0.001101</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>7.0</td>
      <td>...</td>
      <td>0.178107</td>
      <td>0.037098</td>
      <td>0.448005</td>
      <td>0.076216</td>
      <td>0.709488</td>
      <td>2.211589</td>
      <td>NaN</td>
      <td>0.071436</td>
      <td>0.057566</td>
      <td>-0.038844</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABMD</td>
      <td>Abiomed</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Danvers, Massachusetts</td>
      <td>2018-05-31</td>
      <td>815094</td>
      <td>1981</td>
      <td>2.0</td>
      <td>...</td>
      <td>0.225749</td>
      <td>0.137531</td>
      <td>0.466354</td>
      <td>0.088683</td>
      <td>0.000000</td>
      <td>12.164233</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>-0.090781</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>345</th>
      <td>XYL</td>
      <td>Xylem</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Machinery</td>
      <td>White Plains, New York</td>
      <td>2011-11-01</td>
      <td>1524472</td>
      <td>2011</td>
      <td>5.0</td>
      <td>...</td>
      <td>0.123735</td>
      <td>0.116602</td>
      <td>0.093904</td>
      <td>0.024773</td>
      <td>0.288586</td>
      <td>2.131411</td>
      <td>NaN</td>
      <td>0.022568</td>
      <td>0.131538</td>
      <td>-0.112041</td>
    </tr>
    <tr>
      <th>346</th>
      <td>YUM</td>
      <td>Yum! Brands</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Restaurants</td>
      <td>Louisville, Kentucky</td>
      <td>1997-10-06</td>
      <td>1041061</td>
      <td>1997</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.341853</td>
      <td>0.346396</td>
      <td>0.142038</td>
      <td>0.000000</td>
      <td>1.071959</td>
      <td>8.046718</td>
      <td>NaN</td>
      <td>0.097687</td>
      <td>0.044192</td>
      <td>-0.122372</td>
    </tr>
    <tr>
      <th>347</th>
      <td>ZBRA</td>
      <td>Zebra</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Electronic Equipment &amp; Instruments</td>
      <td>Lincolnshire, Illinois</td>
      <td>2019-12-23</td>
      <td>877212</td>
      <td>1969</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.192104</td>
      <td>0.077691</td>
      <td>0.006368</td>
      <td>0.094884</td>
      <td>0.250478</td>
      <td>3.225952</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.160740</td>
      <td>-0.092335</td>
    </tr>
    <tr>
      <th>348</th>
      <td>ZBH</td>
      <td>Zimmer Biomet</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Warsaw, Indiana</td>
      <td>2001-08-07</td>
      <td>1136869</td>
      <td>1927</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.102051</td>
      <td>0.095139</td>
      <td>0.025078</td>
      <td>0.021081</td>
      <td>0.281545</td>
      <td>1.556915</td>
      <td>NaN</td>
      <td>0.007983</td>
      <td>0.184000</td>
      <td>-0.211926</td>
    </tr>
    <tr>
      <th>349</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>5.0</td>
      <td>...</td>
      <td>0.218709</td>
      <td>0.184409</td>
      <td>0.167518</td>
      <td>0.039584</td>
      <td>0.529320</td>
      <td>6.019250</td>
      <td>NaN</td>
      <td>0.027198</td>
      <td>0.080500</td>
      <td>-0.084422</td>
    </tr>
  </tbody>
</table>
<p>350 rows × 54 columns</p>
</div>




```python
final.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CIK</th>
      <th>Risk 1</th>
      <th>Risk 2</th>
      <th>Risk 3</th>
      <th>Risk 4</th>
      <th>Risk 5</th>
      <th>gvkey</th>
      <th>lpermno</th>
      <th>fyear</th>
      <th>sic</th>
      <th>...</th>
      <th>prof_a</th>
      <th>ppe_a</th>
      <th>cash_a</th>
      <th>xrd_a</th>
      <th>dltt_a</th>
      <th>invopps_FG09</th>
      <th>sales_g</th>
      <th>dv_a</th>
      <th>short_debt</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.500000e+02</td>
      <td>350.000000</td>
      <td>350.000000</td>
      <td>350.000000</td>
      <td>350.000000</td>
      <td>350.000000</td>
      <td>350.000000</td>
      <td>350.000000</td>
      <td>350.000000</td>
      <td>350.000000</td>
      <td>...</td>
      <td>350.000000</td>
      <td>350.000000</td>
      <td>350.000000</td>
      <td>350.000000</td>
      <td>350.000000</td>
      <td>330.000000</td>
      <td>0.0</td>
      <td>350.000000</td>
      <td>344.000000</td>
      <td>350.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.737542e+05</td>
      <td>2.671429</td>
      <td>1.614286</td>
      <td>9.102857</td>
      <td>2.545714</td>
      <td>1.280000</td>
      <td>45804.234286</td>
      <td>53757.260000</td>
      <td>2018.885714</td>
      <td>4333.465714</td>
      <td>...</td>
      <td>0.151481</td>
      <td>0.247846</td>
      <td>0.127078</td>
      <td>0.031470</td>
      <td>0.295356</td>
      <td>2.706785</td>
      <td>NaN</td>
      <td>0.025548</td>
      <td>0.112456</td>
      <td>-0.123394</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.566815e+05</td>
      <td>2.843186</td>
      <td>1.897766</td>
      <td>5.543859</td>
      <td>2.558635</td>
      <td>0.850687</td>
      <td>61452.674439</td>
      <td>30216.895683</td>
      <td>0.318613</td>
      <td>1954.535773</td>
      <td>...</td>
      <td>0.074714</td>
      <td>0.219934</td>
      <td>0.139121</td>
      <td>0.050450</td>
      <td>0.181912</td>
      <td>2.107665</td>
      <td>NaN</td>
      <td>0.027052</td>
      <td>0.111857</td>
      <td>0.095211</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.800000e+03</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1045.000000</td>
      <td>10104.000000</td>
      <td>2018.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>-0.323828</td>
      <td>0.009521</td>
      <td>0.002073</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.405435</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.610145</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.625150e+04</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>6321.750000</td>
      <td>19516.750000</td>
      <td>2019.000000</td>
      <td>2845.750000</td>
      <td>...</td>
      <td>0.102232</td>
      <td>0.091405</td>
      <td>0.032031</td>
      <td>0.000000</td>
      <td>0.175292</td>
      <td>1.265457</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.026330</td>
      <td>-0.154969</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.751825e+05</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>13973.000000</td>
      <td>58965.500000</td>
      <td>2019.000000</td>
      <td>3812.000000</td>
      <td>...</td>
      <td>0.139286</td>
      <td>0.162726</td>
      <td>0.072970</td>
      <td>0.009533</td>
      <td>0.282816</td>
      <td>2.166066</td>
      <td>NaN</td>
      <td>0.020574</td>
      <td>0.084983</td>
      <td>-0.103884</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.136887e+06</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>12.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>61654.750000</td>
      <td>82648.750000</td>
      <td>2019.000000</td>
      <td>5523.250000</td>
      <td>...</td>
      <td>0.187125</td>
      <td>0.337210</td>
      <td>0.168847</td>
      <td>0.043755</td>
      <td>0.387926</td>
      <td>3.301717</td>
      <td>NaN</td>
      <td>0.037675</td>
      <td>0.151700</td>
      <td>-0.062951</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.757898e+06</td>
      <td>19.000000</td>
      <td>11.000000</td>
      <td>35.000000</td>
      <td>15.000000</td>
      <td>5.000000</td>
      <td>316056.000000</td>
      <td>93436.000000</td>
      <td>2019.000000</td>
      <td>8742.000000</td>
      <td>...</td>
      <td>0.390384</td>
      <td>0.928562</td>
      <td>0.694612</td>
      <td>0.336795</td>
      <td>1.071959</td>
      <td>12.164233</td>
      <td>NaN</td>
      <td>0.138594</td>
      <td>0.761029</td>
      <td>0.048830</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 44 columns</p>
</div>



## Risk 1


```python
final['Risk 1'].describe()
```




    count    350.000000
    mean       2.671429
    std        2.843186
    min        0.000000
    25%        0.250000
    50%        2.000000
    75%        4.000000
    max       19.000000
    Name: Risk 1, dtype: float64



__Variable 1__
```Python 
new_risk_var1 = len(re.findall(
            NEAR_regex(['(incease|emerging|new)','(firms|competition|competitors|entrants)'],10),text)) 
```
__Explain and describe to readers your risk measurements__

I chose to analyze the risk competitors have on each firm in the S&P 500. Firms voice their concerns for the business, and the more a risk appears in the report, the more potential it has to affect the health of the company. 

__How were they measured? (Mechanical description)__

This risk variables observes each annual 10k report for each firm in the S&P 500 and counts the number of hits, times the risk is present. Anytime the word increase, emerging, or new is found within ten words of firms, competition, competitors, or entrants the amount of risk increases by 1. At the end, the code produces the total number of occurences within the 10k report.  

__Why did you choose them and what do you hope they capture? (Economic reasoning)__

My hope was to see how competitors may have taken the onset of Covid as an opportunity to have a larger part in the industry. After graphing the returns of the firsm with respect this risk variable during one of the first weeks of Covid, I relaized that no matter how high the risk compitiors played, almost all firms had negative returns. I was expecting firms with lower risk to have better returns than firms with higher risk. It makes sense, if the largest and most powerful companies in each industry performed poorly that week, then so were the smaller ones. 

__What are their statistical properties? (Do you have values for most/all firms, they should have variation within them, are they correlated with any accounting measures)__

The mean for this risk variable is 2.0, so on average firms in S&P 500 reffered to new compition twice in their annual reports. There were some outliers, 10 out of the 350 firms analyzed had their risk above 8 and they were mostly in either Health Care or Information Technology Industry.




```python
risk_return1 = sns.regplot(x="Risk 1", y ="R", data= final).set(title = "Risk and Return (Mar 9-13 2020)", xlabel = "Risk 1", ylabel = "Return")
```


    
![png](output_8_0.png)
    


## Risk 2


```python
final['Risk 2'].describe()
```




    count    350.000000
    mean       1.614286
    std        1.897766
    min        0.000000
    25%        0.000000
    50%        1.000000
    75%        2.000000
    max       11.000000
    Name: Risk 2, dtype: float64



__Variable 2__
```Python
new_risk_var2 = len(re.findall(
            NEAR_regex(['(supply|supplies|supplier|supply chain|materials)',
                    '(risk|bottleneck|shortage|decrease|problem|problems|disruption)'],10),text)) 
```
__Explain and describe to readers your risk measurements__

The second risk factor measures the risk a company's supply chain has on the health of the business. Firms are fearful of not being able to service their customers due to not being able to meet their needs because things are not getting to where they are supposed to.

__How were they measured? (Mechanical description)__

This risk variables observes each annual 10k report for each firm in the S&P 500 and counts the number of hits, times the risk is present. Anytime the word supply, supplies, supplier, supply chain, or materials is found within ten words of risk, bottleneck, shortage, decrease, problem, problems, or disruption, the amount of risk increases by 1. At the end, the code produces the total number of occurences within the 10k report.

__Why did you choose them and what do you hope they capture? (Economic reasoning)__

From my expierence of Covid, I know just how much the supply chain of all industries was affected. Ports were closed, shipping services delayed due to high demand and lack of resources, and many other factors inhibited supply chains from meeting the needs of their consumers. However, after creating a scatter plot that analyzed the level of risk and returns for the week of March 9th, the results showed that there were a lot of firms that did not include this risk in their annual report. I believe this is why the firms who did not think of this as a risk has a worse week in terms of returns compared to those who were aware of the risks and prepared properly 

__What are their statistical properties? (Do you have values for most/all firms, they should have variation within them, are they correlated with any accounting measures)__

The mean for this risk variable is 1.61 with a standard deviatopm of 1.89. The firms in the S&P 500 had a very similar worry of supply chain risks inm geneeral. It shows that this is always a concern for firms, and that events like Covid 19 can really impact the returns a firm sees because their inability to meet the needs. 



```python
risk_return2 = sns.regplot(x="Risk 2", y ="R", data= final).set(title = "Risk and Return (Mar 9-13 2020)", xlabel = "Risk 1", ylabel = "Return")

```


    
![png](output_12_0.png)
    


## Risk 3


```python
final['Risk 3'].describe()
```




    count    350.000000
    mean       9.102857
    std        5.543859
    min        0.000000
    25%        5.000000
    50%        8.000000
    75%       12.000000
    max       35.000000
    Name: Risk 3, dtype: float64



__Variable 3__
```Python
new_risk_var3 = len(re.findall(
            NEAR_regex(['(increase|rise)','(cost|rate|rates|expense)'],5),text))
```
__Explain and describe to readers your risk measurements__

Risk varibales 3-5 all measure the risk of the business cycle. Risk 3 specifically measures the increase of interest rates and cost of goods sold. In certain industires, specific time of the year can mean higher costs and less returns becasue of the nature of their business. 

__How were they measured? (Mechanical description)__

This risk variables observes each annual 10k report for each firm in the S&P 500 and counts the number of hits, times the risk is present. Anytime the word increase or rise is found within five words of cost, rate, rates, or expense, the amount of risk increases by 1. At the end, the code produces the total number of occurences within the 10k report.

__Why did you choose them and what do you hope they capture? (Economic reasoning)__

I chose this risk variable because one threat to a business regarding the business cycle is the risk of increasing rates and costs to produce. Again, firms that did not see this as a risk performed worse than those who did. During this week, every firm did poorly, no matter how much of a risk this was for them. 

__What are their statistical properties? (Do you have values for most/all firms, they should have variation within them, are they correlated with any accounting measures)__

The mean of this risk variable is 9.10 which is relatively high compared to others. It also had a standard deviation of 5.54 which shows how spread out the results were. 



```python
risk_return3 = sns.regplot(x="Risk 3", y ="R", data= final).set(title = "Risk and Return (Mar 9-13 2020)", xlabel = "Risk 1", ylabel = "Return")

```


    
![png](output_16_0.png)
    


## Risk 4


```python
final['Risk 4'].describe()
```




    count    350.000000
    mean       2.545714
    std        2.558635
    min        0.000000
    25%        1.000000
    50%        2.000000
    75%        4.000000
    max       15.000000
    Name: Risk 4, dtype: float64



__Variable 4__
```Python
new_risk_var4 = len(re.findall(
            NEAR_regex(['(decrease|shrink|reduce)','(demand|willingness|consumer|customer)'],10),text)) 
```
__Explain and describe to readers your risk measurements__

The fourth risk variable also measure the risk the business cycle has on these firms, but focuses on customers. It measures the concerns firms have about the reduce in demand from their consumers. 

__How were they measured? (Mechanical description)__

This risk variables observes each annual 10k report for each firm in the S&P 500 and counts the number of hits, times the risk is present. Anytime a word decrease, shrink, or reduce is found within ten words of demand, willingness, consumer, or customer, the amount of risk increases by 1. At the end, the code produces the total number of occurences within the 10k report.


__Why did you choose them and what do you hope they capture? (Economic reasoning)__

During Covid, demand for certain goods increased dramatically while the demand for other decreased due to the quarantine period. I wanted to identify any correlations between the two. 

__What are their statistical properties? (Do you have values for most/all firms, they should have variation within them, are they correlated with any accounting measures)__

The mean was 2.54 and which is similar to the other risk variables. The standard deviation for this risk variable was 2.55, the firms in the dataset thought of this as a similar level of risk.


```python
risk_return4 = sns.regplot(x="Risk 4", y ="R", data= final).set(title = "Risk and Return (Mar 9-13 2020)", xlabel = "Risk 1", ylabel = "Return")
```


    
![png](output_20_0.png)
    


## Risk 5


```python
final['Risk 5'].describe()
```




    count    350.000000
    mean       1.280000
    std        0.850687
    min        0.000000
    25%        1.000000
    50%        1.000000
    75%        2.000000
    max        5.000000
    Name: Risk 5, dtype: float64



__Variable 5__
```Python
new_risk_var5 = len(re.findall(
            NEAR_regex(['(employee|staff|workforce)','(decrease|low|shortage|risk|susceptible|threat)'],10),text)) 
```
__Explain and describe to readers your risk measurements__

This risk variable also measures the risk of the changing business cycle has on firms. Risk variable 5 focuses on measuring the risk of a shrinking workforce and lack of employees to fulfill the needs of consumers.  
__How were they measured? (Mechanical description)__

This risk variables observes each annual 10k report for each firm in the S&P 500 and counts the number of hits, times the risk is present. Anytime the word employee, staff,  or workforce is found within ten words of decrease, low, shortage, risk, susceptible, or threat, the amount of risk increases by 1. At the end, the code produces the total number of occurences within the 10k report.


__Why did you choose them and what do you hope they capture? (Economic reasoning)__

I wanted to show the effects Covid had on workers staying home due to mandates, which affected the capacity firms had. 

__What are their statistical properties? (Do you have values for most/all firms, they should have variation within them, are they correlated with any accounting measures)__

Suprsingly the max risk level was 5, which is extremely low compared to the other risk variables. The mean was 1.28, so on average the firms referred to this risk once in their annual report. I do not beleive any company imagined their work staff would be forced to drop to practically nothing, which is why all the firms had negative returns that week. 


```python
risk_return5 = sns.regplot(x="Risk 5", y ="R", data= final).set(title = "Risk and Return (Mar 9-13 2020)", xlabel = "Risk 1", ylabel = "Return")
```


    
![png](output_24_0.png)
    


## Examples of Risk Matches
"Because the Company currently obtains certain components from single or limited sources, the Company is subject to significant supply and pricing risks" APPL 10K


"Therefore, the Company remains subject to significant risks of supply shortages and price increases that could materially adversely affect its financial condition and operating results" APPL 10K

"In addition to an adverse impact on demand for the Company’s products, uncertainty about, or a decline in, global or regional economic conditions could have a significant impact on the Company’s suppliers, contract manufacturers, logistics providers, distributors, cellular network carriers and other channel partners." APPL 10K

These are a few example of good hits my code was able to identify. The reflect exactly what I was looking for. I wanted to see how often these firms were adressing the concerns. I thought the more they were addressed, the more prepared they would be, especially in terms of supply chain. 





```python

```
