Milestone Report: Predicting Hotel Cancellations
================
Matthew Merrill

# Project Proposal

My project will seek to solve a common but surging issue for the hotel
industry, predicting cancellations and maintaining brand integrity among
surging customer interest in booking through online travel agents,
rather than direct. According to a study conducted by D-Edge Hospitality
Solutions, cancellation rates in the hotel industry increased more than
8 percent from 2014 to 2017\[1\]

The impact on the industry has come from pressure primarily stemming
from online travel agencies and their adoption of ‘Risk Free
Reservations’. While OTA’s use cancellations as a way to expand their
market availability and retain customer loyalty, hotels risk the ability
to forecast revenue and maintain brand integrity in the process\[2\]

Hotels often desire to create a personalized ease of service for
customers from the moment of booking, but OTA’s want their customers to
adopt a ‘book now, ask questions later’ mentality. The preference for
customers often falls towards the OTA’s, because they will often
advertise a lower price than booking direct, as they absorb the
cancellation risk, which creates a strain on the hotels’ customer
relationship and diminishes the booking experience. While OTA’s do draw
in customers and expand outreach for hotels, there is an opportunity to
optimize customer channels and lower risk, while increasing real-time
income.

The outcome of this project will give the client insights into
predicting cancellations and provide suggestions for modifying their
cancellation policy. This will provide the hotel with the ability to
optimize customer channels, maintain brand integrity and increase
customer loyalty.

The dataset was obtained from [Science
Direct](https://www.sciencedirect.com) and contains a collection of
observations taken from 2015 to 2017 of guest bookings for two hotels,
both located in Portugal. The data was collected directly from the
hotels’ PMS (property management system) database and was relatively
clean and structured upon retrieval. Each observation represents a
booking, with the variables capturing details including when it was
booked, dates of stay, through which operator the guest chose to go
through, if that booking was cancelled and the average daily rate. Once
this project is complete, I will be able to provide insights on the
following questions:

> Can we predict when and if a guest may cancel a reservation? What
> inferences can we make that would help optimize customer channels and
> lower the overall risk? What customers should we be targeting to book
> directly through the hotel?

I will attempt to solve this problem by investigating where
cancellations primarily occur and during what time of the year. After
drawing insights from the exploratory data analysis phase, the dataset
will be modified for the modeling process with the goal of predicting
the cancelation column of the set. Insights from the model and from the
exploratory phase will lead to suggestions to the client that will help
them forecast cancellations and optimize customer channels as mentioned
above.

The client will receive the working model in the form of a web
application as well as a summary paper and slide deck. The slide deck
will summarize the findings and attempt to “sell” the work to the
client, and the paper will summarize the details. Code will be provided
for inspection and reproducibility for the client.

# Data Wrangling and Cleaning Steps

The data I obtained was relatively clean to begin with, however there
were a few instances of “NULL” and empty entries as well as mislabeled
entries that needed clarification. The convention for null data was a
string with 7 spaces before NULL (“NULL”), so this was found and
replaced with `numpy`’s NAN entry for ease of identification. At this
point it was a matter of understanding the data to replace NAN values
with appropriate labels for the variable.

The variables with NAN entries were the company, agent and country
columns. Upon more exploration of the data and its source website, it
was discovered that for company and agent, null entries correlated to
customers that did not go through a company or agent to book. To correct
this, the original company and agent names were replaced with string
numerics to protect anonymity so to continue with this convention, nan
entries were replaced with “0” to represent “no agent” or “no company”.
The country column had no added clarification for the nan entries, so it
was decided that “UNK” for unknown would be used instead.

The last step for cleaning was to create a datetime column for a guest’s
date of arrival. The data provided the day, month and year separately
for arrival, so these were combined and converted to datetime format to
allow more versatility in analysis.

Lastly, the data was explored to identify any outliers that may skew
analysis. A few peculiar entries were found. For example, one group
booked 2 years in advance and one brought 10 children and 10 kids, one
guest also had 26 previous cancellations. Upon further exploration there
was no clear evidence to rule out these observations, so no steps were
taken to alter these data.

The code below highlights the data wrangling part. I have carried out
the analysis in the RStudio IDE which also allows me to use Python code
using the `reticulate` package.

``` r
library(tidyverse)
library(reticulate)
```

Read in the data.

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
hotels = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv')

hotels.describe(exclude=[np.number]).T
```

    ##                           count unique         top    freq
    ## hotel                    119390      2  City Hotel   79330
    ## arrival_date_month       119390     12      August   13877
    ## meal                     119390      5          BB   92310
    ## country                  118902    177         PRT   48590
    ## market_segment           119390      8   Online TA   56477
    ## distribution_channel     119390      5       TA/TO   97870
    ## reserved_room_type       119390     10           A   85994
    ## assigned_room_type       119390     12           A   74053
    ## deposit_type             119390      3  No Deposit  104641
    ## customer_type            119390      4   Transient   89613
    ## reservation_status       119390      3   Check-Out   75166
    ## reservation_status_date  119390    926  2015-10-21    1461

# Data Cleaning

``` python
# convert 'NULL' entries to np.nan
cols = ['agent', 'company']
for col in cols:
    for i in range(len(hotels[col])):
        if (hotels[col].iloc[i] == '       NULL'):
            hotels[col].iloc[i] = np.nan
        else:
            continue
```

``` python
# Insert 0 for each missing Children entry and convert to integer.
hotels.children = pd.to_numeric(hotels.children.replace(np.nan, 0), downcast = 'integer')
```

``` python
# Convert missing values for Company and Agent to 0, meaning 'no agent' and 'no company'.
cols = ['company', 'agent']
for col in cols:
        hotels[col] = hotels[col].replace(np.nan, '0')
```

``` python
# If country of origin is missing, replace as 'UNK' for unknown.
hotels['country'] = hotels['country'].replace(np.nan, 'UNK')
```

``` python
# convert reservation_status_date to datetime
hotels['reservation_status_date'] = pd.to_datetime(hotels['reservation_status_date'])
```

``` python
hotels.name = 'Hotels'
null_data = hotels[hotels.isnull().any(axis=1)]
if null_data.empty:
    print(hotels.name + ' contains no null values')
else:
    print(hotels.name + ' does contain null values')
```

    ## Hotels contains no null values

# Exploratory Data Analysis

``` python
fig, ax = plt.subplots(figsize=(10, 6))
hotels.market_segment.value_counts().plot.bar(ax=ax)
ax.set_xlabel('Market Segment')
ax.set_ylabel('Count')
ax.set_title('Booking source', fontsize = 15, weight = 'bold')
```

![](Milestone_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

# Exploratory Data Analysis:

Before beginning data analysis it was hypothesised that over time,
customer bookings from online travel agencies increase along with the
portion of cancellations from these bookings. I followed up with asking
a number of questions to gain insight into other variables and factors
that may contribute to my proposal.

### Initial findings of numeric columns:

  - About 25% of bookings from 2015 to 2017 were cancelled
  - 50% of bookings occurred between the 16th and 38th week of the year.
  - The median lead time is 69 with an IQR of 142
  - Guests typically do not spend any time on a waiting list, but the
    max was 391 days.
  - The average daily rate (ADR) was about 94.5 with a max of 5400.

### Initial findings of categorical columns:

  - August is the most popular month between both hotels.
  - Most people who visit are from Portugal (the country where both
    hotels are located).
  - Most bookings are made with an online travel agent.
  - Most bookings are made with travel agent ‘9.0’.
  - Most people do not go through a
company.

### What is the most common means of booking? Through which booking channel do most cancellations occur?

  - It was found that 47% of bookings between both hotels originated
    from OTAs.
  - Of those who booked through an OTA, 37% cancelled.
  - Of all cancellations, regardless of booking channel, 47% came from
    OTAs.

This compares sharply to customers who booked direct, with only 11% of
bookings from 2015 to 2017 occuring direct with only 15% of those
customers
cancelling.

### How have the rate of cancellations changed over time relative to how the market segment has changed?

#### In 2015…

  - 28% of bookings were through an online travel agency and 27% of
    those were cancelled.
  - 37% of all bookings were cancelled.
  - 20% of cancellations were from online travel agency bookings.

#### In 2016…

  - 49% of bookings were through an online travel agency and 36% of
    those were cancelled.
  - 36% of all bookings were cancelled.
  - 48% of cancellations were from online travel agency bookings.

#### In 2017…

  - 56% of bookings were through an online travel agency and 41% of
    those were cancelled.
  - 39% of all bookings were cancelled.
  - 59% of cancellations were from online travel agency bookings.

Bookings from online travel agencies increased from 28% to 56% from 2015
to 2017. During that time, cancellations increased from 37% to 39%, with
cancellations from OTA’s increasing from 27% to 41%.

### Does lead time correlate to cancellations?

``` python
canceled = hotels[hotels.is_canceled == 1]
not_canceled = hotels[hotels.is_canceled == 0]
```

``` python
fig, ax = plt.subplots(3, figsize=(10, 8), sharex=True, sharey=True, gridspec_kw={'hspace': 0.5})
hotels.lead_time.hist(ax=ax[0], bins = 30)
canceled.lead_time.hist(ax=ax[1], bins = 30)
not_canceled.lead_time.hist(ax=ax[2], bins = 30)
ax[0].set_title('All Bookings')
ax[1].set_title('Canceled')
ax[2].set_title('Not Canceled')
for ax in ax.flat:
    ax.set(xlabel='Lead Time', ylabel='Count')
```

    ## [Text(0, 0.5, 'Count'), Text(0.5, 0, 'Lead Time')]
    ## [Text(0, 0.5, 'Count'), Text(0.5, 0, 'Lead Time')]
    ## [Text(0, 0.5, 'Count'), Text(0.5, 0, 'Lead Time')]

``` python
plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` python
median_lead_canc = np.median(canceled.lead_time)
median_lead_not_canc = np.median(not_canceled.lead_time)
```

  - The median lead time for canceled bookings is 113
  - The median lead time for not canceled bookings is 45

The average lead time for canceled bookings is 2.5 times greater on
average than for those who did not cancel. It may be that people booking
far out are looking for better deals, or other options opening up may be
more likely given a larger lead
time.

### Do people who book through online travel agencies have larger median lead times?

``` python
ta_lead_time = hotels[hotels.market_segment == 'Online TA']['lead_time']
ta_canceled_lead_time = hotels[(hotels.market_segment == 'Online TA') & (hotels.is_canceled == 1)]['lead_time']
not_ta_lead_time = hotels[hotels.market_segment != 'Online TA']['lead_time']
# plot hist of lead time for canceled and non-canceled guests
fig, ax = plt.subplots(3, figsize=(10, 8), sharex=True, sharey=True, gridspec_kw={'hspace': 0.5})
ta_lead_time.hist(ax=ax[0], bins = 30)
not_ta_lead_time.hist(ax=ax[1], bins = 30)
ta_canceled_lead_time.hist(ax=ax[2], bins = 30)

ax[0].set_title('Online TA Lead Time')
ax[1].set_title('All Other Methods of Booking')
ax[2].set_title('Canceled lead times from Online TAs')
for ax in ax.flat:
    ax.set(xlabel='Lead Time', ylabel='Count')
```

    ## [Text(0, 0.5, 'Count'), Text(0.5, 0, 'Lead Time')]
    ## [Text(0, 0.5, 'Count'), Text(0.5, 0, 'Lead Time')]
    ## [Text(0, 0.5, 'Count'), Text(0.5, 0, 'Lead Time')]

``` python
plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` python
median_lead_ota = np.median(hotels.lead_time)
median_lead_not_ota = np.median(hotels.market_segment != 'Online TA')
```

  - The median lead time for OTA bookings was 113
  - The median lead time for all other methods of booking was 113

Online travel agencies do not have a higher lead time on average, there
may be another area here to explore. Lead time and bookings from OTA’s
seem to correlate to higher rates of cancellation.

### How have distribution channels changed over time?

``` python
hotels_2015 = hotels[hotels.arrival_date_year  == 2015]
hotels_2016 = hotels[hotels.arrival_date_year  == 2016]
hotels_2017 = hotels[hotels.arrival_date_year  == 2017]
print(hotels_2015.head())
```

    ##           hotel  is_canceled  ...  reservation_status  reservation_status_date
    ## 0  Resort Hotel            0  ...           Check-Out               2015-07-01
    ## 1  Resort Hotel            0  ...           Check-Out               2015-07-01
    ## 2  Resort Hotel            0  ...           Check-Out               2015-07-02
    ## 3  Resort Hotel            0  ...           Check-Out               2015-07-02
    ## 4  Resort Hotel            0  ...           Check-Out               2015-07-03
    ## 
    ## [5 rows x 32 columns]

``` python
# plot hist per week for arrival_date_week_number
fig, ax = plt.subplots(3, figsize=(8, 10), sharex=True, sharey=True, gridspec_kw={'hspace': 0.5})
hotels_2015.market_segment.hist(ax=ax[0], bins = 30)
hotels_2016.market_segment.hist(ax=ax[1], bins = 30)
hotels_2017.market_segment.hist(ax=ax[2], bins = 30)

ax[0].set_title('2015', fontsize = 15, weight = 'bold')
ax[1].set_title('2016', fontsize = 15, weight = 'bold')
ax[2].set_title('2017', fontsize = 15, weight = 'bold')
for ax in ax.flat:
    ax.set(xlabel='distribtion_channel\n', ylabel='Count')
```

    ## [Text(0, 0.5, 'Count'), Text(0.5, 0, 'distribtion_channel\n')]
    ## [Text(0, 0.5, 'Count'), Text(0.5, 0, 'distribtion_channel\n')]
    ## [Text(0, 0.5, 'Count'), Text(0.5, 0, 'distribtion_channel\n')]

``` python
plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` python
# list of data by year
data = [hotels_2015, hotels_2016, hotels_2017]
loc = ['City Hotel', 'Resort Hotel']
for hotel in data:
    ms = hotel.groupby(['hotel', 'market_segment']).agg({'market_segment': 'count'})
    ms_pct = ms.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    for place in loc:
      print(ms_pct.loc[place].loc['Online TA'])
```

    ## market_segment    22.387078
    ## Name: Online TA, dtype: float64
    ## market_segment    37.310561
    ## Name: Online TA, dtype: float64
    ## market_segment    51.672784
    ## Name: Online TA, dtype: float64
    ## market_segment    42.83406
    ## Name: Online TA, dtype: float64
    ## market_segment    58.081285
    ## Name: Online TA, dtype: float64
    ## market_segment    50.641172
    ## Name: Online TA, dtype: float64

Percent booking from OTAs for the city hotel were… - 22.4% in 2015. -
51.6% in 2016. - 58.1% in 2017. Percent booking from OTAs for the resort
hotel were… - 37.3% in 2015. - 42.8% in 2016. - 50.6% in 2017.

In 2015, the city hotel received 22% of it’s bookings through OTAs while
the resort hotel received 37%. By 2017, city hotel recieved 58% of its
bookings through an online TA with the resort hotel increasing their
portion to 51%. It is clear that both increased significantly over time
with the city hotel taking a major leap from 2015 to 2016, increasing
from 22% to
52%.

### Are guests with kids more likely to cancel?

``` python
# Calculate number of guests with babies who canceled and who did not cancel
babies = hotels.loc[(hotels.babies != 0)]['babies'].count()
not_canceled_babies = hotels.loc[(hotels.is_canceled == 0) & (hotels.babies != 0)]['babies'].count()

# Do the same calculation for famalies with children and with no children
children = hotels.loc[(hotels.children != 0)]['children'].count()
not_canceled_children = hotels.loc[(hotels.is_canceled == 0) & (hotels.children != 0)]['children'].count()

# No children
conditions = (hotels.children == 0) & (hotels.babies == 0)
no_kids = hotels.loc[conditions]['adults'].count()
not_canceled_no_kids = hotels.loc[(hotels.is_canceled == 0) & (conditions)]['adults'].count()
```

``` python
# percentage of famalies who did not cancel
perc_babies = str(round(not_canceled_babies/babies, 2)*100) + '%'
perc_children = str(round(not_canceled_children/children, 2)*100) + '%'
perc_no_kids = str(round(not_canceled_no_kids/no_kids, 2)*100) + '%'
```

``` python
d1 = {'guest_type': ['babies', 'children', 'without_children'], \
     'not_cancled': [not_canceled_babies, not_canceled_children, not_canceled_no_kids],\
    'percent': [perc_babies, perc_children, perc_no_kids]}

d = {'guest_type': ['Babies', 'Children', 'No Kids'], \
    'percent': [perc_babies, perc_children, perc_no_kids]}

pie_data = pd.DataFrame(data=d)
pie_data
```

    ##   guest_type percent
    ## 0     Babies   82.0%
    ## 1   Children   64.0%
    ## 2    No Kids   63.0%

``` python
import re

def donut_plot(data, plotnumber):
    # create donut plots
    startingRadius = 0.7 + (0.3* (len(data)-1))
    for index, row in data.iterrows():
        scenario = row["guest_type"]
        percentage = row["percent"]
        textLabel = scenario
        
        percentage = int(re.search(r'\d+', percentage).group())
        remainingPie = 100 - percentage
        
        donut_sizes = [remainingPie, percentage]
        
        plt.text(0.01, startingRadius - 0.25, textLabel, ha='right', va='bottom', fontsize = 12, fontweight = 'bold')
        
        plt.pie(donut_sizes, radius=startingRadius, startangle=90, colors=['lightgray', 'tomato'],
                wedgeprops={"edgecolor": "white", 'linewidth': 1.5})
        
        startingRadius-=0.3

    # equal ensures pie chart is drawn as a circle (equal aspect ratio)
    plt.axis('equal')

    # create circle and place onto pie chart
    circle = plt.Circle(xy=(0, 0), radius=0.35, facecolor='white')
    plt.gca().add_artist(circle)
    plt.savefig('donutPlot' +plotnumber+ '.jpg')
    plt.show()
```

``` python
# plot the proportion of cancellations based on whether guests had babies, children or none.
plt.title('82% of guests with babies \n did not cancel \n', fontsize = '18', fontweight = 'bold')
donut_plot(pie_data, '1')
```

![](Milestone_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

Guests with babies followed through with their booking 83% of the time
and 86% of guests who stayed with babies had at least one special
request.

### How do the number of stays vary by month based on family size?

``` python
#The first thing to do is create a dataframe filtered for non-canceled hotel stays
#Prior to that convert children to a categorical variable
hotels['kids'] = hotels.children + hotels.babies

hotels = (hotels.assign(kids = lambda df: df.kids.map(
                    lambda kids: 'kids' if kids > 0 else 'none')))
```

``` python
#How do the hotel stays of guests with/without children vary throughout the year? 
#Is this different in the city and the resort hotel?

# Map months of the year to numeric values to create arrival date column for exploration
d = {'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12, 'January':1, 'February':2, 'March':3, 'April':4, 'May': 5, 'June':6}
hotels.arrival_date_month = hotels.arrival_date_month.map(d)

#Recreated plot from tidy tuesday 

df = (
      hotels
          .groupby(['hotel', 'arrival_date_month', 'kids'])
          .size()
          .groupby(level=[0,2])
          .apply(lambda x: x/x.sum())
          .reset_index()
      )

df.rename(columns = {0:'count'}, inplace=True)

from plotnine import *

ggplot(df, aes(x = 'arrival_date_month', y = 'count', fill = 'kids')) + \
    geom_col(position='dodge') + facet_wrap(['hotel'], nrow = 2) + \
    ggtitle("Percent Distribution of Guests with and without Kids")
```

    ## <ggplot: (7023981725)>

![](Milestone_files/figure-gfm/unnamed-chunk-24-1.png)<!-- --> Between
both hotels, a majority of guests come without babies or children. Out
of the year however, it is easy to predict when families will arrive,
between both hotels families typically arrived in the summer months.
This could be an opportunity to explore for each hotel.

### How does the average daily rate change with family size?

``` python
kids_condition = (hotels['children'] !=0) | (hotels['babies'] != 0)
kids_per_guest = hotels[['hotel','adr','adults', 'babies', 'children']]
kids_per_guest['kids'] = kids_per_guest['babies'] + kids_per_guest['children']

# Filter out guests with more than 3 kids to eliminate outliers.
```

    ## /Users/mattmerrill/opt/anaconda3/bin/python3:1: SettingWithCopyWarning: 
    ## A value is trying to be set on a copy of a slice from a DataFrame.
    ## Try using .loc[row_indexer,col_indexer] = value instead
    ## 
    ## See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

``` python
kids_per_guest = kids_per_guest[kids_per_guest['kids'] <= 3]
```

``` python
# plot average daily rate based on number of kids (babies and children combined) without adr outliers
g = sns.catplot(x="hotel",
y="adr", data=kids_per_guest, kind="box", hue = "kids", showfliers = False)
plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-26-1.png)<!-- --> - The
average revenue increases on average as family size increases for both
hotels.

The average revenue increases on average as family size increases for
both hotels. After exploring the data, it is clear that my hypothesis
has been confirmed. The use of online travel agencies and cancelations
increased, reaching a peak in the last year of the data with 56% of
guests booking through an OTA and 41% of all guests canceling (up from
28% and 20%, respectively). However, there were some surprising insights
gained as well. Customers who cancel typically have a longer lead time.
Although most cancellations come from customers who book through an OTA,
their average lead time is shorter than for other means of booking. This
means lead time may be another likely factor in determining
cancellations. Guests who booked with at least one baby were more likely
to follow through with the booking and were more likely to require
accommodations. The average daily rate increases with each added child
to the itinerary. Guests with kids represent a small portion of guests
for both hotels, suggesting an untapped market. In conclusion, customers
who book through an online travel agency, and those who book with a
larger lead time are more likely to cancel. It seems that the trends in
customer preference for online booking are directly correlated with an
increase in cancellations. This leads me to believe that the market is
seeking the flexibility and freedoms that come from risk free booking
with OTAs. There are however a number of areas to continue exploring and
confirm through statistical analysis. How confident can I be in this
conclusion? The sample size for families is quite small, could their low
rate of cancellation be due to the small sample size? How confident can
I be in identifying lead time as a major factor in deciding
cancellations?

# Statistical Analysis

Exploratory analysis revealed a number of potential predicting factors
to determine if a guest will cancel. We can see that guests are more
likely to cancel if they are from an OTA or book with a larger lead
time, and guests with babies are far less likely to cancel. We want to
now consider what other variables may contribute to cancellations, such
as ‘agent’, ‘distribution\_channel’, ‘is\_repeated\_guest’, ‘country’,
‘arrival\_date\_month’, ‘kids’ and ‘hotel’. Since this variable is
categorical in nature, my approach will utilize the chi-squared test for
inferential analysis. To perform the chi-squared test, we create a
contingency table of observed values and calculate the expected
frequencies assuming the null hypothesis. Following this we calculate
the chi-squared statistic with the following formula:

\[\chi^2 = \sum_\ \frac{(O - E)^2}{E}\] Once this statistic is
calculated, we compare it to the critical values of the chi-squared
distribution, calculated with the degrees of freedom and the certainty
level (we use the 95% convention). Assuming the null hypothesis is true,
we can reject it if our test statistic is greater than our critical
value. Below is an analysis of each of the identified variables with the
result.

``` python
# define function to output chi-squared test and results
def chi2results(var1, var2, prob):
    observed = pd.crosstab(var1, var2)
    chi_square, p_val, dof, expected = chi2_contingency(observed)
    critical = chi2.ppf(prob, dof)
    
    if abs(chi_square) >= critical:
        result = 'Dependent (reject H0)'
    else:
        result = 'Independent (fail to reject H0)'
    alpha = 1.0 - prob
    
    return chi_square, critical, p_val, dof, alpha, result
```

``` python
observed = pd.crosstab(hotels.is_canceled, hotels.market_segment)
print(observed)
```

    ## market_segment  Aviation  Complementary  ...  Online TA  Undefined
    ## is_canceled                              ...                      
    ## 0                    185            646  ...      35738          0
    ## 1                     52             97  ...      20739          2
    ## 
    ## [2 rows x 8 columns]

``` python
#chi-square statistic - χ2
from scipy.stats import chisquare
from scipy.stats import chi2
from scipy.stats import chi2_contingency
# set columns to test
cols = ['market_segment', 'agent', 'distribution_channel', 'is_repeated_guest', \
        'country', 'arrival_date_month', 'hotel', 'kids']
test_col = 'is_canceled'

# Creating an empty Dataframe with column names only
chi_square_results = pd.DataFrame(columns=['chi_square', 'critical_val', 'p_val', \
                                            'dof', 'alpha', 'result'], index = cols)
for col in cols:
    res = chi2results(hotels[test_col], hotels[col], prob = 0.95)
    chi_square_results.loc[col] = [res[0], res[1], res[2], res[3], res[4], res[5]]

chi_square_results
```

    ##                      chi_square critical_val  ... alpha                 result
    ## market_segment          8497.22      14.0671  ...  0.05  Dependent (reject H0)
    ## agent                   17785.2      376.555  ...  0.05  Dependent (reject H0)
    ## distribution_channel    3745.79      9.48773  ...  0.05  Dependent (reject H0)
    ## is_repeated_guest       857.406      3.84146  ...  0.05  Dependent (reject H0)
    ## country                 15565.2      209.042  ...  0.05  Dependent (reject H0)
    ## arrival_date_month      588.692      19.6751  ...  0.05  Dependent (reject H0)
    ## hotel                   2224.92      3.84146  ...  0.05  Dependent (reject H0)
    ## kids                    19.3888      3.84146  ...  0.05  Dependent (reject H0)
    ## 
    ## [8 rows x 6 columns]

The null hypothesis is rejected for each of the tested columns,
suggesting an association with cancellations for each variable.

# Machine Learning

``` python
hotels.describe(exclude=[np.number]).T
```

    ##                           count unique  ...      first       last
    ## hotel                    119390      2  ...        NaT        NaT
    ## meal                     119390      5  ...        NaT        NaT
    ## country                  119390    178  ...        NaT        NaT
    ## market_segment           119390      8  ...        NaT        NaT
    ## distribution_channel     119390      5  ...        NaT        NaT
    ## reserved_room_type       119390     10  ...        NaT        NaT
    ## assigned_room_type       119390     12  ...        NaT        NaT
    ## deposit_type             119390      3  ...        NaT        NaT
    ## agent                    119390    334  ...        NaT        NaT
    ## company                  119390    353  ...        NaT        NaT
    ## customer_type            119390      4  ...        NaT        NaT
    ## reservation_status       119390      3  ...        NaT        NaT
    ## reservation_status_date  119390    926  ... 2014-10-17 2017-09-14
    ## kids                     119390      2  ...        NaT        NaT
    ## 
    ## [14 rows x 6 columns]

``` python
hotels.describe(include=[np.number]).T
```

    ##                                    count         mean  ...     75%     max
    ## is_canceled                     119390.0     0.370416  ...     1.0     1.0
    ## lead_time                       119390.0   104.011416  ...   160.0   737.0
    ## arrival_date_year               119390.0  2016.156554  ...  2017.0  2017.0
    ## arrival_date_month              119390.0     6.552483  ...     9.0    12.0
    ## arrival_date_week_number        119390.0    27.165173  ...    38.0    53.0
    ## arrival_date_day_of_month       119390.0    15.798241  ...    23.0    31.0
    ## stays_in_weekend_nights         119390.0     0.927599  ...     2.0    19.0
    ## stays_in_week_nights            119390.0     2.500302  ...     3.0    50.0
    ## adults                          119390.0     1.856403  ...     2.0    55.0
    ## children                        119390.0     0.103886  ...     0.0    10.0
    ## babies                          119390.0     0.007949  ...     0.0    10.0
    ## is_repeated_guest               119390.0     0.031912  ...     0.0     1.0
    ## previous_cancellations          119390.0     0.087118  ...     0.0    26.0
    ## previous_bookings_not_canceled  119390.0     0.137097  ...     0.0    72.0
    ## booking_changes                 119390.0     0.221124  ...     0.0    21.0
    ## days_in_waiting_list            119390.0     2.321149  ...     0.0   391.0
    ## adr                             119390.0   101.831122  ...   126.0  5400.0
    ## required_car_parking_spaces     119390.0     0.062518  ...     0.0     8.0
    ## total_of_special_requests       119390.0     0.571363  ...     1.0     5.0
    ## 
    ## [19 rows x 8 columns]

``` python
# load appropriate modules
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

features = ['market_segment', 'agent', 'distribution_channel', 'is_repeated_guest', \
        'country', 'arrival_date_month', 'hotel', 'kids', 'company', 'adr', 'lead_time', 'reserved_room_type', 'total_of_special_requests', 'days_in_waiting_list', 'babies']

X = hotels[features]
        
# Dummy coding
y = hotels['is_canceled']

# Dummy coding for col in cols
cols = ['market_segment', 'distribution_channel', 'hotel', 'kids', 'reserved_room_type']
transformed = []
for col in cols:
  X[col] = pd.get_dummies(X[col])

# One Hot Encoder for country column
```

    ## /Users/mattmerrill/opt/anaconda3/bin/python3:2: SettingWithCopyWarning: 
    ## A value is trying to be set on a copy of a slice from a DataFrame.
    ## Try using .loc[row_indexer,col_indexer] = value instead
    ## 
    ## See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

``` python
data = hotels['country'].to_list()
values = array(data)

# initialize labelencoder, fit and transform
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
transformed.append(integer_encoded)

# Drop country column and replace with encoded 
X.drop('country', axis = 1, inplace = True)
```

    ## /Users/mattmerrill/opt/anaconda3/lib/python3.7/site-packages/pandas/core/frame.py:4102: SettingWithCopyWarning: 
    ## A value is trying to be set on a copy of a slice from a DataFrame
    ## 
    ## See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    ##   errors=errors,

``` python
X['country'] = integer_encoded
```

    ## /Users/mattmerrill/opt/anaconda3/bin/python3:1: SettingWithCopyWarning: 
    ## A value is trying to be set on a copy of a slice from a DataFrame.
    ## Try using .loc[row_indexer,col_indexer] = value instead
    ## 
    ## See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

``` python
print(X.values.shape)
```

    ## (119390, 15)

``` python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#scaler in pipeline object, use logreg algorith
steps = [('scaler', StandardScaler()), \
         ('logreg', LogisticRegression())]
pipeline = Pipeline(steps)

logreg = LogisticRegression()
X_train, X_test, y_train, y_test = \
    train_test_split(X.values, y.values, test_size=0.3, random_state=3)

# fit on training set
logreg_scaled = pipeline.fit(X_train, y_train)
# predict on test set
y_pred = pipeline.predict(X_test)
accuracy_score(y_test, y_pred)
```

    ## 0.738894938157858

``` python
def display_plot(cv_scores, cv_scores_std):
    '''This function will plot the r squared 
    score as well as standard error for each alpha'''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, \
                    cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
```

``` python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 3, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X.values, y.values, cv = 5)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)
```

![](Milestone_files/figure-gfm/unnamed-chunk-35-1.png)<!-- -->

``` python
# Lasso for feature selection in scikit learn
#store feature names
from sklearn.linear_model import Lasso
names = X.columns
lasso = Lasso(alpha=0.001)
# extract coef attribute and store
lasso_coef = lasso.fit(X, y).coef_
_ = plt.figure(figsize=(20,10))
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=45)
_ = plt.ylabel('Coefficients')
plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-36-1.png)<!-- -->

``` python
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import roc_curve

# plotting the ROC curve
y_pred_prob = pipeline.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr, label = 'Logistic Regression')
```

    ## [<matplotlib.lines.Line2D object at 0x1c30618410>]

``` python
plt.xlabel('True Positive Rate')
```

    ## Text(0.5, 0, 'True Positive Rate')

``` python
plt.ylabel('True Positive Rate')
```

    ## Text(0, 0.5, 'True Positive Rate')

``` python
plt.title('Logistic Regression ROC Curve')
```

    ## Text(0.5, 1.0, 'Logistic Regression ROC Curve')

``` python
plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-37-1.png)<!-- -->

``` python
# tuning the modal
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def cv_score(clf, x, y, score_func=accuracy_score):
    result = 0
    nfold = 5
    for train, test in KFold(nfold).split(x): # split data into train/test groups, 5 times
        clf.fit(x[train], y[train]) # fit
        result += score_func(clf.predict(x[test]), y[test]) # evaluate score function on held-out data
    return result / nfold # average
```

``` python
#scaler in pipeline object, use logreg algorith
steps = [('scaler', StandardScaler()), \
         ('logreg', LogisticRegression())]
pipeline = Pipeline(steps)
score = cv_score(pipeline, X_train, y_train)
print(score)
```

    ## 0.7337298767879719

``` python
#the grid of parameters to search over
Cs = [0.001, 0.1, 1, 10, 100]

# create empty dataframe 
df = pd.DataFrame(columns=['Cs', 'cv_score'])

# loop through and add scores    
for c in Cs:
    #scaler in pipeline object, use knn algorith
    steps = [('scaler', StandardScaler()), \
         ('logreg', LogisticRegression(C=c))]
    pipeline = Pipeline(steps)
    score = cv_score(pipeline, X_train, y_train)
    df = df.append({'Cs' : c , 'cv_score' : score} , ignore_index=True)
print(df[df.cv_score == df.cv_score.max()])
```

    ##       Cs  cv_score
    ## 0  0.001   0.73513

``` python
#scaler in pipeline object, use logreg algorith
steps = [('scaler', StandardScaler()), \
         ('logreg', LogisticRegression(C = 0.001))]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
# Print the accuracy from the testing data.
```

    ## Pipeline(memory=None,
    ##          steps=[('scaler',
    ##                  StandardScaler(copy=True, with_mean=True, with_std=True)),
    ##                 ('logreg',
    ##                  LogisticRegression(C=0.001, class_weight=None, dual=False,
    ##                                     fit_intercept=True, intercept_scaling=1,
    ##                                     l1_ratio=None, max_iter=100,
    ##                                     multi_class='auto', n_jobs=None,
    ##                                     penalty='l2', random_state=None,
    ##                                     solver='lbfgs', tol=0.0001, verbose=0,
    ##                                     warm_start=False))],
    ##          verbose=False)

``` python
print(accuracy_score(pipeline.predict(X_test), y_test))
```

    ## 0.7398442080576263

``` python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rfc = RandomForestClassifier(n_estimators=20, max_features='sqrt', random_state=94)
rfc.fit(X_train, y_train)
```

    ## RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
    ##                        criterion='gini', max_depth=None, max_features='sqrt',
    ##                        max_leaf_nodes=None, max_samples=None,
    ##                        min_impurity_decrease=0.0, min_impurity_split=None,
    ##                        min_samples_leaf=1, min_samples_split=2,
    ##                        min_weight_fraction_leaf=0.0, n_estimators=20,
    ##                        n_jobs=None, oob_score=False, random_state=94, verbose=0,
    ##                        warm_start=False)

``` python
y_pred = rfc.predict(X_test)
print(y_pred)
```

    ## [1 1 0 ... 0 1 1]

``` python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

print(confusion_matrix(y_test,y_pred))
```

    ## [[20682  1893]
    ##  [ 3089 10153]]

``` python
print(classification_report(y_test,y_pred))
```

    ##               precision    recall  f1-score   support
    ## 
    ##            0       0.87      0.92      0.89     22575
    ##            1       0.84      0.77      0.80     13242
    ## 
    ##     accuracy                           0.86     35817
    ##    macro avg       0.86      0.84      0.85     35817
    ## weighted avg       0.86      0.86      0.86     35817

``` python
print(accuracy_score(y_test, y_pred))
```

    ## 0.8609040399810146

``` python
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import roc_curve

# plotting the ROC curve
y_pred_prob = rfc.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr, label = 'Random Forest')
```

    ## [<matplotlib.lines.Line2D object at 0x1c32b837d0>]

``` python
plt.xlabel('True Positive Rate')
```

    ## Text(0.5, 0, 'True Positive Rate')

``` python
plt.ylabel('True Positive Rate')
```

    ## Text(0, 0.5, 'True Positive Rate')

``` python
plt.title('Random Forest ROC Curve')
```

    ## Text(0.5, 1.0, 'Random Forest ROC Curve')

``` python
plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-44-1.png)<!-- -->

``` python
from sklearn.model_selection import RandomizedSearchCV
# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 10)]
# number of features at every split
# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)

# create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_depth': max_depth,
 }
 
# Random search of parameters
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, cv = 5, random_state=42)
# Fit the model
rfc_random.fit(X_train, y_train)
# print results
```

    ## RandomizedSearchCV(cv=5, error_score=nan,
    ##                    estimator=RandomForestClassifier(bootstrap=True,
    ##                                                     ccp_alpha=0.0,
    ##                                                     class_weight=None,
    ##                                                     criterion='gini',
    ##                                                     max_depth=None,
    ##                                                     max_features='sqrt',
    ##                                                     max_leaf_nodes=None,
    ##                                                     max_samples=None,
    ##                                                     min_impurity_decrease=0.0,
    ##                                                     min_impurity_split=None,
    ##                                                     min_samples_leaf=1,
    ##                                                     min_samples_split=2,
    ##                                                     min_weight_fraction_leaf=0.0,
    ##                                                     n_estimators=20,
    ##                                                     n_jobs=None,
    ##                                                     oob_score=False,
    ##                                                     random_state=94, verbose=0,
    ##                                                     warm_start=False),
    ##                    iid='deprecated', n_iter=10, n_jobs=None,
    ##                    param_distributions={'max_depth': [100, 140, 180, 220, 260,
    ##                                                       300, 340, 380, 420, 460,
    ##                                                       500, None],
    ##                                         'n_estimators': [20, 40, 60, 80, 100,
    ##                                                          120, 140, 160, 180,
    ##                                                          200]},
    ##                    pre_dispatch='2*n_jobs', random_state=42, refit=True,
    ##                    return_train_score=False, scoring=None, verbose=0)

``` python
print(rfc_random.best_params_)
```

    ## {'n_estimators': 160, 'max_depth': 260}

``` python
rfc = RandomForestClassifier(n_estimators=160, max_depth=260, max_features='sqrt')
rfc.fit(X_train,y_train)
```

    ## RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
    ##                        criterion='gini', max_depth=260, max_features='sqrt',
    ##                        max_leaf_nodes=None, max_samples=None,
    ##                        min_impurity_decrease=0.0, min_impurity_split=None,
    ##                        min_samples_leaf=1, min_samples_split=2,
    ##                        min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                        n_jobs=None, oob_score=False, random_state=None,
    ##                        verbose=0, warm_start=False)

``` python
rfc_predict = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, X.values, y.values, cv=5, scoring='roc_auc')
print("=== Confusion Matrix ===")
```

    ## === Confusion Matrix ===

``` python
print(confusion_matrix(y_test, rfc_predict))
```

    ## [[20689  1886]
    ##  [ 2868 10374]]

``` python
print('\n')
```

``` python
print("=== Classification Report ===")
```

    ## === Classification Report ===

``` python
print(classification_report(y_test, rfc_predict))
```

    ##               precision    recall  f1-score   support
    ## 
    ##            0       0.88      0.92      0.90     22575
    ##            1       0.85      0.78      0.81     13242
    ## 
    ##     accuracy                           0.87     35817
    ##    macro avg       0.86      0.85      0.86     35817
    ## weighted avg       0.87      0.87      0.87     35817

``` python
print('\n')
```

``` python
print("=== All AUC Scores ===")
```

    ## === All AUC Scores ===

``` python
print(rfc_cv_score)
```

    ## [0.68063269 0.64134212 0.5848166  0.5179055  0.70776668]

``` python
print('\n')
```

``` python
print("=== Mean AUC Score ===")
```

    ## === Mean AUC Score ===

``` python
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
```

    ## Mean AUC Score - Random Forest:  0.6264927179962638

``` python
tn, fp, fn, tp = confusion_matrix(y_test, rfc_predict).ravel()
recall = (tp/(tp + fn))
specificity = (tn/(tn + fp))
accuracy = (tp+tn)/(tp+fp+fn+tn)
print(recall, specificity, accuracy)
```

    ## 0.7834164023561395 0.9164562569213732 0.8672697322500489

``` python
from sklearn.ensemble import GradientBoostingClassifier

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=160, learning_rate=learning_rate,
    max_features='sqrt', max_depth=260, random_state=4)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
```

    ## GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
    ##                            learning_rate=0.05, loss='deviance', max_depth=260,
    ##                            max_features='sqrt', max_leaf_nodes=None,
    ##                            min_impurity_decrease=0.0, min_impurity_split=None,
    ##                            min_samples_leaf=1, min_samples_split=2,
    ##                            min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                            n_iter_no_change=None, presort='deprecated',
    ##                            random_state=4, subsample=1.0, tol=0.0001,
    ##                            validation_fraction=0.1, verbose=0,
    ##                            warm_start=False)
    ## Learning rate:  0.05
    ## Accuracy score (training): 0.993
    ## Accuracy score (validation): 0.862
    ## GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
    ##                            learning_rate=0.075, loss='deviance', max_depth=260,
    ##                            max_features='sqrt', max_leaf_nodes=None,
    ##                            min_impurity_decrease=0.0, min_impurity_split=None,
    ##                            min_samples_leaf=1, min_samples_split=2,
    ##                            min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                            n_iter_no_change=None, presort='deprecated',
    ##                            random_state=4, subsample=1.0, tol=0.0001,
    ##                            validation_fraction=0.1, verbose=0,
    ##                            warm_start=False)
    ## Learning rate:  0.075
    ## Accuracy score (training): 0.993
    ## Accuracy score (validation): 0.863
    ## GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
    ##                            learning_rate=0.1, loss='deviance', max_depth=260,
    ##                            max_features='sqrt', max_leaf_nodes=None,
    ##                            min_impurity_decrease=0.0, min_impurity_split=None,
    ##                            min_samples_leaf=1, min_samples_split=2,
    ##                            min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                            n_iter_no_change=None, presort='deprecated',
    ##                            random_state=4, subsample=1.0, tol=0.0001,
    ##                            validation_fraction=0.1, verbose=0,
    ##                            warm_start=False)
    ## Learning rate:  0.1
    ## Accuracy score (training): 0.993
    ## Accuracy score (validation): 0.863
    ## GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
    ##                            learning_rate=0.25, loss='deviance', max_depth=260,
    ##                            max_features='sqrt', max_leaf_nodes=None,
    ##                            min_impurity_decrease=0.0, min_impurity_split=None,
    ##                            min_samples_leaf=1, min_samples_split=2,
    ##                            min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                            n_iter_no_change=None, presort='deprecated',
    ##                            random_state=4, subsample=1.0, tol=0.0001,
    ##                            validation_fraction=0.1, verbose=0,
    ##                            warm_start=False)
    ## Learning rate:  0.25
    ## Accuracy score (training): 0.993
    ## Accuracy score (validation): 0.863
    ## GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
    ##                            learning_rate=0.5, loss='deviance', max_depth=260,
    ##                            max_features='sqrt', max_leaf_nodes=None,
    ##                            min_impurity_decrease=0.0, min_impurity_split=None,
    ##                            min_samples_leaf=1, min_samples_split=2,
    ##                            min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                            n_iter_no_change=None, presort='deprecated',
    ##                            random_state=4, subsample=1.0, tol=0.0001,
    ##                            validation_fraction=0.1, verbose=0,
    ##                            warm_start=False)
    ## Learning rate:  0.5
    ## Accuracy score (training): 0.993
    ## Accuracy score (validation): 0.861
    ## GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
    ##                            learning_rate=0.75, loss='deviance', max_depth=260,
    ##                            max_features='sqrt', max_leaf_nodes=None,
    ##                            min_impurity_decrease=0.0, min_impurity_split=None,
    ##                            min_samples_leaf=1, min_samples_split=2,
    ##                            min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                            n_iter_no_change=None, presort='deprecated',
    ##                            random_state=4, subsample=1.0, tol=0.0001,
    ##                            validation_fraction=0.1, verbose=0,
    ##                            warm_start=False)
    ## Learning rate:  0.75
    ## Accuracy score (training): 0.993
    ## Accuracy score (validation): 0.860
    ## GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
    ##                            learning_rate=1, loss='deviance', max_depth=260,
    ##                            max_features='sqrt', max_leaf_nodes=None,
    ##                            min_impurity_decrease=0.0, min_impurity_split=None,
    ##                            min_samples_leaf=1, min_samples_split=2,
    ##                            min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                            n_iter_no_change=None, presort='deprecated',
    ##                            random_state=4, subsample=1.0, tol=0.0001,
    ##                            validation_fraction=0.1, verbose=0,
    ##                            warm_start=False)
    ## Learning rate:  1
    ## Accuracy score (training): 0.993
    ## Accuracy score (validation): 0.861
    ## GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
    ##                            learning_rate=1.25, loss='deviance', max_depth=260,
    ##                            max_features='sqrt', max_leaf_nodes=None,
    ##                            min_impurity_decrease=0.0, min_impurity_split=None,
    ##                            min_samples_leaf=1, min_samples_split=2,
    ##                            min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                            n_iter_no_change=None, presort='deprecated',
    ##                            random_state=4, subsample=1.0, tol=0.0001,
    ##                            validation_fraction=0.1, verbose=0,
    ##                            warm_start=False)
    ## Learning rate:  1.25
    ## Accuracy score (training): 0.993
    ## Accuracy score (validation): 0.859
    ## GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
    ##                            learning_rate=1.5, loss='deviance', max_depth=260,
    ##                            max_features='sqrt', max_leaf_nodes=None,
    ##                            min_impurity_decrease=0.0, min_impurity_split=None,
    ##                            min_samples_leaf=1, min_samples_split=2,
    ##                            min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                            n_iter_no_change=None, presort='deprecated',
    ##                            random_state=4, subsample=1.0, tol=0.0001,
    ##                            validation_fraction=0.1, verbose=0,
    ##                            warm_start=False)
    ## Learning rate:  1.5
    ## Accuracy score (training): 0.993
    ## Accuracy score (validation): 0.858
    ## GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
    ##                            learning_rate=1.75, loss='deviance', max_depth=260,
    ##                            max_features='sqrt', max_leaf_nodes=None,
    ##                            min_impurity_decrease=0.0, min_impurity_split=None,
    ##                            min_samples_leaf=1, min_samples_split=2,
    ##                            min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                            n_iter_no_change=None, presort='deprecated',
    ##                            random_state=4, subsample=1.0, tol=0.0001,
    ##                            validation_fraction=0.1, verbose=0,
    ##                            warm_start=False)
    ## Learning rate:  1.75
    ## Accuracy score (training): 0.992
    ## Accuracy score (validation): 0.856
    ## GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
    ##                            learning_rate=2, loss='deviance', max_depth=260,
    ##                            max_features='sqrt', max_leaf_nodes=None,
    ##                            min_impurity_decrease=0.0, min_impurity_split=None,
    ##                            min_samples_leaf=1, min_samples_split=2,
    ##                            min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                            n_iter_no_change=None, presort='deprecated',
    ##                            random_state=4, subsample=1.0, tol=0.0001,
    ##                            validation_fraction=0.1, verbose=0,
    ##                            warm_start=False)
    ## Learning rate:  2
    ## Accuracy score (training): 0.989
    ## Accuracy score (validation): 0.846

``` python
gb_clf2 = GradientBoostingClassifier(n_estimators=160, learning_rate=0.1, max_features='sqrt', max_depth=260, random_state=4)
gb_clf2.fit(X_train, y_train)
```

    ## GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
    ##                            learning_rate=0.1, loss='deviance', max_depth=260,
    ##                            max_features='sqrt', max_leaf_nodes=None,
    ##                            min_impurity_decrease=0.0, min_impurity_split=None,
    ##                            min_samples_leaf=1, min_samples_split=2,
    ##                            min_weight_fraction_leaf=0.0, n_estimators=160,
    ##                            n_iter_no_change=None, presort='deprecated',
    ##                            random_state=4, subsample=1.0, tol=0.0001,
    ##                            validation_fraction=0.1, verbose=0,
    ##                            warm_start=False)

``` python
predictions = gb_clf2.predict(X_test)

print("Confusion Matrix:")
```

    ## Confusion Matrix:

``` python
print(confusion_matrix(y_test, predictions))
```

    ## [[20544  2031]
    ##  [ 2879 10363]]

``` python
print("Classification Report")
```

    ## Classification Report

``` python
print(classification_report(y_test, predictions))
```

    ##               precision    recall  f1-score   support
    ## 
    ##            0       0.88      0.91      0.89     22575
    ##            1       0.84      0.78      0.81     13242
    ## 
    ##     accuracy                           0.86     35817
    ##    macro avg       0.86      0.85      0.85     35817
    ## weighted avg       0.86      0.86      0.86     35817

``` python
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
recall = (tp/(tp + fn))
specificity = (tn/(tn + fp))
accuracy = (tp+tn)/(tp+fp+fn+tn)
print(recall, specificity, accuracy)
```

    ## 0.7825857121280774 0.9100332225913621 0.8629142585922885

1.  Hertzfeld, Esther. Study: Cancellation Rate at 40% as OTAs Push Free
    Change Policy. Hotel Management, 23 Apr. 2019,
    www.hotelmanagement.net/tech/study-cancelation-rate-at-40-as-otas-push-free-change-policy.

2.  Funnell, Rob. “The Real Cost of ‘Free’ Cancellations for Hotels.”
    Triptease, Triptease - Attract. Convert. Compete., 13 May 2019,
    www.triptease.com/blog/the-real-cost-of-free-cancellations/?utm\_source=MediaPartner\&utm\_medium=HotelSpeak.
