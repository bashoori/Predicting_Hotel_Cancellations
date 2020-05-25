Milestone Report: Predicting Hotel Cancellations
================
Matthew Merrill

## Project Proposal

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

## Data Wrangling and Cleaning Steps

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

### Read in the data

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import os

# load data
hotels = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv')

print(hotels.head())
```

    ##           hotel  is_canceled  ...  reservation_status  reservation_status_date
    ## 0  Resort Hotel            0  ...           Check-Out               2015-07-01
    ## 1  Resort Hotel            0  ...           Check-Out               2015-07-01
    ## 2  Resort Hotel            0  ...           Check-Out               2015-07-02
    ## 3  Resort Hotel            0  ...           Check-Out               2015-07-02
    ## 4  Resort Hotel            0  ...           Check-Out               2015-07-03
    ## 
    ## [5 rows x 32 columns]

### Cleaning steps

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
# drop undefined values
hotels = hotels[hotels.market_segment != 'Undefined']
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

## Exploratory Data Analysis:

Before beginning data analysis, it was hypothesised that over time,
customer bookings from online travel agencies increase along with the
portion of cancellations from these bookings. I followed up with asking
a number of questions to gain insight into other variables and factors
that may contribute to feature selection.

``` python
# Inspect categorical columns
hotels.describe(exclude=[np.number]).T
```

    ##                           count unique  ...      first       last
    ## hotel                    119388      2  ...        NaT        NaT
    ## arrival_date_month       119388     12  ...        NaT        NaT
    ## meal                     119388      5  ...        NaT        NaT
    ## country                  119388    178  ...        NaT        NaT
    ## market_segment           119388      7  ...        NaT        NaT
    ## distribution_channel     119388      5  ...        NaT        NaT
    ## reserved_room_type       119388     10  ...        NaT        NaT
    ## assigned_room_type       119388     12  ...        NaT        NaT
    ## deposit_type             119388      3  ...        NaT        NaT
    ## agent                    119388    334  ...        NaT        NaT
    ## company                  119388    353  ...        NaT        NaT
    ## customer_type            119388      4  ...        NaT        NaT
    ## reservation_status       119388      3  ...        NaT        NaT
    ## reservation_status_date  119388    926  ... 2014-10-17 2017-09-14
    ## 
    ## [14 rows x 6 columns]

``` python
# Inspect numerical columns
hotels.describe(include=[np.number]).T
```

    ##                                    count         mean  ...     75%     max
    ## is_canceled                     119388.0     0.370406  ...     1.0     1.0
    ## lead_time                       119388.0   104.013134  ...   160.0   737.0
    ## arrival_date_year               119388.0  2016.156574  ...  2017.0  2017.0
    ## arrival_date_week_number        119388.0    27.165092  ...    38.0    53.0
    ## arrival_date_day_of_month       119388.0    15.798439  ...    23.0    31.0
    ## stays_in_weekend_nights         119388.0     0.927606  ...     2.0    19.0
    ## stays_in_week_nights            119388.0     2.500327  ...     3.0    50.0
    ## adults                          119388.0     1.856393  ...     2.0    55.0
    ## children                        119388.0     0.103888  ...     0.0    10.0
    ## babies                          119388.0     0.007949  ...     0.0    10.0
    ## is_repeated_guest               119388.0     0.031913  ...     0.0     1.0
    ## previous_cancellations          119388.0     0.087119  ...     0.0    26.0
    ## previous_bookings_not_canceled  119388.0     0.137099  ...     0.0    72.0
    ## booking_changes                 119388.0     0.221128  ...     0.0    21.0
    ## days_in_waiting_list            119388.0     2.321188  ...     0.0   391.0
    ## adr                             119388.0   101.832576  ...   126.0  5400.0
    ## required_car_parking_spaces     119388.0     0.062519  ...     0.0     8.0
    ## total_of_special_requests       119388.0     0.571347  ...     1.0     5.0
    ## 
    ## [18 rows x 8 columns]

#### Initial findings of numeric columns:

  - About 25% of bookings from 2015 to 2017 were cancelled
  - 50% of bookings occurred between the 16th and 38th week of the year.
  - The median lead time is 69 with an IQR of 142
  - Guests typically do not spend any time on a waiting list, but the
    max was 391 days.
  - The average daily rate (ADR) was about 94.5 with a max of 5400.

#### Initial findings of categorical columns:

  - August is the most popular month between both hotels.
  - Most people who visit are from Portugal (the country where both
    hotels are located).
  - Most bookings are made with an online travel agent.
  - Most bookings are made with travel agent ‘9.0’.
  - Most people do not go through a
company.

### What is the most common means of booking? Through which booking channel do most cancellations occur?

``` python
# Canceled bookings by market segment
canceled_pct = round(100*hotels[hotels.is_canceled == 1].market_segment\
                                                        .value_counts(normalize = True, sort = False)\
                                                        .sort_index(), 1)
# Percent of bookings from each market segment
market_segment_pct = round(100*hotels.market_segment\
                                     .value_counts(normalize = True, sort = False)\
                                     .sort_index(), 1)

# set axis labels for market_segment options
labels = list(canceled_pct.index)
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6)) 
market_segment_canceled = ax.bar(x + width/2, canceled_pct, width, label = 'Cancelation Distribution')
market_segment_not_canceled = ax.bar(x - width/2, market_segment_pct, width, label = 'Market Distribution')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percent of canceled/not canceled bookings')
ax.set_title('Customer Channels')
ax.set_xticks(x)
```

    ## [<matplotlib.axis.XTick object at 0x1a24c4db10>, <matplotlib.axis.XTick object at 0x1a24c4d190>, <matplotlib.axis.XTick object at 0x1a24e70a90>, <matplotlib.axis.XTick object at 0x1a24ccbe90>, <matplotlib.axis.XTick object at 0x1a24cd6510>, <matplotlib.axis.XTick object at 0x1a24cd6a90>, <matplotlib.axis.XTick object at 0x1a24ce0190>]

``` python
ax.set_xticklabels(labels)
```

    ## [Text(0, 0, 'Aviation'), Text(0, 0, 'Complementary'), Text(0, 0, 'Corporate'), Text(0, 0, 'Direct'), Text(0, 0, 'Groups'), Text(0, 0, 'Offline TA/TO'), Text(0, 0, 'Online TA')]

``` python
ax.legend(fontsize = 14)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(market_segment_canceled)
autolabel(market_segment_not_canceled)

fig.tight_layout()

plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

> The most common means of booking is through an online TA, with over
> 47% of customers going through this channel. Online TA’s do account
> for almost half of cancellations over the three year period the data
> accounts for. Group bookings also stand out for booking cancellations,
> with only 17% of bookings originating from groups while accounting for
> almost 30% of overall cancellations. Direct bookings stand out for
> their relatively low cancellation rate. They contribute only 4.4% of
> cancellations, this being about a 15% cancellation rate among direct
> bookings. This compares starkly to online TA bookings, where 37% of
> those bookings are
canceled.

### How have customer channels changed over time?

``` python
# Isolate data by year, eliminating Aviation for continuity between years (Aviation wasn't offered until 2016)
hotels_2015 = hotels[(hotels.arrival_date_year  == 2015) & (hotels.market_segment != 'Aviation')]
hotels_2016 = hotels[(hotels.arrival_date_year  == 2016) & (hotels.market_segment != 'Aviation')]
hotels_2017 = hotels[(hotels.arrival_date_year  == 2017) & (hotels.market_segment != 'Aviation')]
```

``` python

# Market segment breakdown by year
ms_2015 = round(100*hotels_2015.market_segment.value_counts(normalize = True, sort=False).sort_index(), 1)
ms_2016 = round(100*hotels_2016.market_segment.value_counts(normalize = True, sort=False).sort_index(), 1)
ms_2017 = round(100*hotels_2017.market_segment.value_counts(normalize = True, sort=False).sort_index(), 1)

# plot hist per week for arrival_date_week_number
labels = list(ms_2015.index)
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 5))
year_2015 = ax.bar(x - width, ms_2015, width, label='2015')
year_2016 = ax.bar(x, ms_2016, width, label='2016')
year_2017 = ax.bar(x + width, ms_2017, width, label='2017')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Bookings')
ax.set_title('Customer Channel Distribution over time')
ax.set_xticks(x)
```

    ## [<matplotlib.axis.XTick object at 0x1a24e86ed0>, <matplotlib.axis.XTick object at 0x1a24d225d0>, <matplotlib.axis.XTick object at 0x1a24ebb810>, <matplotlib.axis.XTick object at 0x1a27517b90>, <matplotlib.axis.XTick object at 0x1a27517e10>, <matplotlib.axis.XTick object at 0x1a2751f610>]

``` python
ax.set_xticklabels(labels)
```

    ## [Text(0, 0, 'Complementary'), Text(0, 0, 'Corporate'), Text(0, 0, 'Direct'), Text(0, 0, 'Groups'), Text(0, 0, 'Offline TA/TO'), Text(0, 0, 'Online TA')]

``` python
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(year_2015)
autolabel(year_2016)
autolabel(year_2017)


fig.tight_layout()

plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

> In the 2015 group, offline and online TA’s made up a majority of
> bookings, with a little over a quarter of bookings distributed to each
> in that year. By 2016, almost half of bookings occured online, with a
> little over a third contributed by groups and offline TA’s. By 2017,
> online TA’s gained a clear advantage over the travel industry between
> these hotels, making up about 56% of bookings. Direct, corporate and
> complimentary bookings saw little to no change over the same time
> period.

### How has the rate of cancellations changed over time?

``` python
# Cancellations by year based on market segment
canceled = hotels[(hotels.market_segment != 'Aviation')
                          & (hotels.is_canceled == 1)]
canceled_2015 = round(100*canceled[(canceled.arrival_date_year  == 2015)]\
                                .market_segment\
                                .value_counts(normalize = True, sort=False)\
                                .sort_index(), 1)

canceled_2016 = round(100*canceled[(canceled.arrival_date_year  == 2016)]\
                                .market_segment\
                                .value_counts(normalize = True, sort=False)\
                                .sort_index(), 1)

canceled_2017 = round(100*canceled[(canceled.arrival_date_year  == 2017)]\
                                .market_segment\
                                .value_counts(normalize = True, sort=False)\
                                .sort_index(), 1)
```

``` python
# plot hist per week for arrival_date_week_number
labels = list(canceled_2015.index)
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 5))
year_2015 = ax.bar(x - width, canceled_2015, width, label='2015')
year_2016 = ax.bar(x, canceled_2016, width, label='2016')
year_2017 = ax.bar(x + width, canceled_2017, width, label='2017')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Bookings')
ax.set_title('Cancelation Distribution over time')
ax.set_xticks(x)
```

    ## [<matplotlib.axis.XTick object at 0x1a276a36d0>, <matplotlib.axis.XTick object at 0x1a27699d90>, <matplotlib.axis.XTick object at 0x1a27699c50>, <matplotlib.axis.XTick object at 0x1a276f7bd0>, <matplotlib.axis.XTick object at 0x1a291a1050>, <matplotlib.axis.XTick object at 0x1a291a1750>]

``` python
ax.set_xticklabels(labels)
```

    ## [Text(0, 0, 'Complementary'), Text(0, 0, 'Corporate'), Text(0, 0, 'Direct'), Text(0, 0, 'Groups'), Text(0, 0, 'Offline TA/TO'), Text(0, 0, 'Online TA')]

``` python
ax.legend(fontsize=14)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(year_2015)
autolabel(year_2016)
autolabel(year_2017)


fig.tight_layout()

plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

> Groups were the major contributors to cancellations in 2015, with 47%
> of cancellations originating from this channel. However there was a
> clear drop in group cancellations from 2015 to 2016, with a sharp
> increase from online TA’s in the same time period. By 2017, more than
> half of cancellations took place online.

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

``` python
# list of data by year
data = [hotels_2015, hotels_2016, hotels_2017]
loc = ['City Hotel', 'Resort Hotel']
pct = []

# bookings by hotel for each year
for hotel in data:
    ms = hotel.groupby(['hotel', 'market_segment']).agg({'market_segment': 'count'})
    ms_pct = ms.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    for place in loc:
      pct.append(round(ms_pct.loc[place].loc['Online TA'].values[0], 1))
```

Percentage of bookings from OTAs for city hotel: - % in 2015. - 37.3% in
2016. - 42.8% in 2017. Percentage of bookings from OTAs for resort
hotel: - 22.4% in 2015. - 51.8% in 2016. - 58.3% in 2017.

In 2015, the city hotel received 22% of it’s bookings through OTAs while
the resort hotel received 37%. By 2017, city hotel recieved 58% of its
bookings through an online TA with the resort hotel increasing their
portion to 51%. It is clear that both increased significantly over time
with the city hotel taking a major leap from 2015 to 2016, increasing
from 22% to 52%.

### Does lead time correlate to cancellations?

``` python
# canceled and not canceled bookings
canceled = hotels[hotels.is_canceled == True]
not_canceled = hotels[hotels.is_canceled == False]

# outlier labels
red_point = dict(markerfacecolor='r', marker='p')

# boxplot of lead time for canceled and not canceled bookings
fig, ax = plt.subplots(2, figsize=(10, 4), sharex=True, sharey=True, gridspec_kw={'hspace': 0})
ax[0].boxplot(canceled.lead_time, vert=False, flierprops=red_point)
```

    ## {'whiskers': [<matplotlib.lines.Line2D object at 0x1a290a9710>, <matplotlib.lines.Line2D object at 0x1a290b5b50>], 'caps': [<matplotlib.lines.Line2D object at 0x1a291b1c50>, <matplotlib.lines.Line2D object at 0x1a290be590>], 'boxes': [<matplotlib.lines.Line2D object at 0x1a290a96d0>], 'medians': [<matplotlib.lines.Line2D object at 0x1a290bead0>], 'fliers': [<matplotlib.lines.Line2D object at 0x1a290befd0>], 'means': []}

``` python
ax[1].boxplot(not_canceled.lead_time, vert=False, flierprops=red_point)
```

    ## {'whiskers': [<matplotlib.lines.Line2D object at 0x1a290a9b90>, <matplotlib.lines.Line2D object at 0x1a290a97d0>], 'caps': [<matplotlib.lines.Line2D object at 0x1a290d05d0>, <matplotlib.lines.Line2D object at 0x1a290d0ad0>], 'boxes': [<matplotlib.lines.Line2D object at 0x1a290a9f90>], 'medians': [<matplotlib.lines.Line2D object at 0x1a290c45d0>], 'fliers': [<matplotlib.lines.Line2D object at 0x1a29146550>], 'means': []}

``` python
ax[0].set_title('Canceled')
ax[1].set_title('Not Canceled')
ax[1].set(xlabel='Lead Time')
fig.tight_layout()
plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

``` python
# median lead time for canceled and not canceled bookings
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
# define lead time for ota's and non-ota
ta_lead_time = hotels[hotels.market_segment == 'Online TA']['lead_time']
not_ta_lead_time = hotels[hotels.market_segment != 'Online TA']['lead_time']

# plot boxplot of lead time for canceled and non-canceled guests
fig, ax = plt.subplots(2, figsize=(10, 4), sharex=True, sharey=True, gridspec_kw={'hspace': 0})
ax[0].boxplot(ta_lead_time, vert=False, flierprops=red_point)
```

    ## {'whiskers': [<matplotlib.lines.Line2D object at 0x1a29170e10>, <matplotlib.lines.Line2D object at 0x1a277ca910>], 'caps': [<matplotlib.lines.Line2D object at 0x1a277cae10>, <matplotlib.lines.Line2D object at 0x1a27545c50>], 'boxes': [<matplotlib.lines.Line2D object at 0x1a277c4550>], 'medians': [<matplotlib.lines.Line2D object at 0x1a277d2890>], 'fliers': [<matplotlib.lines.Line2D object at 0x1a277d2d90>], 'means': []}

``` python
ax[1].boxplot(not_ta_lead_time, vert=False, flierprops=red_point)
```

    ## {'whiskers': [<matplotlib.lines.Line2D object at 0x1a277c4c50>, <matplotlib.lines.Line2D object at 0x1a277dad90>], 'caps': [<matplotlib.lines.Line2D object at 0x1a277d2e10>, <matplotlib.lines.Line2D object at 0x1a2966a7d0>], 'boxes': [<matplotlib.lines.Line2D object at 0x1a277c4f90>], 'medians': [<matplotlib.lines.Line2D object at 0x1a2966ad10>], 'fliers': [<matplotlib.lines.Line2D object at 0x1a277dae10>], 'means': []}

``` python
ax[0].set_title('OTA Lead Time')
ax[1].set_title("Non-OTA Lead Time")
ax[1].set(xlabel = 'Lead Time')
fig.tight_layout()
plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

``` python
median_lead_ota = np.median(hotels.lead_time)
median_lead_not_ota = np.median(hotels.market_segment != 'Online TA')
```

  - The median lead time for OTA bookings was 113
  - The median lead time for all other methods of booking was 113

Online travel agencies do not have a higher lead time on average, there
may be another area here to explore. Lead time and bookings from OTA’s
seem to correlate to higher rates of
cancellation.

### What trends are there in the distribution of company and travel agent bookings?

``` python
fig, ax = plt.subplots(figsize=(12, 6))
# top 6 companies
top_n = hotels.company.value_counts().index[:6]
# assign any company falling outside of the top 6 to "other"
(hotels.assign(company=hotels.company.where(hotels.company.isin(top_n), "Other" ))
.company.value_counts().plot.bar(ax=ax))
ax.set_xlabel('Companies')
ax.set_ylabel('Count')
```

![](Milestone_files/figure-gfm/unnamed-chunk-22-1.png)<!-- --> \> There
are over 300 different companies but a vast majority of visitors do not
go through them for booking. The ‘0’ company represents guests who did
not go through a company.

``` python
fig, ax = plt.subplots(figsize=(10, 6))
# top 20 agents
top_n = hotels.agent.value_counts().index[:20]
# assign agents outside of top 20 to 'other'
(hotels.assign(agent=hotels.agent.where(hotels.agent.isin(top_n), "Other" ))
.agent.value_counts().plot.bar(ax=ax))
ax.set_xlabel('Agents')
ax.set_ylabel('Count')
```

![](Milestone_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

> The story seems to be different for travel agents, a majority use this
> channel with most going through agent ‘9.0’.

### How are cancellations distributed throughout the year?

``` python
# mask for accessing data in plot function
mask = hotels.hotel.isin(['City Hotel', 'Resort Hotel'])
g = sns.catplot(x="hotel",\
                y="arrival_date_week_number", 
                data=hotels[mask], 
                kind="box", 
                hue = "is_canceled")
plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

> Resort Hotel shows a higher concentration of cancelations during the
> summer months, while the City Hotel shows a closely matching
> correlation between bookings and canceled bookings, both uniformly
> distributed throughout the
year.

### Are guests with kids more likely to cancel?

``` python
# Calculate number of guests with babies who canceled and who did not cancel
babies = hotels.loc[(hotels.babies != 0)]['babies'].count()
not_canceled_babies = hotels.loc[(hotels.is_canceled == 0) & (hotels.babies != 0)]['babies'].count()

# Do the same calculation for families with children and with no children
children = hotels.loc[(hotels.children != 0)]['children'].count()
not_canceled_children = hotels.loc[(hotels.is_canceled == 0) & (hotels.children != 0)]['children'].count()

# No children
conditions = (hotels.children == 0) & (hotels.babies == 0)
no_kids = hotels.loc[conditions]['adults'].count()
not_canceled_no_kids = hotels.loc[(hotels.is_canceled == 0) & (conditions)]['adults'].count()
```

``` python
# percentage of families who did not cancel
perc_babies = str(round(not_canceled_babies/babies, 2)*100) + '%'
perc_children = str(round(not_canceled_children/children, 2)*100) + '%'
perc_no_kids = str(round(not_canceled_no_kids/no_kids, 2)*100) + '%'
```

``` python
# set data frame with results if kids breakdown

d1 = {'guest_type': ['babies', 'children', 'without_children'], \
     'not_cancled': [not_canceled_babies, not_canceled_children, not_canceled_no_kids],\
    'percent': [perc_babies, perc_children, perc_no_kids]}

d = {'guest_type': ['Babies', 'Children', 'No Kids'], \
    'percent': [perc_babies, perc_children, perc_no_kids]}

pie_data = pd.DataFrame(data=d)
```

``` python
import re

def donut_plot(data, plotnumber):
    ''' Create donut plot with data and plot number for multiple plots'''
    startingRadius = 0.7 + (0.3* (len(data)-1))
    for index, row in data.iterrows():
        scenario = row["guest_type"]
        percentage = row["percent"]
        textLabel = scenario
        
        percentage = int(re.search(r'\d+', percentage).group())
        remainingPie = 100 - percentage
        
        donut_sizes = [remainingPie, percentage]
        
        plt.text(0.01, startingRadius - 0.25, textLabel, ha='right', va='bottom', fontsize = 12, 
        fontweight = 'bold')
        
        plt.pie(donut_sizes, radius=startingRadius, startangle=0, colors=['lightgray', 'tomato'],
                wedgeprops={"edgecolor": "white", 'linewidth': 1.5})
        
        startingRadius-=0.3

    # equal ensures pie chart is drawn as a circle (equal aspect ratio)
    plt.axis('equal')

    # create circle and place onto pie chart
    circle = plt.Circle(xy=(0, 0), radius=0.35, facecolor='white')
    plt.gca().add_artist(circle)
    plt.show()
```

``` python
# plot the proportion of cancellations based on whether guests had babies, children or none.
plt.title('82% of guests with babies \n did not cancel \n', fontsize = '18', fontweight = 'bold')
donut_plot(pie_data, '1')
```

![](Milestone_files/figure-gfm/unnamed-chunk-29-1.png)<!-- -->

Guests with babies followed through with their booking 83% of the time
and 86% of guests who stayed with babies had at least one special
request.

### How do the hotel stays of guests with/without children vary throughout the year? Is this different between hotels?

``` python
#The first thing to do is create a dataframe filtered for non-canceled hotel stays
#Prior to that convert children to a categorical variable
hotels['kids'] = hotels.children + hotels.babies

hotels = (hotels.assign(kids = lambda df: df.kids.map(
                    lambda kids: 'kids' if kids > 0 else 'none')))
```

``` python
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

    ## <ggplot: (7018379205)>

![](Milestone_files/figure-gfm/unnamed-chunk-31-1.png)<!-- -->

Between both hotels, a majority of guests come without babies or
children. Out of the year however, it is easy to predict when families
will arrive, between both hotels families typically arrived in the
summer months. This could be an opportunity to explore for each hotel.

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

![](Milestone_files/figure-gfm/unnamed-chunk-33-1.png)<!-- -->

> The average revenue increases on average as family size increases for
> both hotels.

After exploring the data, it is clear that my hypothesis has been
confirmed. The use of online travel agencies and cancelations increased,
reaching a peak in the last year of the data with 56% of guests booking
through an OTA and 41% of all guests canceling (up from 28% and 20%,
respectively). However, there were some surprising insights gained as
well.

  - Customers who cancel typically have a longer lead time. Although
    most cancellations come from customers who book through an OTA,
    their average lead time is shorter than for other means of booking.
    This means lead time may be another likely factor in determining
    cancellations.
  - Guests who booked with at least one baby were more likely to follow
    through with the booking and were more likely to require
    accommodations.
  - The average daily rate increases with each added child to the
    itinerary.
  - Guests with kids represent a small portion of guests for both
    hotels, suggesting an untapped market.

In conclusion, customers who book through an online travel agency, and
those who book with a larger lead time are more likely to cancel. It
seems that the trends in customer preference for online booking are
directly correlated with an increase in cancellations. This leads me to
believe that the market is seeking the flexibility and freedoms that
come from risk free booking with OTAs. There are however a number of
areas to continue exploring and confirm through statistical analysis.
How confident can I be in this conclusion? The sample size for families
is quite small, could their low rate of cancellation be due to the small
sample size? How confident can I be in identifying lead time as a major
factor in deciding cancellations?

## Statistical Analysis

Exploratory analysis revealed a number of potential predicting factors
to determine if a guest will cancel. We can see that guests are more
likely to cancel if they are from an OTA or book with a larger lead
time, and guests with babies are far less likely to cancel. We want to
now consider what other variables may contribute to cancellations, such
as ‘agent’, ‘distribution\_channel’, ‘is\_repeated\_guest’, ‘country’,
‘arrival\_date\_month’, ‘kids’ and ‘hotel’. Since these variables are
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
# import appropriate modules
from scipy.stats import chisquare
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from scipy.stats import norm
from scipy.stats import t
from numpy.random import seed
from scipy import stats
```

### Analyze Categorical Variables

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
# two way table of market segment vs cancellations
observed = pd.crosstab(hotels.is_canceled, hotels.market_segment)
print(observed)
```

    ## market_segment  Aviation  Complementary  ...  Offline TA/TO  Online TA
    ## is_canceled                              ...                          
    ## 0                    185            646  ...          15908      35738
    ## 1                     52             97  ...           8311      20739
    ## 
    ## [2 rows x 7 columns]

``` python
# set columns to test
cat_cols = list(hotels.describe(exclude=[np.number]).columns.drop(['reservation_status', 'reservation_status_date', 'deposit_type', 'assigned_room_type', 'meal', 'country']))
test_col = 'is_canceled'

# Creating an empty Dataframe with column names only
chi_square_results = pd.DataFrame(columns=['chi_square', 'critical_val', 'p_val', \
                                            'dof', 'alpha', 'result'], index = cat_cols)
for col in cat_cols:
    res = chi2results(hotels[test_col], hotels[col], prob = 0.95)
    chi_square_results.loc[col] = [res[0], res[1], res[2], res[3], res[4], res[5]]

print(chi_square_results[['p_val', 'result']])
```

    ##                              p_val                 result
    ## hotel                            0  Dependent (reject H0)
    ## market_segment                   0  Dependent (reject H0)
    ## distribution_channel             0  Dependent (reject H0)
    ## reserved_room_type    8.87222e-134  Dependent (reject H0)
    ## agent                            0  Dependent (reject H0)
    ## company               1.80972e-297  Dependent (reject H0)
    ## customer_type                    0  Dependent (reject H0)
    ## kids                   1.07702e-05  Dependent (reject H0)

The null hypothesis is rejected for each of the tested columns,
suggesting an association with cancellations for each variable.

### Analyze Numerical Variables

To perform statistical analysis on numerical predictors, we calculate
the t-statistic and determine the corresponding p-value. If the p-value
is less than the 5% threshold for our 95% confidence interval we can
consider this variable statistically significant in determining
cancellations.

``` python
mean = np.mean(hotels['adr'])
std = np.std(hotels['adr'], ddof = 1)
_ = plt.hist(hotels['adr'], bins = 40, range = (0, 500))
_ = plt.xlabel('Average Daily Rate (adr)')
_ = plt.ylabel('frequency')
_ = plt.axvline(mean, color='r')
_ = plt.axvline(mean+std, color='r', linestyle='--')
_ = plt.axvline(mean-std, color='r', linestyle='--')
_ = plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-38-1.png)<!-- -->

``` python
print('mean charges ' + str(round(mean, 2)))
```

    ## mean charges 101.83

``` python
print('standard deviation ' + str(round(std, 2)))
```

    ## standard deviation 50.53

The null hypothesis for the average daily rate is that it has no
influence on whether or not a booking is likely to be canceled.

``` python
# Define samples
sample0 = hotels[hotels.is_canceled == 1]['adr']
sample1 = hotels[hotels.is_canceled == 0]['adr']
n_0 = len(sample0)
n_1 = len(sample1)
# Define mean and standard deviation for each group
x_0 = np.mean(sample0)
x_1 = np.mean(sample1)
s_0 = np.std(sample0, ddof = 1)
s_1 = np.std(sample1, ddof = 1)
# degrees of freedom
df = n_0 + n_1 - 2
```

``` python
# calculate t-statistic and p-value
t, pval = stats.ttest_ind_from_stats(x_0, s_0, n_0, x_1, s_1, n_1)

print(t, pval)
```

    ## 16.464155106230862 7.749454084810186e-61

The results reject the null hypothesis. The confidence interval is far
greater than the accepted 95% needed to accept this outcome. Thus we can
conclude that the average daily rate is important in determining if a
booking will be canceled. Next we will determine the significance of
each of the other continuous variables and enter the results in a data
frame to be analyzed. The following code summarizes the findings.

``` python
def significance(col):
  ''' this function will return the t-stat and pval of a feature of interest '''
  # Define samples
  sample0 = hotels[hotels.is_canceled == 1][col]
  sample1 = hotels[hotels.is_canceled == 0][col]
  n_0 = len(sample0)
  n_1 = len(sample1)
  # Define mean and standard deviation for each group
  x_0 = np.mean(sample0)
  x_1 = np.mean(sample1)
  s_0 = np.std(sample0, ddof = 1)
  s_1 = np.std(sample1, ddof = 1)
  # degrees of freedom
  df = n_0 + n_1 - 2
  
  t, pval = stats.ttest_ind_from_stats(x_0, s_0, n_0, x_1, s_1, n_1)
  
  if pval <= 0.05:
      result = 'Dependent (reject H0)'
  else:
      result = 'Ind (fail to reject H0)'
    
  return t, pval, result
```

``` python
# List of numerical excluding features with potential for data leakage. 
num_cols = list(hotels.describe(include=[np.number]).columns.drop(['stays_in_week_nights', 'stays_in_weekend_nights', 'adults', 'children', 'is_canceled', 'total_of_special_requests', 'required_car_parking_spaces', 'booking_changes']))

# Creating an empty Dataframe with column names only
t_stat_results = pd.DataFrame(columns=['t_stat', 'p_val', 'result'], index = num_cols)

for col in num_cols:
  res = significance(col)
  t_stat_results.loc[col] = [res[0], res[1], res[2]]

print(t_stat_results)
```

    ##                                  t_stat         p_val                 result
    ## lead_time                       105.945             0  Dependent (reject H0)
    ## arrival_date_year               5.76971    7.9602e-09  Dependent (reject H0)
    ## arrival_date_month              3.80503   0.000141855  Dependent (reject H0)
    ## arrival_date_week_number         2.8128    0.00491207  Dependent (reject H0)
    ## arrival_date_day_of_month      -2.10804     0.0350293  Dependent (reject H0)
    ## babies                         -11.2319   2.94126e-29  Dependent (reject H0)
    ## is_repeated_guest               -29.403  2.39542e-189  Dependent (reject H0)
    ## previous_cancellations          38.2876             0  Dependent (reject H0)
    ## previous_bookings_not_canceled -19.8507   1.50658e-87  Dependent (reject H0)
    ## days_in_waiting_list            18.7513   2.45186e-78  Dependent (reject H0)
    ## adr                             16.4642   7.74945e-61  Dependent (reject H0)

## Machine Learning

Our findings from the statistical analysis phase suggest a large number
of significant features to use in our model. However each of the
confirmed statistically significant features must be examined to
determine if they may cause data leakage. Hotel bookings typically
change over time given customers options to customize their reservation.
Some of these options affect how their profile is logged, making those
customers who followed through with their booking seem more distinct
than those who did not. For example, a customer who makes changes
leading up to their booking will by far more likely to keep the
reservation. Since our goal is to predict the possibility of a canceled
booking given only the initial parameters of a booking, we must
eliminate many of these sources of data leakage. After examining the
data and reading carefully into the source of the data, it was
determined that all features that hold the potential for changing over
time must be eliminated.

To begin the machine learning phase the data must be split into features
and labels, and the feature variables must be encoded to resolve issues
for our model such as strings and high level categorical columns.

### Split the Data

``` python
# Use results from statistical analysis to form features list
cat_features = chi_square_results.index.to_list()
num_features = t_stat_results[t_stat_results['result'] != 'Ind (fail to reject H0)'].index.to_list()
features = cat_features + num_features
```

``` python
# load appropriate modules
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

X = hotels[features]
        
# Dummy coding
y = hotels['is_canceled']

# Dummy coding for col in cols
dummy_cols = ['market_segment', 'distribution_channel', 'hotel', 'kids', 'reserved_room_type', 'customer_type']
transformed = []
for col in dummy_cols:
  X[col] = pd.get_dummies(X[col])

# Label Encoder
```

    ## /Users/mattmerrill/opt/anaconda3/bin/python3:2: SettingWithCopyWarning: 
    ## A value is trying to be set on a copy of a slice from a DataFrame.
    ## Try using .loc[row_indexer,col_indexer] = value instead
    ## 
    ## See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

``` python
enc_cols = ['agent', 'company']
for col in enc_cols:
  data = hotels[col].to_list()
  values = array(data)
  # initialize labelencoder, fit and transform
  label_encoder = LabelEncoder()
  integer_encoded = label_encoder.fit_transform(values)
  transformed.append(integer_encoded)
  # Drop country column and replace with encoded 
  X.drop(col, axis = 1, inplace = True)
  X[col] = integer_encoded
```

    ## /Users/mattmerrill/opt/anaconda3/lib/python3.7/site-packages/pandas/core/frame.py:4102: SettingWithCopyWarning: 
    ## A value is trying to be set on a copy of a slice from a DataFrame
    ## 
    ## See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    ##   errors=errors,
    ## /Users/mattmerrill/opt/anaconda3/bin/python3:10: SettingWithCopyWarning: 
    ## A value is trying to be set on a copy of a slice from a DataFrame.
    ## Try using .loc[row_indexer,col_indexer] = value instead
    ## 
    ## See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

``` python
print(X.head())
```

    ##    hotel  market_segment  distribution_channel  ...   adr  agent  company
    ## 0      0               0                     0  ...   0.0      0        0
    ## 1      0               0                     0  ...   0.0      0        0
    ## 2      0               0                     0  ...  75.0      0        0
    ## 3      0               0                     1  ...  75.0    157        0
    ## 4      0               0                     0  ...  98.0    103        0
    ## 
    ## [5 rows x 19 columns]

### Logistic regression

``` python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#scaler in pipeline object, use logreg algorithm
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

    ## 0.701287098305274

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

#### Perform ridge regression and plot ridge scores vs. ridge scores std

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

![](Milestone_files/figure-gfm/unnamed-chunk-47-1.png)<!-- -->

#### Perform Lasso feature selection to determine influential variables.

``` python
# Lasso for feature selection in scikit learn
#store feature names
from sklearn.linear_model import Lasso
names = X.columns
lasso = Lasso(alpha=0.01)
# extract coef attribute and store
lasso_coef = lasso.fit(X, y).coef_
_ = plt.figure(figsize=(20,10))
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=45)
_ = plt.ylabel('Coefficients')
plt.show()
```

![](Milestone_files/figure-gfm/unnamed-chunk-48-1.png)<!-- -->

> It seems that hotels may be a source of overfitting. Since the
> documentation does not allude to any potential for data leakage among
> the hotels, this will be ignored for now.

#### Plot the ROC curve.

``` python
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import roc_curve

# plotting the ROC curve
y_pred_prob = pipeline.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr, label = 'Logistic Regression')
```

    ## [<matplotlib.lines.Line2D object at 0x1a2a137f50>]

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

![](Milestone_files/figure-gfm/unnamed-chunk-49-1.png)<!-- -->

#### Tune the model with Kfold cross validation

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
#scaler in pipeline object, use logreg algorithm
steps = [('scaler', StandardScaler()), \
         ('logreg', LogisticRegression(max_iter=200))]
pipeline = Pipeline(steps)
score = cv_score(pipeline, X_train, y_train)
print(score)
```

    ## 0.7033061312572861

#### Use grid search to optimize the inverse regularization parameter

``` python
#the grid of parameters to search over
Cs = [0.001, 0.1, 1, 10, 100]

# create empty dataframe 
df = pd.DataFrame(columns=['Cs', 'cv_score'])

# loop through and add scores    
for c in Cs:
    #scaler in pipeline object, use logreg algorithm
    steps = [('scaler', StandardScaler()), \
         ('logreg', LogisticRegression(C=c, max_iter=1000))]
    pipeline = Pipeline(steps)
    score = cv_score(pipeline, X_train, y_train)
    df = df.append({'Cs' : c , 'cv_score' : score} , ignore_index=True)
print(df[df.cv_score == df.cv_score.max()])
```

    ##       Cs  cv_score
    ## 4  100.0  0.703402

#### Compute accuracy score with the c-value that produces the highest accuracy.

``` python
#scaler in pipeline object, use logreg algorithm
steps = [('scaler', StandardScaler()), \
         ('logreg', LogisticRegression(C = 100, max_iter=1000))]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
# Print the accuracy from the testing data.
```

    ## Pipeline(memory=None,
    ##          steps=[('scaler',
    ##                  StandardScaler(copy=True, with_mean=True, with_std=True)),
    ##                 ('logreg',
    ##                  LogisticRegression(C=100, class_weight=None, dual=False,
    ##                                     fit_intercept=True, intercept_scaling=1,
    ##                                     l1_ratio=None, max_iter=1000,
    ##                                     multi_class='auto', n_jobs=None,
    ##                                     penalty='l2', random_state=None,
    ##                                     solver='lbfgs', tol=0.0001, verbose=0,
    ##                                     warm_start=False))],
    ##          verbose=False)

``` python
print(accuracy_score(pipeline.predict(X_test), y_test))
```

    ## 0.701287098305274

> The accuracy score with logistic regression may be improved by using a
> Random Forest but there is more potential for overfitting with this
> approach.

### Random Forests

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

    ## [0 1 0 ... 1 0 0]

``` python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

print(confusion_matrix(y_test,y_pred))
```

    ## [[20244  2267]
    ##  [ 4151  9155]]

``` python
print(classification_report(y_test,y_pred))
```

    ##               precision    recall  f1-score   support
    ## 
    ##            0       0.83      0.90      0.86     22511
    ##            1       0.80      0.69      0.74     13306
    ## 
    ##     accuracy                           0.82     35817
    ##    macro avg       0.82      0.79      0.80     35817
    ## weighted avg       0.82      0.82      0.82     35817

``` python
print(accuracy_score(y_test, y_pred))
```

    ## 0.8208113465672725

#### Plot Random Forests ROC curve

``` python
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import roc_curve

# plotting the ROC curve
y_pred_prob = rfc.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr, label = 'Random Forest')
```

    ## [<matplotlib.lines.Line2D object at 0x1a2baad390>]

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

![](Milestone_files/figure-gfm/unnamed-chunk-56-1.png)<!-- -->

> The accuracy score did improve with RF and the ROC curve is looking
> much better. Next I will move to tune some of the hyperparameters,
> specifically the number of estimators and the max depth.

#### Hyperparameter tuning with RandomizedSearchCV

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
# Implementing parameter tuning with n_estimators=160 and max_depth=260
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
# Cross validation
rfc_cv_score = cross_val_score(rfc, X_train, y_train, cv=5, scoring='roc_auc')
print("=== Confusion Matrix ===")
```

    ## === Confusion Matrix ===

``` python
print(confusion_matrix(y_test, rfc_predict))
```

    ## [[20289  2222]
    ##  [ 4052  9254]]

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
    ##            0       0.83      0.90      0.87     22511
    ##            1       0.81      0.70      0.75     13306
    ## 
    ##     accuracy                           0.82     35817
    ##    macro avg       0.82      0.80      0.81     35817
    ## weighted avg       0.82      0.82      0.82     35817

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

    ## [0.88495728 0.89112038 0.88446442 0.8864108  0.88539778]

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

    ## Mean AUC Score - Random Forest:  0.8864701310589705

``` python
tn, fp, fn, tp = confusion_matrix(y_test, rfc_predict).ravel()
recall = (tp/(tp + fn))
specificity = (tn/(tn + fp))
accuracy = (tp+tn)/(tp+fp+fn+tn)
print(recall, specificity, accuracy)
```

    ## 0.6954757252367353 0.9012927013460086 0.8248317837898205

> We see here an overall improvement with Random Forests, the accuracy
> score improved from 0.70 to 0.83 after tuning the hyperparameters and
> using cross validation. Next I will implement Gradient Boosting to see
> if this can’t be improved upon.

### Gradient Boosting

``` python
from sklearn.ensemble import GradientBoostingClassifier

# Learning rate to fit over, keeping hyperparameters from before
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
    ## Accuracy score (training): 0.985
    ## Accuracy score (validation): 0.824
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
    ## Accuracy score (training): 0.985
    ## Accuracy score (validation): 0.824
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
    ## Accuracy score (training): 0.985
    ## Accuracy score (validation): 0.823
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
    ## Accuracy score (training): 0.985
    ## Accuracy score (validation): 0.825
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
    ## Accuracy score (training): 0.985
    ## Accuracy score (validation): 0.822
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
    ## Accuracy score (training): 0.985
    ## Accuracy score (validation): 0.822
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
    ## Accuracy score (training): 0.985
    ## Accuracy score (validation): 0.819
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
    ## Accuracy score (training): 0.985
    ## Accuracy score (validation): 0.820
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
    ## Accuracy score (training): 0.985
    ## Accuracy score (validation): 0.819
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
    ## Accuracy score (training): 0.984
    ## Accuracy score (validation): 0.818
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
    ## Accuracy score (training): 0.980
    ## Accuracy score (validation): 0.806

> The training data is nearing perfect accuracy which suggests some
> overfitting
here.

``` python
# Implement gradient boosting with all tuned hyperparameters, including an optimal learning rate of 0.25
gb_clf2 = GradientBoostingClassifier(n_estimators=160, learning_rate=0.25, max_features='sqrt', max_depth=260, random_state=4)
gb_clf2.fit(X_train, y_train)
```

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

``` python
predictions = gb_clf2.predict(X_test)

print("Confusion Matrix:")
```

    ## Confusion Matrix:

``` python
print(confusion_matrix(y_test, predictions))
```

    ## [[20133  2378]
    ##  [ 3900  9406]]

``` python
print("Classification Report")
```

    ## Classification Report

``` python
print(classification_report(y_test, predictions))
```

    ##               precision    recall  f1-score   support
    ## 
    ##            0       0.84      0.89      0.87     22511
    ##            1       0.80      0.71      0.75     13306
    ## 
    ##     accuracy                           0.82     35817
    ##    macro avg       0.82      0.80      0.81     35817
    ## weighted avg       0.82      0.82      0.82     35817

``` python
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
recall = (tp/(tp + fn))
specificity = (tn/(tn + fp))
accuracy = (tp+tn)/(tp+fp+fn+tn)
print(recall, specificity, accuracy)
```

    ## 0.7068991432436494 0.8943627559859624 0.824720104978083

## Conclusion

The results of machine learning have yielded a model with over an 80%
accuracy in predicting cancellations. The implications of this work
bring a myriad of options to lower cancellation rate and increase
actualized income. Hotels can now include a probability of cancellation
with each booking, which would allow for them to create focused effort
towards preventing cancellations and thus optimize the distribution of
customer channels.

An example of how this work may be taken advantage of would be to employ
outreach tactics for bookings that have a high probability of cancelling
with discounted packages, perks and early options to customize stay
details. This focused effort would lower the overall cancellation rate,
increasing the actualized daily income and giving more control over
booking variability. Hotels can now be the ones taking advantage of the
online travel agency phenomenon by pivoting to meet the market where it
is by making use of this new innovative market prediction tool.

1.  Hertzfeld, Esther. Study: Cancellation Rate at 40% as OTAs Push Free
    Change Policy. Hotel Management, 23 Apr. 2019,
    www.hotelmanagement.net/tech/study-cancelation-rate-at-40-as-otas-push-free-change-policy.

2.  Funnell, Rob. “The Real Cost of ‘Free’ Cancellations for Hotels.”
    Triptease, Triptease - Attract. Convert. Compete., 13 May 2019,
    www.triptease.com/blog/the-real-cost-of-free-cancellations/?utm\_source=MediaPartner\&utm\_medium=HotelSpeak.
