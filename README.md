# Predicting Hotel Cancellations

**Can we predict if a guest will cancel a reservation at the moment of booking?**

The booking process has changed dramatically over the past decade, with more guests choosing to book online rather than direct. While the accessibility of online travel agencies may increase exposure and demand for many hotels, it has been met with an increase in cancellation rates. While cancellations are a familiar foe of the hotel industry, it has been the advent of the "risk free cancellations" campaign put on by online travel agencies that have made it a damaging statistic worthy of a second look. 

According to a study conducted by D-Edge Hospitality Solutions, cancellation rates in the hotel industry peaked at 41.3% in 2017, up from around 32% in 2014. What is importance to note here is how heavily skewed this average is by online travel agencies like "Booking.com" who posted a whopping 50% cancellation rate in 2018^[Hertzfeld, Esther. Study: Cancellation Rate at 40% as OTAs Push Free Change Policy. Hotel Management, 23 Apr. 2019, www.hotelmanagement.net/tech/study-cancelation-rate-at-40-as-otas-push-free-change-policy.] This is in stark contrast to an average cancellation rate of 18.2% in 2018 for customers booking direct. The booking process has changed and hotels are now forced to find ways of limiting the damage caused by cancellations. My work here aims to predict cancellations and offer a solution based on early outreach for *red flags* or high cancellation risk bookings. 

The dataset was obtained from [Science Direct](https://www.sciencedirect.com) and contains a collection of observations taken from 2015 to 2017 of guest bookings for two hotels, both located in Portugal. The data was collected directly from the hotels’ PMS (property management system) database and was relatively clean and structured upon retrieval. Each observation represents a booking, with the variables capturing details including when it was booked, dates of stay, through which operator the guest chose to go through, if that booking was cancelled and the average daily rate.

-------------------

## Overview

The project comprises all steps of Data Science work broken down as follows:

* Data collection and wangling: done in Jupyter Notebook
* Exploratory Data Analysis: using python in Rstudio with the reticulate library for statistical data analysis
* Machine learning: using Python - Logistic Regression and Random Forests with scikit-learn and a final CatBoost algorithm in Jupyter Notebook
* Report completed and rendered as Rmarkdown document

## Links

The work has been broken down in stages and summary slides have been created for a quick look at the results.

* [Data Wrangling and Cleaning](https://github.com/merrillm1/Predicting_Hotel_Cancellations/blob/master/Jupyter_Notebooks/Data_Wrangling_and_Cleaning_Steps.ipynb) - Cleaning steps with justification
* [Exploratory analysis](https://github.com/merrillm1/Predicting_Hotel_Cancellations/blob/master/Jupyter_Notebooks/Exploratory_data_analysis.ipynb) - Found trends and initial insights into cancellations
* [Statistical Analysis](https://github.com/merrillm1/Predicting_Hotel_Cancellations/blob/master/Jupyter_Notebooks/Statistical_Analysis.ipynb) - Identified statistically significant features
* [Machine Learning](https://github.com/merrillm1/Predicting_Hotel_Cancellations/blob/master/Jupyter_Notebooks/Machine%20Learning.ipynb) - The Catboost plots render nicely here
* [Final Report](https://github.com/merrillm1/Predicting_Hotel_Cancellations/blob/master/Milestone.md) - Very little code, reports findings of each stage
* [Summary Slides](https://www.slideshare.net/MatthewMerrill14/predicting-hotel-booking-cancellations-238388419) Link to ppt

## Author

* [Matthew Merrill](https://www.linkedin.com/in/matthew-merrill-246a1b55/)

## Acknowledgements

* **Dhiraj Khanna** - *Springboard mentor* 
