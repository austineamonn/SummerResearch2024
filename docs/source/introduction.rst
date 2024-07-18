Introduction
============

IntelliShield: A Privacy Preserving Explainable Educational Recommendation System
For the iCompBio REU program Summer of 2024 at the University of Tennessee Chattanooga.

Project Lead: Austin Nicolas.

Project Mentor: Dr. Shahnewaz Sakib.

General Outline of Summer Research Project:
-------------------------------------------
First a synthetic dataset was generated based on both real life data and synthetic mappings. Within the mapping there are three column types: Xp are private data that should not be leaked, X are the data being used to calculate Xu the utility data that the machine learning model is trying to predict. Then the feature importance for the dataset was calculated based on how much each X column impacted the target Xu column. Then the data was privatized using a variety of techniques including variations on differential privacy and random shuffling. Then the privacy loss - utility gain tradeoff was calculated across machine learning models and privatization techniques.

Goal:
-----
Take student input data and build a privatized version to train a machine learning model. The machine learning model will provide students with topics for future study and possible career paths. Then the students take these topics and paths to advisors, professors, counselors, peers, and others. These people will help the student consider next steps (picking classes, career fairs, etc.) based on the results.

Applications:
-------------
Data privatization techniques are vital for allowing data to be shared without risking exposing the sensitive data to being identifiable by malicious parties. Though this project uses student demographic information as the sensitive data, this work is very applicable to medical data collection and analysis.
