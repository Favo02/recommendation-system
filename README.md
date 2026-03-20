## Recommendation System

> [!CAUTION]
> This repository contains the final project.
> The notes for the course are in a separate [repository](https://github.com/Favo02/algorithms-for-massive-datasets).

Project for [Algorithms for Massive Datasets](https://malchiodi.di.unimi.it/teaching/AMD/2025-26/) course at Università degli Studi di Milano - Master degree in Computer Science (a.y. 2025/26).

The implementation can be found in the [`main.ipynb`](./main.ipynb) notebook, while the [`report`](./report/) in its own folder _(written in [Typst](https://typst.app/), which is better than LaTeX)_.

One-click run with Google Colab: <a href="https://colab.research.google.com/github/Favo02/recommendation-system/blob/main/main.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Abstract

The project implements a **content-based** recommendation system for news articles using the _New York Times Articles & Comments (2020)_ dataset.
User interests are inferred from their **commenting behavior**, and both users and articles are represented as **feature vectors** based on article metadata (newsdesk, section, keywords) and comment history.
To efficiently identify similar user-article pairs from billions of potential comparisons, the system uses Local Sensitive Hashing (**LSH**) with random **hyperplane sketches**, followed by **cosine similarity** calculation.
The implementation leverages **Apache Spark** for distributed processing, enabling scalability to massive datasets.
Results show the approach generates high-quality recommendations for focused user profiles, with poorer performance for noisy user profiles.
