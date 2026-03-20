#import "@preview/rubber-article:0.5.2": *
#import "@preview/equate:0.3.2": equate

#show ref: underline
#show link: underline

#show: equate.with(sub-numbering: true)
#show: article.with(
  eq-chapterwise: false,
  eq-numbering: "(1.1)",
  header-display: true,
  lang: "en",
  page-paper: "a4",
  heading-numbering: "1.1.",
)

#let note = content => box(stroke: gray, inset: 0.8em, width: 100%)[
  _Note:_ #content
]

#let comment = content => text(size: 8pt)[(#content)]

#maketitle(
  title: "Nyt Articles & Comments (2020): A Content-Based Recommendation System",
  authors: ("Luca Favini",),
  date: datetime.today().display("[day]. [month repr:long] [year]"),
)

#align(center)[
  Algorithms for Massive Datasets -- Final Project

  Università degli Studi di Milano -- Master degree in Computer Science
]

#abstract(width: 70%, alignment: center)[
  The project implements a *content-based* recommendation system for news articles using the _New York Times Articles & Comments (2020)_ dataset.
  User interests are inferred from their *commenting behavior*, and both users and articles are represented as *feature vectors* based on article metadata (newsdesk, section, keywords) and comment history.
  To efficiently identify similar user-article pairs from billions of potential comparisons, the system uses Local Sensitive Hashing (*LSH*) with random *hyperplane sketches*, followed by *cosine similarity* calculation.
  The implementation leverages *Apache Spark* for distributed processing, enabling scalability to massive datasets.
  Results show the approach generates high-quality recommendations for focused user profiles, with poorer performance for noisy user profiles.
]

= Dataset

The _New York Times Articles & Comments (2020)_ @NytArticlesCommentsDataset dataset contains all articles published in 2020 and all comments left on those articles during the year.
The version 6 of the dataset was downloaded from Kaggle on March 10, 2026.
It contains approximately $15,000$ article records and $5,000,000$ comments.

== Preprocessing

For each record, the dataset contains various information.
Once loaded, the dataset is preprocessed to discard all the fields not used by the system.
Specifically:

/ Articles data: Only the _newsdesk_, _section_, _subsection_, and _keywords_, beyond the _articleID_, are kept.
  These describe the article's topic, the first three organize articles in a hierarchy in the New York Times website, while keywords are specific topic tags added by the editors.
  The keyword field comes as a text list like `['keyword1', 'keyword2', ...]`, and it is parsed into a real list.

/ Comments data: These are used to infer the users behaviour and interest.
  Only the _articleID_, the article on which the comment was left, and _userID_, the user that commented, are kept.

#vspace

This information is used to build profiles that describe _articles_ and _users_.
More on that in @utility-matrix and @profiles.

== Subsampling

To keep the computation time reasonable, a subsampling system is implemented.
By default, only a part of the comments is used, loading a smaller file with $500,000$ comments.
Furthermore, both articles and comments can be subsampled using Spark primitives by tweaking the configuration (@parameters).

#note[
  From now on, it is assumed the default parameters are used: smaller comments file with no further subsampling.
]

= Method

The general pipeline for performing recommendations is:
- Build the utility matrix, mapping interactions between users and articles
- Build profiles for users and articles based on the utility matrix and article metadata
- Find similar user-article pairs based on cosine similarity:
  - Compress profiles into sketches using random hyperplanes
  - Use LSH to find candidate pairs of user-article that are similar enough to be recommended
  - Compute the actual cosine similarity for each candidate pair and select the top recommendations for each user

== Utility Matrix Construction <utility-matrix>

The dataset does not have explicit ratings from users, but this information is needed when dealing with recommendations.
To fill this hole, the assumption that if a user comments on an article, they are interested in that topic is made.
This is reasonable: users spend time commenting on articles that interest them.

A simple matrix $M in RR^(|U| times |A|)$ is created that shows which user commented on which article, where $|U|$ is the number of users and $|A|$ is the number of articles.
$
  M_(u, a) = cases(
    1 quad & "if" u "commented" a,
    0 quad & "otherwise"
  )
$
The matrix is very sparse, on average each user interacted with $4$ articles.

#vspace

/ Duplicate removal: Multiple comments on the same article from the same user are treated as just one interaction.
  Commenting many times does not hold value, intuitively heated discussions does not mean more interest and bring additional complexity.

== Profile Representation <profiles>

Each _article_ and _user_ is represented as a vector, where each dimension of the vector corresponds to one feature.
These features come primary from two sets:
- _Structural_: newsdesk, section, subsection (fewer than 200 different values total)
- _Keywords_: about 15,000 different keywords

#note[
  Another idea would be to use indexes like TF.IDF to extract the features of articles.
  Without full article body, this is not very feasible.
]

_Articles_ have trivial boolean vectors (just 1 or 0 for each feature) based on their categories.
_User_ profiles are slightly more complex: each component counts how many times that feature appears across articles the user commented.
User profiles are thus vectors of positive integers.

#note[
  Normalizing user profiles (making the entries a probability distribution) does not give any advantage, for the cosine similarity (described in @similarity) the vector and the normalized vector are equal.
]

Each profile can be encoded in two ways: using a _dictionary_ or using a _hash function_.

#vspace

/ Dictionary metod: All the distinc features (both structural and keywords) are collected.
  Then an indexing dictionary is built for easy conversion from string to vector index.

  If all keywords are kept, the feature list becomes too large, over $15,000$.
  Therefore, keywords that appear fewer than 5 times are removed.
  This brings down the number of features to around $3,500$.
  The idea behind this is to simplify computation by ignoring rare keywords that do not hold much value for recommendations.

  #note[
    The frequency threshold can be tweaked by changing the `MIN_KW_FREQ` parameter in the configuration (@parameters).
  ]

  The size of this dictionary is approximately $(L_"key" + 4) dot |"features"|$ where $L_"key"$ is the space occupied by the string of the key.
  For around $3,500$ features, the dictionary is less than $1 "MB"$.

  - Advantage: Any feature number can be looked up to determine exactly what it means by building a reversed dictionary.
  - Disadvantage: If a new article has a keyword not previously seen, the vectors need to be recomputed to accomodate the new feature.
  - Disadvantage: The size of the dictionary is not bound, it depends on the dataset and it could explode with the number of keywords.

#vspace

/ Hashing method: Instead of explicitely building a dictionary, an hash function is applied to a feature to map it to an index (the obtained bucket).
  By setting an appropriate number of buckets (by default $5,000$), the built profile is granular enough to identify different features.

  #note[
    The number of buckets can be tweaked by changing the `FEATURES_SIZE` parameter in the configuration (@parameters).
  ]

  - Advantage: There is no need to store any data structure, only the hash function.
  - Advantage: New features fit simply by hashing them, no need to recompute anything.
  - Advantage: No need to prune rarely used keywords.
  - Disadvantage: There could be collisions (with more distinct features than buckets, there will be), so different features will map to the same index in the profile.
    These collisions do not pose a big problem as are probably distributed equally among all keywords.
  - Disadvantage: Impossibility to reverse lookup a profile, an hash function is not navigable back.

#vspace

With a relatively small and fixes number of keywords, the dictionary method is preferable.
For bigger and constantly updated datasets, the hashing approach is better.

#note[
  The strategy for building profiles can be tweaked by changing the `PROFILE_STRATEGY` parameter in the configuration (@parameters).
]

== Finding Similar Users and Articles <similarity>

Once profiles have been computed, the recommendation system needs to find _similar_ pairs of user-article.
These will become the recommendations.

Comparing every user to every article would require $O(|U| times |A|)$ comparisons.
With $91,000$ users and $16,000$ articles, this amounts to about $1.5$ billion comparisons, which is computationally infeasible.
To reduce that complexity, _local sensitive hashing_ (LSH) is used.
Out of all user-article pairs, only some pass the LSH filter, becoming the candidate pairs.

#vspace

/ Cosine Similarity: The similarity measure used for evaluating similarity is the _cosine similarity_, which measures the angle between the profile vectors:
  $ "sim"(u, a) = (u dot a) / (||u|| ||a||) $
  Cosine similarity best fits this application:
  - _Better than Euclidean distance:_ Euclidean distance considers a user interested in sports 500 times and politics 100 times in a very different way than a user interested in sports 5 times and politics 1 time.
    But these users share the same preferences.
  - _Better than Jaccard:_ Jaccard ignores weighted counts.
    A user interested 500 times in Sports and 1 time about politics is the same of a user interested equal times in sports and politics.

#vspace

/ Compressing the Profile (Sketches):
  Each profile (user or article) is compressed into a smaller representation, a _sketch_.
  This is done by checking if the profile is on the positive or negative side of 100 random $D$-dimensionale hyperplanes (each identified by a vector $underline(v) in \{-1, +1\}^D$) in feature space.
  This can be done by checking the result of the dot product between the profile and the vector $underline(v)$.

  #note[
    The number of hyperplanes (sketch size) can be tweaked by changing the `LSH_NUM_SKETCHES` parameter in the configuration (@parameters).
  ]

  #note[
    Because the vectors are not strictly binary, min-hashing technique is not optimal.
    Instead, the random hyperplanes technique described above is used.
    This produces a 100-bit sketch for each user and article.
  ]

#vspace

/ Finding Candidates Pairs (LSH): The sketches are divided into $b$ bands of $r$ bits.
  Each band is hashed into a _bucket_.
  Users and articles that fall in the same bucket are considered candidates for recommendation.

  To calculate the probability of two profiles being in the same bucket, we can use the fact that the probability of two profiles being on the same side of a random hyperplane is related to their cosine similarity.

  #note[
    Because the vectors are positive, the similarity between a user and an article ranges between $0.0$ and $1.0$, equivalent to angles of $90°$ and $0°$ respectively.
  ]

  $
    p = 1 - theta / (180°) quad & #comment[cosine similarity] \
                  PP = p^r quad & #comment[two profiles are equal in a band of r bits] \
              PP = 1 - p^r quad & #comment[not matching a full band] \
          PP = (1 - p^r)^b quad & #comment[not matching all bands] \
      PP = 1 - (1 - p^r)^b quad & #comment[matching in at least one band]
  $

  With default parameters of $r = 10$ and $b = 10$, the probability of passing the filter for various similarities $p$ is:
  - $"sim" = 0.9, space PP approx 0.99 approx 99%$
  - $"sim" = 0.8, space PP approx 0.67 approx 67%$
  - $"sim" = 0.7, space PP approx 0.25 approx 25%$
  - $"sim" = 0.6, space PP approx 0.05 approx 5%$
  - $"sim" = 0.5, space PP approx 0.009 approx 0.9%$

  This poses the theoretical threshold $t$ (the amount of similarity $p$ where the probability of passing the filter is 50%) at approximately $0.765$.
  The influence of the filter on the results is further discussed in @results.

#vspace

/ Actual Cosine Similarity:
  For each pair that passed the LSH filter, the actual cosine similarity is computed to discard False Positives.

#vspace

/ Performing Recommendations: For each user, top 10 articles with highest cosine similarity are selected.
  Articles the user has already commented are ignored.

= Implementation

The steps described above are implemented using PySpark, the Python API for Apache Spark.

== Data Storege: Sparseness

All data is stored in RDDs (Resilient Distributed Dataset) consisting of _sparse_ tuples.
Spark does not enforce key-value pairs for all operations.

Only non-zero values are stored to conserve memory and permit processing of huge datasets.
For example, some data strucutes previously described are stored as:
- Utility matrix: `(article_id, user_id)` pairs
- Article profiles: `(item_id, (feature_index, weight))` pairs, collected to the dense format `(item_id, [(feature_index1, weight1), ...])` only when stricly necessary (still making sure that the maximum size is limited by the number of features)
- LSH buckets: `(bucket, item_id)`

== Spark Broadcasting

Some data strucutes are needed by all Spark nodes, so they need to be broadcasted.

#vspace

/ Features Dictionary: For the dictionary method, the feature mapping must be sent to all worker machines.
  As calculated above, the order of magniture of this data structure are MB, so the broadcasting is achieved without problems.
  An alternative approach to that is the hashing method.

#vspace

/ Hyperplanes Broadcasting: To generate sketches for each profile, all nodes need all the norm vector describing the hyperplanes.
  The dimensionality and the number of each hyperplane is fixed and is, again, in the order of MB.

== Hash Functions

Because of random salting, the python built-in hash function is not consistent across different processes running on different machines.
This is a problem for the hashing approach: the same feature must be hashed to the same bucket.
The MurmurHash3 (`mmh3`) library for non-cryptographic hash functions, that provides a seed, is used.

== Lazyness and Caching

Spark computes every transformation in a lazy way, everything is recomputed each time a result is requested @SparkCaching.
If logging is enable (by default it is), at the end of every step, `count` and `take` are called, to have an idea of the amount and structure of the data.
Then, when that new RDD is used, the whole pipeline is recomputed.
This is a performance issue, in a production environment, these should be omitted.

A possible workaround would be to cache the RDDs and release (`unpersist`) after the use.
After a few experiment, without performances improvements (the opposite, some OOM failures due to too many cached RDDs), this idea have been dropped.
Caching has been applied only to RDDs that get actually used multiple times, and are expensive to recompute.

== Parameters <parameters>

The system is configurable by tweaking the parameters defined in the first cell of the notebook:
- `LOGGING = True`: Enable logging for amount and format of the RDDs (slower performances)
- `SEED = 25`: Seed of hash and random functions for reproducibility
- `SAMPLE_SIZE = 0.4`: Fraction of the dataset used (between 0.0 and 1.0)
- `PROFILE_STRATEGY = "dict"`: Strategy for building profiles, either "dict" or "hash"
- `MIN_KW_FREQ = 5`: Minimum frequency of keywords to be included in the profile (for the dictionary approach)
- `FEATURES_SIZE = 5000`: Number of features, buckets of the hash (for the hashing approach)
- `LSH_NUM_SKETCHES = 100`: Number of random hyperplanes (signatures/sketches)
- `LSH_BANDS = 10`: Number of bands for LSH
- `LSH_ROWS = 10`: Number of rows per band for LSH
- `RECOMMENDED_ARTICLES = 10`: Number of articles to recommend to each user

= Experimental Results and Analysis <results>

Running the system, playing with the parameters, and looking at the results, some observations can be made.

The dataset is not ideal for this approach, as it does not contain explicit ratings, but only comments.
Probably a collaborative filtering approach, that does not need explicit ratings, would have performed better.

#vspace

/ Recommendations behavior: The quality of the recommendations is very variable, depending on the user profile.
  Specifically, the system behaves differently in three cases:
  - _One topic:_ when the user commented only one article or articles of very similar categories, the recommendations are pretty good, with high similarity.
  - _Different topics:_ when the user commented on very different things, the recommendations are pretty bad, with low similarity.
    This is probably due to the fact that the user profile is very noisy, with many different features, and it is hard to find articles that span all those features.
  - _No recommendations:_ some users do not receive any recommendation ($approx 40%$ with subsampling, $approx 9%$ without subsampling), this can be caused either by subsampling (if the user commented on articles that are not in the subsample, they will not have any profile and thus no recommendation) or by the LSH filter (if the user profile is very noisy, it is unlikely to find any article that passes the filter).

#vspace

/ LSH Threshold: Another problem is posed by the tradeoffs needed to make the system work in a reasonable time for large datasets.
  By setting a treshold $t approx 0.765$, the pairs that are expected to pass the filter are those with very high similarity.
  But with complex user profiles, it is really hard to find articles that are similar enough to pass the filter.

  The simple idea of tweaking the parameters to make the filter less strict (increasing the number of bands and decreasing the number of rows) gets into the problem of not being able to reduce the amount of comparisons enough, making the LSH filter ineffective and exploding the computation time.

  The tradeoff made is that very good recommendations are found in a relatively small amount of time, but average recommendations are not found, and many users do not receive any recommendation.

#vspace

/ Scalability: The system runs in a reasonable time on the whole dataset with no subsampling (around $1$ hour) and is designed to scale to even bigger datasets without modification (using the hashing approach).
  The choice of storing data in a sparse format and the enforcement of never collecting something that could grow to enormous sizes make it possible to process huge datasets without running into memory issues (given that there are many nodes).

#show: appendix.with(title: "Disclaimer")

The project is part of the final evaluation of the _Algorithms for Massive Datasets_ course taught by _Dario Malchiodi_ at _Università degli Studi di Milano_ -- Master degree in Computer Science.

#vspace

_I declare that this material, which I now submit for assessment, is entirely my own work and has not been taken from the work of others, save and to the extent that such work has been cited and acknowledged within the text of my work.
I understand that plagiarism, collusion, and copying are grave and serious offences in the university and accept the penalties that would be imposed should I engage in plagiarism, collusion or copying.
This assignment, or any part of it, has not been previously submitted by me or any other person for assessment on this or any other course of study.
No generative AI tool has been used to write the code or the report content._

#vspace

The source code can be found at #link("https://github.com/Favo02/recommendation-system")[`https://github.com/Favo02/recommendation-system`].

#bibliography("sources.bib")
