#import "@preview/rubber-article:0.5.2": *

#show: article.with(
  eq-chapterwise: true,
  header-display: true,
  lang: "en",
  page-paper: "a4",
  heading-numbering: "1.1.",
)

// Frontmatter
#maketitle(
  title: "Nyt Articles & Comments (2020): A Content-Based Recommendation System",
  authors: ("Luca Favini",),
  date: datetime.today().display("[day]. [month repr:long] [year]"),
)

// [ ] correctness of the general methodological approach,
// [ ] replicability of the experiments,
// [ ] correctness of the approach,
// [ ] scalability of the proposed solution,
// [ ] clarity of exposition.

// [x] the considered version of the dataset (in terms of access date, as the dataset might be updated frequently), and the parts of the latter which have been considered,
// [x] how data have been organized,
// [x] the applied pre-processing techniques,
// [x] the considered algorithms and their implementations,
// [x] how the proposed solution scales up with data size,
// [x] a description of the experiments,
// [x] comments and discussion on the experimental results.

#abstract(width: 70%, alignment: center)[
  This project contains an Apache Spark implementation of a content-based recommendation system for articles in the _New York Times Articles & Comments (2020)_ dataset.
  Users and articles are encoded as a vector of features.
  Similar users and articles are computed, using cosine similarity.
  Because of size of the dataset, the user-article pairs are filtered using LSH.
]

= Dataset

The dataset, _New York Times Articles & Comments (2020)_, found on Kaggle (version 6, downloaded on 10 March 2026 @NytArticlesCommentsDataset) contains all the *articles* published in 2020, alogn with all the *comments* left on those articles.

- for articles, the only information used is newsdesk, section, subsection, keywords
- this information was used to build a profile on what topics the article covers
- if the dataset included the whole body of the article, an idea could be to extract manually the topics of the article, for instance with techniques like TF.IDF

- these articles must be recommended to users, so also a user profile is needed
- because there isn't any information on the users, it is all inferred by the comments left by the users on the articles
- so the only information kept from comments file is the association between a user and an article (more on that in section Method > Utility Matrix)

- the artciles are in a single file with ~16k articles (all loaded and processed)
- the comments are already divided in files with ~500k commente each, only one of these files is processed (subsampling for keeping the running time reasonable)
- there is also a parameter for subsampling the dataset

- preprocessing: all useless columns dropped
- the title and abstract, used for human evaluation and then re-joined with the article id after the whole recommendation calculation
- the keywords field is a list of tags encoded as a string, so it is preprocessed by being parsed back to an actual list of strings

= Method

== Utility Matrix

- as said above, there is not explicit information on wheter a user liked (or even just read) an article
- so the matrix is build by putting to 1 the entries where a user interacted with an article, resulting in a binary matrix (for this, just use the comments datafile)
- multiple comments on the same article (from the same user, of course) are discarded as do not hold many importance, the idea is that a heated discussion does not mean the user is really more interested in an article

== Building Profiles

- profiles are a vector representation of the topics an article is about / an user is interested in
- each entry of the vector corresponds to a feature (one of newsdesk, section, subsection and keywords)
- building the profiles is achieved with two approaches: dictionary and hashing

=== Features Dictionary

- so the size of the vector is the sum of all the distinct newsdesk, section, subsection, keywords
- because of the abundace of kwywords (~15000), the ones used really few times (e.g. < 5) are discarded (result: ~5000)
  - TODO: source of that pruning? the idea is these keywords give a few information, we recommend based on the other more frequent ones
- then a dictionary mapping the feature name to an index is build for easy conversion
- pros: each feature is identified by a single index in the vector
- pros: given a profile, you can go back to wheter feature that is referred to
- contro: when new features are added (e.g. by a new article), the vector need to be extended (and so recomputed)
- contro: the vector could grow to infinite size

=== Features Hashing

- instead of mapping each feature (newsdesk, section, subsection, keyword) to a index, we hash it and then put to `1` the bucket they ended up
- the profile is now a fixed size vector (the number of buckets of the hash)
- pros: faster to compute (no need to compute the dictionary)
- pros: new features handled like the old ones, just hash
- contro: no way to determine a `1` in the profile to what feature is referred (no way to navigate back an hash function)
- contro: collisions, as there are more distinc features than buckets, there must be collisions, multiple features to the same bucket
- contro/pro: we dont care about collisions

=== Article and User Profiles

- article profiles are boolean: 1 wher the feature is on for that article, 0 where not
- user profiles are build by counting the number of articles commented for that feature by the user, so these are not boolean but integer positive numbers
- no need to normalize the user vectors because the distance used after (more in cosine similarity section) does not depend on that ([20,5] == [4,1])

== Similarity

- once all profiles are built the task is simply to find article profiles that are similar to user profiles, these will be our suggestions
- beceause of the amount of data, we cannot do a trivial n^2 approach, so we use LSH
- first of all we need a compressed version of the profiles (like the signature seen during the course)

=== Profile Sketches

- a sketch is the generic term for a signature (the signature refers to the minhash, but we dont use minhash)
- we dont use minhash because it strictly used for binary data (while our user profiles are not binary)
- so we use the random hyperplane technique:
  - a random hyperplan is generated (by generating a norm vector)
  - check wheter the profile is above or below the hyperplane
  - this check is performed by calculating the dot product
- 100 hyperplanes are generated and the 100 binary results are the sketch of a profile

=== LSH

- once the sketches are ready, we can perform LSH, so we divide in bands and rows
- TODO: calculate r and b parameters, and check wether the results are over treshold t (otherwise are bad results)
- hash each band and send to a bucket
- bands of articles and bands of users that end up in the same bucket are candidate similar pairs
- buckets too big are pruned (ignored)
  - TODO: justification to this? why simply not increase the number of rows of each band
- from that, we remove duplicates (can match on multiple bands) and articles users already commented

=== Actual Similarity: Cosine Similarity

- cosine similarity has been chosen because it is based purely on the direction of the vectors (profiles), ignoring magnitude
  - euclidean distance: bad because two users that read articles in the same proportion, 20% sport, 80% politics, e.g. [1, 4] and [5, 20] are really distant and so receive different recommendations
  - jaccard distance: bad because user profiles are not boolean, so a user with 1 comment on sport articles and 100 comments on politics articles receives recommendations both on sport and politics with the same frequency
- actual similarity computed for all candidates pairs on profiles
- TODO: make sense to actually calculate the cosine similarity not on the profiles but on the actual features (so that the collision of the hashing approach are addressed), less FP
- TODO: calculate probability of passing the similarity based on r and b

== Performing Recommendations

- with actual similarity, just group by user and sort by similarity
- the top N articles are suggested to the user

= Implementation

In order to elaborate huge datasets, the system is implemented using the PySpark API for Apache Spark.
The framework uses a MapReduce framework.

- everything is encoded by tuples, spark does not enforce (key, value) pairs, but for many operations this will be needed

== Setup

- configuration setup, parameters that can be tweaked in first cell:

- `MIN_KW_FREQ` (deafult `5`): For the dictionary approach (TODO link), only features that appear at least `MIN_KW_FREQ` times will be included in the profiles (for both Users and Articles).
- `FEATURES_SIZE` (deafult `5000`): For the hashing approach (TODO link), the number buckets of the hash function, in other words, the number of different features.
- `LSH_NUM_HASHES` (default `100`): number of random hyperplanes (signatures/sketches)
- `LSH_BANDS` (deafult `10`): play with b and r (calculate the threshold for different values)
- `LSH_ROWS` (deafult `10`):
- `LSH_MAX_BUCKET_SIZE` (deafult `500`): maximum number of users/articles in the same bucket
- `RECOMMENDED_ARTICLES` (deafult `10`): number of articles to recommend to each user
- `SEED` (default `None`): seed for replicability

== Hash function: mmh3

- problems of the built-in hash function:
  - for strings: each time a new python interpreter starts, it is initialized with a different salt that gets appended to strings
  - that means that on different nodes of the spark cluster, the same string would get hashed to different digests, breaking the hashing-based feature approach
- furthermore, the external library provides a better mathematical distribution

== Features dictionary

- the features dictionary built for the dictionary approach must be broadcasted constructed and then broadcasted to all spark nodes (all need it)
- when the number of features is limited this is ok (calculation of how many KB broadcasted at max)
- when the number of features increases, use hashing approach, no broadcast needed at all

== Sparseness

- because of the high dimensionality of basically everything (utility matrix, profiles, ...) everything is implemented with a sparse approach
  - profiles: (item_id, (feature_id, quantity)) - quantity is fixed to 1 for article profiles
  - utility matrix: (article_id, user_id)

== LSH: hyperplanes

- the hyperplanes need to be known by each node performing lsh hashing, so need to be broadcasted
- the size and number of hyperplanes is fixed, so the size broadcasted is fixed

== Optimization: Spark Caching

- particular attention to shuffling operations, done only where needed
- due to the lazy nature of spark, each time a "get" operation is done on an rdd, the full pipeline is run and then discarded
- the rdd used multiple times are cached and unpersisted as soon as they are not needed anymore
- TODO: lifetimes graph
- after calculating a step (basically a notebook cell) i run a take() to see the format of the data
- this runs the pipeline and then discards the result, so a caching before is needed
- in a production environment, this is not needed, so a few caches can be removed
- optimization: caching
  - https://luminousmen.com/post/explaining-the-mechanics-of-spark-caching/
  - https://luminousmen.com/post/spark-tips-caching

= Scalability

- the current dataset contains a "limited" amount of entries
- all the implemented operations are ok with handling even bigger datasets:
  - the only thing that could explode (and give OOM) with bigger datasets is the dictionary approach, where the features needs to be collected (for really really huge datasets)
  - just use hashing approach instead
- the things that could suffer are collect and group operations, performed only in:
  - dictionary approach (replace with hashing)
  - computing sketches: the user and articles profiles are not sparse anymore but in the form `item_id, [(feature_id, frequency), ...]`
  - this is not a problem because the number of features of each item is limited (if using the hashing approach)
  - same thing for actual cosine calculation
  - in human evaluation, who cares, its just for testing
- TODO: run on the full dataset (change caching to persist MEMORY and DISK or remove all take)

= Results

- runtime: on my machine, ~15m for 16787 articles and 500000 comments
- TODO: run for the whole dataset (hashing approach only)
- visualizing the results is complex as i literally recommend articles to users
- most of the users receive recommendations (82875 over 91437 users)
- TODO: why some users do not receive anything? check these ones
- watching manually some users, it seems that recommendations make sense
  - TODO: various "categories of users" (a lot of comments, a few comments, ...)
- more information would certanly benefit the process, for example the articles read (or clicked) by each user
- with this information, the more assumptions could be made, like open+comment = liked, open but not comment = not liked

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
