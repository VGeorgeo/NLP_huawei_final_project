# NLP_huawei_final_project
> This is the Final Projects of Natural Language Processing course from [ods.ai](https://ods.ai/tracks/nlp-course) and Huawei by [Valentin Malykh](http://val.maly.hk/)

## Project Abstract
A properly functioning search engine is an important part of creating websites. Correct recommendations can increase your traffic and profits. The results are based on the quantity and quality of the collected data and the selected recommendation model. In this work, I created a search engine for an aggregator of educational resources. The problem was solved by several approaches that were compared with each other.

## Problem
In given case, we are dealing with a limited set of names and descriptions of websites with an emphasis on training and education. The main problem of this task is the lack of user requests data and website transitions, so we can use only titles and descriptions. Also, the data is not labelled, so we can’t measure the accuracy of suggestions. However, by trial and error when using the current TF-IDF model, cases of unsatisfactory recommendations were identified. They can be used as a approbation of the proposed models. Moreover, the adequacy of the recommendations can be checked ‘manually’.

## Data

Unfortunately, I am unable to provide a full description of the data frame because it is a corporate secret. The main features that will be used are ‘title’ and ‘description’ (if it is available) of courses and other educational resources. The full size of the data frame is more than 50 thousand rows. The main language of the text and requests is Russian. However, the database is very muddled: some rows are uninformative, others have incorrect and incomplete translations.

## Data cleaning and pre-processing.

The first step was to remove the unhelpful information from the important data. The text analysis showed the existence of more than 2 languages, so the rows without English and Russian words were deleted. Also, the analysis showed some unique descriptors of uninformative strings which were assembled manually; the rows with these descriptors were deleted too. Similarly, strings of less than 20 symbols were found to be misleading and were cleaned. The problem of incorrect and incomplete translation was solved by translating texts with 2 languages types of letters into English and then into Russian again. The final clean database consists of three columns: 'title', 'description', and 'title + description’. It can be used with pre-trained nlp models like BERT.

However, other approaches require more complicated pre-processing. For example, the TF-IDF method is not able to distinguish between the same basic root words with different prefixes or suffixes. Therefore, another database with text pre-processing was developed. These are the stages of pre-processing:

- Clean data from html and other symbols
- Delete English and Russian stop words
- Make a lemmatization of each word
- Make a stemming of each word

Finally, there are two data frames: without preprocessing and with preprocessing, which will be used to make recommendations after the search query.


## Results.

All models described in the section "Data and methods" have been tested manually for various requests. In this text the phrase "Инженерное налогообложение" (**sequence**) which translates from English as 'Engineering Taxation' was chosen to demonstrate and compare the ability of the selected methods. This phrase was chosen because the previous model had problems with adequate recommendations: it often proposed texts about 'engineering', even though the main point of **sequence** is 'taxation'. Moreover, there is no actual information about the engineering taxation in the database, but it is still important to offer similar overlapping topics.

First of all, the TF-IDF method was applied. Model were fitted on the full text or only on the title. Since there are no any bigram equals or anything similar to **sequence** in the database, the model that collected bigram-only information did not find any similar texts or made bad recommendations. The TF-IDF models that were fitted on titles and only on unigrams proposed quite good suggestions, but some of the recommendations were about the engineering. The best results were demonstrated by models that fitted on the pre-processed full text (collected unigrams or both unigrams and bigrams). Despite the inaccurate suggestions TF-IDF approach can be used as one of the parts of the recommendation system, the similarity values can be multiplied by a coefficient k < 1.

Pre-trained nlp models showed more meaningful recommendations. Basically, the text did not contain the words from **sequence** but it was about overlapping topics such as financial accounting, tax law, etc. It is worth noting that, on the one hand, tiny models gave more misleading recommendations, but, on the other hand, the best recommendations are more specific and accurate. The basic models (large) give more general recommendations, but they are most often correct. RuBERT base, LaBSE, and ruBERT threeway models were chosen as more preferable for given task.

## Conclusion and discussion

In this work, I tried to solve the problem of a search engine and recommendations based on the given database. The database was cleaned from misleading information and pre-processed. TF-IDF and pre-trained nlp model were applied to make recommendations. TF-IDF obviously showed muddled recommendations because it bases them only on words frequency. However, it can still be used to search for certain unigrams, bigrams, and so on. Pre-trained models showed more reliable results, but they can be improved too. First of all, I will finalize the model for this database which should increase the adequacy. Also, it has been noticed that pre-trained models often suggest short texts, for example, with only a title. This may mean that the models of long texts are noisy and show less similarity. Thus, reducing the length of the text can improve accuracy. Another option, long text keywords can be extracted and used for searching similar texts.


These are some results of models recommendations for request: "Инженерное налогообложение"

<p align="center">
  <img src="https://user-images.githubusercontent.com/8645410/146072063-f2863bb6-a030-40ab-9af7-152f97cfb0d3.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/8645410/146072160-e87b87c5-b95c-4591-84be-6590c0580e60.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/8645410/146073860-ce9b529e-326f-450a-8d6c-7b74221f4b6a.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/8645410/146073250-bc0ef377-b0b3-4ffc-af1e-97a4980ee61e.png">
</p>





