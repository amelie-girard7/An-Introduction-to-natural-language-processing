# An-Introduction-to-natural-language-processing
This repository contain my notes from the course  of Prof. Massimo Piccardi, 32931 TRM, Natural Language Processing at the UTS.

# Table of Contents

1. [What is Natural Language Processing (NLP)?](#what-is-natural-language-processing-nlp)
2. [An Overview of NLP Tasks](#an-overview-of-nlp-tasks)
3. [Document Vectors](#document-vectors)
4. [Word Embeddings](#word-embeddings)
5. [An Example of Two Major Tasks:](#an-example-of-two-major-tasks)
    - [Named-entity Recognition](#named-entity-recognition)
    - [Topic Modelling](#topic-modelling)
6. [Concluding Remarks](#concluding-remarks)

## What is Natural Language Processing (NLP)?
Natural language processing (NLP) lies at the intersection of linguistics and machine learning, focusing on the analysis and generation of human language.
This field has witnessed a rapid growth in interest and has become a dominant area within artificial intelligence.
NLP is applied across various sectors, including but not limited to social media, e-commerce, healthcare, educational technology, financial services, and the entertainment industry.

Historically, NLP methodologies were grounded in ***linguistic theory***, encompassing syntax, vocabulary, morphology, and meaning. However, as the field has evolved, ***statistical methods*** and data-driven ***machine learning*** have taken precedence.


### A historical overview

In the 1950s, the field of NLP emerged with the Turing Test and initial machine translation efforts, setting the stage for computational linguistics. The 1960s solidified this foundation with pivotal algorithms and ELIZA, the first chatbot, signaling the potential for machine understanding of language. The subsequent decade saw a shift towards rule-based systems like SHRDLU, which encoded linguistic rules but struggled with the complexity and variability of natural language.

The 1980s marked a turning point with the adoption of statistical methods, employing mathematical models to grapple with linguistic ambiguities, a significant move away from rigid rule-based systems. This statistical revolution blossomed in the 1990s as burgeoning text corpora became fodder for machine learning, enhancing the precision of NLP applications and giving rise to algorithms suited for a range of linguistic tasks.

Entering the 2000s, these advancements propelled speech recognition forward, leading to its integration into everyday technology and improving human-computer interaction. The next major leap occurred in the 2010s with the advent of deep learning and neural networks, which underpinned the development of sophisticated models capable of learning directly from vast amounts of data, minimizing the need for manual feature engineering.

Today, the 2020s are defined by transformer-based models and large language models like GPT-3, which have dramatically extended the capabilities of NLP. These models, leveraging self-attention mechanisms, have not only set new benchmarks in language generation and translation but have also expanded AI's reach into creative domains previously thought exclusive to human ingenuity.

<img src="./src/img/nlp_history.png" width="50%" height="auto">

### Contemporary NLP: key enablers
Three primary catalysts have propelled the swift expansion of NLP:
1. Enhanced ***algorithms*** that achieve practical accuracy levels in real-world applications.
2. Upgraded ***computational and networking infrastructures***, including GPUs, TPUs, FPGAs, and extensive cloud and edge computing resources. 
<img src="./src/img/computing.png" width="10%" height="auto">
3. The abundance of ***textual data***, which is now widely available in digital format from a diverse array of sources.

### NLP: a profusion of data
The scope of NLP data is broad, drawing from a multitude of sources such as an organization's proprietary documents, established publications, and a variety of web pages. It also includes data from product inventories and customer evaluations, online social discourse and blog entries, as well as journalistic content and broadcast transcriptions. Furthermore, this data is not limited to one language, often encompassing numerous tongues and their automatic translations. Additionally, spoken language data is captured through speech-to-text technologies like those in virtual assistants such as Siri, Alexa, and Cortana.

### ChatGPT
When ChatGPT debuted last November 2022, it quickly became a global phenomenon. It's important to acknowledge the unparalleled and astounding range of tasks it can perform. Its appeal lies in its adaptability, enabled through clever prompting. Yet, it isn't necessarily a substitute for other deep learning models that might be more efficient or superior in performance for certain tasks. Details on this will follow.

## An Overview of NLP Tasks
NLP encompasses a wide array of specialised tasks, typically grouped into:
- "Low-level" tasks, which are closely related to lexicon and syntax.
- "High-level" tasks, which deal more with semantics and meaning.

## Some popular NLP tasks
Some widely recognized NLP tasks are:
- Named-entity recognition
- Sentiment analysis
- Topic modelling
- Summarisation
- Machine translation
- Dialogue systems

Note: There are numerous additional tasks such as entailement recognition , relationship extraction, coreference resolution, wikification, …

**Named-entity recognition**
Named-entity recognition (NER) seeks to pinpoint and extract "named entities" from text, covering entities like person names, places, organizations, and various other specific categories. It's beneficial for categorizing text, detecting intent, identifying relationships, and build knowledge graphs.

<img src="./src/img/ner.png" width="50%" height="auto">
REX (Rosette Entity Extractor, BASIS Technology)

**Sentiment analysis**
Sentiment analysis is designed to determine the emotional tone behind a text, which could be anything from a tweet to an email:
<img src="./src/img/sentiment.png" width="50%" height="auto">


- At its most basic, it involves identifying particular keywords that indicate sentiment.
- In a more advanced method, the entire text is transformed into a vector (known as "document representation"), which a classifier then uses to deduce the sentiment's category.

# Reference 

1. **Turing Test proposal (1950):**
   - Turing, A. M. (1950). Computing machinery and intelligence. Mind, 59(236), 433-460.

2. **Chomsky's Syntactic Structures (1957):**
   - Chomsky, N. (1957). Syntactic Structures. The Hague/Paris: Mouton.

3. **ALPAC report (1966):**
   - Pierce, J. R., et al. (1966). Languages and Machines: Computers in Translation and Linguistics. A report by the Automatic Language Processing Advisory Committee (ALPAC), Division of Behavioral Sciences, National Academy of Sciences, National Research Council.

4. **SHRDLU development (1970):**
   - Winograd, T. (1972). Understanding natural language. Cognitive psychology, 3(1), 1-191.

5. **Statistical methods shift (1980s):**
   - The shift to statistical methods was gradual and involved the work of many researchers. A specific reference is hard to attribute to this general trend.

6. **Statistical NLP dominance (1990s):**
   - Manning, C. D., & Schütze, H. (1999). Foundations of Statistical Natural Language Processing. MIT press.

7. **Annotated corpora use (Late 1990s):**
   - Marcus, M. P., Marcinkiewicz, M. A., & Santorini, B. (1993). Building a large annotated corpus of English: The Penn Treebank. Computational linguistics, 19(2), 313-330.

8. **Introduction of word2vec (2013):**
   - Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.

9. **Transformer model architecture (2018):**
   - Vaswani, A., et al. (2017). Attention is all you need. Advances in neural information processing systems, 30.

10. **Large language models (GPT-3, 2020):**
    - Brown, T., et al. (2020). Language models are few-shot learners. Advances in neural information processing systems, 33, 1877-1901.
