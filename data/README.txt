This folder contains the original datasets used in paper "SynSetExpan: A Joint Framework for Entity Set Expansion and Synonym Discovery” which is currently under review of KDD 2019. 

Each dataset contains following files:

1. corpus.txt: raw text corpus with entity linked to knowledge base. Each line represents a sentence.
2. vocab.txt: the vocabulary file. Each line is a term with its corresponding id, separated by “\t”. 
3. classes/class_<SEMANTC_CLASS_NAME>.txt: the ground truth semantic class. Each line represents an entity synonym set that belongs to the target semantic class.
4. queries/query_<SEMANTC_CLASS_NAME>.txt: the seed query for one semantic class. Each line represents a query that consists several seed entity synonym sets that belong to the target semantic class. 


Note: Although each term here is represented as "<entity-surface-name>||<freebase-id>", the freebase-id information is not used to train the term embedding or to learn the model. 



