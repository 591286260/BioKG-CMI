# BioKG-CMI

BioKG-CMI: A Multi-Source Feature Fusion Model Based on Biological Knowledge Graph for Predicting circRNA-miRNA Interactions

Initially, BioKG-CMI perform subcellular localization by utilizing the sequence information of circRNAs and miRNAs, generating negative samples accordingly. Subsequently, we construct a biological knowledge graph containing known relationships between circRNAs and miRNAs. The DisMult algorithm is used to learn feature representations of entities and relationships in graphs. Then, the spatial proximity between nodes of the same type is calculated, and the BERT is utilized to learn sequence features of circRNAs and miRNAs. Finally, these features are fused using the Adaboost classifier to predict potential CMIs.

Main dependency:  
python 3.8  
tensorFlow 2.10.0  
numpy 1.24.4  
gensim 4.3.1  
keras 2.4.3  

Usage:  
(A) Subcellular localization generation of negative samples and construction of biological knowledge graphs.  
(B) Learning multi-source signatures of circRNAs and miRNAs.  
(C) Feature fusion and prediction of CMIs using Adaboost.
