# Nerval Networks

>  I have a liking for lobsters. They are peaceful, serious creatures. They know the secrets of the sea. - Gerard de Nerval

Repo for ongoing work into directly learning a nerve complex from a dataset imbued with a metric using neural networks, which has nothing to do with Gerard de Nerval or lobsters apart from sharing a working name. 

The high level idea is to use an autoencoder with a a siamese auxillary loss and a sparse bottleneck to represent assignments to a set of overlapping microclusters. The number of these microclusters can be influenced by the regularization term. We can then build a nerve complex based on overlapping assignments between these microclusters to generate the Cech nerve of the metric space without the need for a projection step or refinement of the space through a secondary clustering mechanism. 
