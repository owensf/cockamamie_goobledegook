# README

This is a project I worked on that tried to use word embeddings to try and predict which words an individual person would find funny. Robust_Learning_from_Unfunny_Sources.pdf is the paper I wrote, and functions_untrusted_mychanges.py includes comments delineating original code I wrote. Everything else is credited below.

It is heavily based on some existing github repos, most notably https://github.com/limorigu/Cockamamie-Gobbledegook.

# Code Acknowledgements

## Sources
I utilized public GitHub repositories: 

https://github.com/limorigu/Cockamamie-Gobbledegook (abbreviated as CG)

https://github.com/NikolaKon1994/Robust-Learning-from-Untrusted-Sources (abbreviated as NK)

## main.py
The logic behind was done by me, but it utilizes many helper function code strands from CG, such as loading the Word Embedding .pkl files, and for building databases of words of interest.

## cockamamie.py
Mostly ripped from CG. Much of the structure had to be done by me, as the format of the CG GitHub is an .ipynb without and is disorganized (and a major pain to set up). get_cockammie(), get_EH() are examples of this: I had to parse what code to use from the .ipynb and make the functions in order to utilize it in the main

## functions_untrusted_mychanges.py
As the name suggests, this is where the bulk of my original code is written. First 186 lines are almost entirely original (use some snippets from CG, such as for setting up voter lists). The logic and the code is almost all original, because that's how I built my tests

Rest of the code is lightly modified. I had to change several lines in order to stop get the logic from NK to work with my data sets. I commented changes I made in the rest of this file with TODO: so that it's easy to find exactly what I changed. In addition, the commented out print statements throughout the rest of the code attest to the work I did on it.

# cockamamie_gobbledoog_us_data.json
From CG

# humor_dataset.csv
From https://link.springer.com/article/10.3758/s13428-017-0930-6

# wiki-news-300d-120kEH-subword.pkl
Wouldn't push to github because the file is too large. I can email this or something upon request
From https://fasttext.cc/docs/en/english-vectors.html
