#  DP-MLM Replication and Extension

This repository contains the **replication and extension of DP-MLM (Differentially Private Masked Language Model)** proposed by *Meisner et al., Findings of ACL 2022*.  
The project reproduces the original DP-MLM system and evaluates its generalisation to new datasets  **IMDB**, **Twitter**, and a **synthetic dataset** created to simulate real-world privacy-sensitive text.

---

##  Project Overview

DP-MLM is a privacy-preserving text rewriting approach that replaces sensitive tokens (names, IDs, phone numbers, etc.) with semantically consistent alternatives using masked language models and differential privacy noise.  
This project validates the published results by replicating the core system and extending it across new domains.

The implementation uses **RoBERTa-base** as the underlying transformer model, combined with a configurable privacy parameter `ε (epsilon)` that controls the privacy–utility trade-off.

---

##  Folder Structure

DPMLM-Update/
- DPMLM.py → Core implementation of DP-MLM
- run_dpmlm.py → Entry point to run the model
- rewrite_imdb_demo.py → Rewriting pipeline for IMDB dataset
- rewrite_tweets_demo.py → Rewriting pipeline for Twitter dataset
- new_constructed_dataset.py → Generates the synthetic dataset
- requirements.txt → List of dependencies
- dataset link.txt → Source dataset information
- data/ → Contains input and rewritten CSV files
- Screenshots/ → Visual results (replication.png, IMDB, Twitter, Constructed)
- README.md → Project documentation

##  Setup Instructions

1. Clone this repository or download it as a ZIP file.
   ```bash
   git clone https://github.com/Simran5429/DPMLM-Replication

2. Install required dependencies:
   
   pip install -r requirements.txt

## Running the code
 
   Run each demo script separately to reproduce results for different datasets:

   # For IMDB dataset
python rewrite_imdb_demo.py

   # For Twitter dataset
python rewrite_tweets_demo.py

   # For Synthetic dataset
python new_constructed_dataset.py


