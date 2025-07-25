# Multilingual RAG Telecom

This repository contains code, datasets, and notebooks for building and evaluating a Multilingual Retrieval-Augmented Generation (RAG) pipeline for telecom data. It leverages large language models and vector databases to process, analyze, and retrieve information from multilingual telecom conversations.


## Folder Structure

- `chromadb.zip`  
  Pre-built ChromaDB vector store (large file, tracked with Git LFS).

- `dataset/`  
  Contains raw and processed telecom conversation datasets, including:
  - `aggregated_conversations.csv`: Aggregated telecom conversations.
  - `generated/`: Preprocessed and generated datasets for training, evaluation, and topic modeling.

- `notebooks/`
  - `evaluation/`: Notebooks and scripts for evaluating RAG pipelines using metrics like BertScore and RAGAS. Includes results and plots.
  - `preprocessing/`: Notebooks for cleaning, preprocessing, and topic modeling of telecom data.
  - `rag/`: Main pipeline notebook for RAG implementation.
  - `vector-store-creation/`: Notebook for creating and managing vector stores.

- `presentation-video-ppt/`
  Presentation video, and PowerPoint slides related to the project.

## Key Notebooks

- `notebooks/evaluation/RAG-Evaluation-BertScore.py`: BertScore evaluation script.
- `notebooks/evaluation/RAG-Evaluation-Plots.ipynb`: Visualization of evaluation results.
- `notebooks/evaluation/RAG-Evaluation-RAGAS.py`: RAGAS metric evaluation script.
- `notebooks/evaluation/RAG-Retriver-Evaluation.py`: Retriever evaluation script.
- `notebooks/preprocessing/telecom-preprocess.ipynb`: Data cleaning and preprocessing.
- `notebooks/preprocessing/telecom-topicmodeling.ipynb`: Topic modeling notebook.
- `notebooks/rag/telecom-rag-pipeline.ipynb`: Main RAG pipeline implementation.
- `notebooks/vector-store-creation/telecom-vector-store-creation-new.ipynb`: Vector store creation notebook.

## Results

- Evaluation results and metrics are stored in `notebooks/evaluation/results/` as CSV and Excel files.

## Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/SuryaBandari247/Multilingual-RAG-Telecom.git
   ```
2. Install dependencies (Python, Jupyter, Git LFS):
   ```sh
   pip install -r requirements.txt
   git lfs install
   ```
3. Download large files tracked by Git LFS:
   ```sh
   git lfs pull
   ```
4. Open and run notebooks in the `notebooks/` folder for data processing, pipeline building, and evaluation.

## Notes
- Large files (>100MB) are tracked with Git LFS and may not be available in standard clones.
- For more details on each notebook, see the markdown cells at the top of each notebook.


