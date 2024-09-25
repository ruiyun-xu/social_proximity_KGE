# social_proximity_KGE

# Social Proximity Measure Using Knowledge Graph Embeddings

## Overview
This repository provides a Python implementation of a novel pairwise social proximity measure based on the Knowledge Graph Embedding (KGE) approach, specifically using the TransE model from the PyKEEN library. The code includes functions for training the TransE model on a graph represented by triples (subject, predicate, object) and calculating social proximity scores between pairs of entities based on their embeddings.

## Requirements
- Python 3.7 or higher
- [PyKEEN](https://pykeen.readthedocs.io/en/stable/)
- Pandas
- NumPy
- Torch

To install the necessary dependencies, run:
```bash
pip install pykeen pandas numpy torch
```

## Files
- `example_training.csv`: A CSV file containing the training data in the form of triples (subject, predicate, object).
- `social_proximity_KGE.py`: A Python script implementing the social proximity measure. It can be executed using any Python IDE.
- `social_proximity_KGE.ipynb`: A Jupyter Notebook with the social proximity implementation. It can be run using Jupyter Lab, Google Colab, or any Jupyter-compatible environment.


## Usage
1. **Data Preparation**: Prepare your data in a CSV file with columns: `subject`, `predicate`, and `object`. This data will be used to train the TransE model.

2. **Training the Model**: Use the `TransE_train()` function to train the TransE model on your data and generate entity embeddings.

3. **Calculating Social Proximity**: Use the `social_proximity_given_entity_pairs()` function to calculate the social proximity score (cosine similarity) between any given entity pairs based on their embeddings.

## Functions

### 1. `TransE_train(training_data_path, testing_data_path, embedding_dim, num_epochs)`
This function trains the TransE model using the provided training data and returns the entity embeddings.

**Parameters**:
- `training_data_path` (str): Path to the CSV file containing the training data.
- `testing_data_path` (str): Path to the CSV file containing the testing data.
- `embedding_dim` (int): Dimension of the entity embeddings.
- `num_epochs` (int): Number of epochs for training.

**Returns**:
- `entity_embeddings` (numpy.ndarray): The trained entity embeddings.
- `triples_factory_train` (pykeen.triples.TriplesFactory): The TriplesFactory object for the training data.

### Example Usage:
```python
entity_embeddings, triples_factory_train = TransE_train(
    training_data_path='example_training.csv',
    testing_data_path='example_training.csv',
    embedding_dim=50,
    num_epochs=100
)
```

### 2. `cosine_similarity(entity_id1, entity_id2, entity_embeddings)`
Calculates the cosine similarity between two entity embeddings.

**Parameters**:
- `entity_id1` (int): The ID of the first entity.
- `entity_id2` (int): The ID of the second entity.
- `entity_embeddings` (numpy.ndarray): A NumPy array containing all entity embeddings.

**Returns**:
- `similarity` (float): The cosine similarity between the two entity embeddings.

### Example Usage:
```python
similarity = cosine_similarity(0, 1, entity_embeddings)
```

### 3. `social_proximity_given_entity_pairs(entity_pairs, entity_embeddings, triples_factory)`
Calculates the social proximity score for a given pair of entities based on their embeddings.

**Parameters**:
- `entity_pairs` (list): A list containing two entities (e.g., ["Alice", "Bob"]).
- `entity_embeddings` (numpy.ndarray): A NumPy array containing all entity embeddings.
- `triples_factory` (pykeen.triples.TriplesFactory): The TriplesFactory object for the training data.

**Returns**:
- `None`: Prints the cosine similarity between the given entity pairs.

### Example Usage:
```python
social_proximity_given_entity_pairs(
    entity_pairs=["Alice", "Bob"],
    entity_embeddings=entity_embeddings,
    triples_factory=triples_factory_train
)
```

## Running the Code
1. Ensure the `example_training.csv` file is in the same directory as the script.
2. Run the script to train the TransE model and calculate social proximity scores.

## References
For more details on the PyKEEN library and its usage, visit the [official PyKEEN documentation](https://pykeen.readthedocs.io/en/stable/).
