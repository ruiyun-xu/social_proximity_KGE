
# Social Proximity Measure Using Knowledge Graph Embeddings

## Overview
This repository provides a Python implementation of a novel pairwise social proximity measure based on the Knowledge Graph Embedding (KGE) approach, specifically using the TransE model from the PyKEEN library. The code includes functions for training the TransE model on a graph represented by triples (subject, predicate, object) and calculating social proximity scores between pairs of entities based on their embeddings.

## Definition of Our Proposed Social Proximity Measure
Our proposed social proximity measure, named Entity Proximity, is defined as:

$` \text{Entity Proximity} = \frac{e_1 \cdot e_2^T}{||e_1|| ||e_2||} = \frac{\sum_{m=1}^{M} e_{1,m} e_{2,m}}{\sqrt{\sum_{m=1}^{M} \left(e_{1,m}\right)^2} \sqrt{\sum_{m=1}^{M} \left(e_{2,m}\right)^2}} `$

Where:

*   `e₁` and `e₂` are the embeddings of the entities learned from the TransE model.
*   `e₁ ⋅ e₂ᵀ` denotes the dot product of the two embedding vectors.
*   `||e₁||` and `||e₂||` are the Euclidean norms of the vectors `e₁` and `e₂`.
*   `e₁,ₘ` and `e₂,ₘ` represent the `m`-th component of embeddings `e₁` and `e₂` respectively.
*   `M` is the dimensionality of the embedding vectors.

### Interpretation

Theoretically, the range of entity proximity is between -1 and 1, where a larger value indicates a higher level of social proximity.


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
- `example_training.csv`: A CSV file containing the training data in the form of triples (subject, predicate, object), for example, (Alice works_for CompanyX).
- `social_proximity_kge.py`: A Python script implementing the social proximity measure. It can be executed using any Python IDE.
- `social_proximity_KGE.ipynb`: A Jupyter Notebook with the social proximity implementation. It can be run using Jupyter Lab, Google Colab, or any Jupyter-compatible environment.

## Usage
1. **Data Preparation**: Prepare your data in a CSV file with columns: `subject`, `predicate`, and `object`. This data will be used to train the TransE model.

2. **Training the Model**: Use the `TransE_train()` function to train the TransE model on your data and generate entity embeddings.

3. **Calculating Social Proximity**: Use the `social_proximity_given_entity_pairs()` function to calculate the social proximity score (cosine similarity) between any given entity pairs based on their embeddings.

## Functions

### 1. `TransE_train(training_data_path, embedding_dim, num_epochs)`
This function trains the TransE model using the provided training data and returns the entity embeddings.

**Parameters**:
- `training_data_path` (str): Path to the CSV file containing the training data.
- `embedding_dim` (int): Dimension of the entity embeddings.
- `num_epochs` (int): Number of epochs for training.

**Returns**:
- `entity_embeddings` (numpy.ndarray): The trained entity embeddings.
- `triples_factory_train` (pykeen.triples.TriplesFactory): The TriplesFactory object for the training data.

**Note**: A `TriplesFactory` is a core class in PyKEEN that handles the creation, management, and processing of triples data, which is fundamental for training and evaluating knowledge graph embedding models. A triple in the context of knowledge graphs consists of three elements: (head entity, relation, tail entity), typically represented as `(h, r, t)`. The `TriplesFactory` class is used to convert these triples into a format that can be used by PyKEEN's models and utilities.

### Example Usage:
```python
entity_embeddings, triples_factory_train = TransE_train(
    training_data_path='example_training.csv',
    embedding_dim=50,
    num_epochs=100
)
```

### 2. `social_proximity_given_entity_pairs(entity_pairs, entity_embeddings, triples_factory)`
Calculates the social proximity score for a given pair of entities based on their embeddings. The equation for calculating social proximity is provided in the section titled **'Definition of Our Proposed Social Proximity Measure** above.

**Parameters**:
- `entity_pairs` (list): A list containing two entities (e.g., ["Alice", "Bob"]).
- `entity_embeddings` (numpy.ndarray): A NumPy array containing all entity embeddings.
- `triples_factory` (pykeen.triples.TriplesFactory): The TriplesFactory object for the training data.

**Returns**:
- `entity_proximity`: Social proximity score between the given entity pairs.

### Example Usage:
```python
pairwise_score = social_proximity_given_entity_pairs(
    entity_pairs=["Alice", "Bob"],
    entity_embeddings=entity_embeddings,
    triples_factory=triples_factory_train
)
```

## Running the Code
1. Ensure the path for `example_training.csv` file is correctly specified in the script.
2. Run the script to train the TransE model and calculate social proximity scores.

## References
For more details on the PyKEEN library and its usage, visit the [official PyKEEN documentation](https://pykeen.readthedocs.io/en/stable/).
