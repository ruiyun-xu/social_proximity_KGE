# -*- coding: utf-8 -*-

# Install PyKEEN if not already installed
!pip install pykeen

# Import the necessary libraries
from pykeen.pipeline import pipeline
from pykeen.models import TransE
from pykeen.datasets import EagerDataset
from pykeen.triples import TriplesFactory
import pandas as pd
import numpy as np
import torch


# Specify data paths for training and testing data
training_data_path = 'example_training.csv'
test_data_path = 'example_training.csv'

# Train the TransE model using the training data
def TransE_train(training_data_path, testing_data_path, embedding_dim, num_epochs):
  # Load your training and testing data from a CSV file
  df_train = pd.read_csv(training_data_path)
  df_test = pd.read_csv(test_data_path)

  # Create a TriplesFactory from your DataFrame
  triples_factory_train = TriplesFactory.from_labeled_triples(
      triples=df_train[['subject', 'predicate', 'object']].values,
  )

  triples_factory_test = TriplesFactory.from_labeled_triples(
      triples=df_test[['subject', 'predicate', 'object']].values,
  )

  # Create an EagerDataset
  training_dataset = EagerDataset(
      training=triples_factory_train,
      validation=None,  # Provide None for validation
      testing=triples_factory_test,
  )

  # Train the TransE model
  pipeline_result = pipeline(
      dataset=training_dataset,
      model=TransE,
      model_kwargs=dict(embedding_dim=embedding_dim),  # Adjust embedding dimension as needed
      training_kwargs=dict(num_epochs=num_epochs),  # Adjust number of epochs as needed
  )

  # Access the trained model
  model = pipeline_result.model

  # Get entity embeddings
  entity_embeddings = model.entity_representations[0](indices=None).detach().numpy()
  # Print or use the entity embeddings
  #print(entity_embeddings)
  return entity_embeddings, triples_factory_train

# Model training by calling the TransE_train() function
entity_embeddings, triples_factory_train = TransE_train(training_data_path, test_data_path, embedding_dim=50, num_epochs=100)

def cosine_similarity(entity_id1, entity_id2, entity_embeddings):
    """Calculates the cosine similarity between two entity embeddings.

    Args:
        entity_id1: The ID of the first entity.
        entity_id2: The ID of the second entity.
        entity_embeddings: A NumPy array containing all entity embeddings.

    Returns:
        The cosine similarity between the two entity embeddings.
    """
    embedding1 = entity_embeddings[entity_id1]
    embedding2 = entity_embeddings[entity_id2]
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Example usage: calculate social proximity/entity proximity for any given entity pairs:
def social_proximity_given_entity_pairs(entity_pairs, entity_embeddings, triples_factory):
  # Get tensor of entity identifiers
  entity_ids = torch.as_tensor(triples_factory.entities_to_ids(entity_pairs))
  similarity = cosine_similarity(entity_ids[0], entity_ids[1], entity_embeddings)
  print(f"Cosine similarity between entities {entity_pairs[0]} and {entity_pairs[1]}: {similarity}")

# Function calling
social_proximity_given_entity_pairs(entity_pairs = ["Alice", "Bob"], entity_embeddings=entity_embeddings, triples_factory=triples_factory_train) # Calculate entity proximity score between Alice and Bob
social_proximity_given_entity_pairs(entity_pairs = ["Alice", "Charlie"], entity_embeddings=entity_embeddings, triples_factory=triples_factory_train) #Calculate entity proximity score between Alice and Charlie