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

# Train the TransE model using the training data
def TransE_train(training_data_path, embedding_dim, num_epochs):
  # Load your training and testing data from a CSV file
  df_train = pd.read_csv(training_data_path)

  # Create a TriplesFactory from your DataFrame
  triples_factory_train = TriplesFactory.from_labeled_triples(
      triples=df_train[['subject', 'predicate', 'object']].values,
  )

  # Create an EagerDataset
  training_dataset = EagerDataset(
      training=triples_factory_train,
      validation=None,  # Provide None for validation
      testing=triples_factory_train, # Provide training data for testing, even though evaluation is not required for calculating social proximity. The PyKEEN library's high-level API is built around the training-evaluation cycle, and the pipeline function inherently includes evaluation.
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
entity_embeddings, triples_factory_train = TransE_train(training_data_path, embedding_dim=50, num_epochs=100)

# Example usage: calculate social proximity/entity proximity for any given entity pairs:
def social_proximity_given_entity_pairs(entity_pairs, entity_embeddings, triples_factory):
  """Calculates the cosine similarity between two entity embeddings.

      Args:
          entity_pairs: A list containing two entity names (e.g., ["Alice", "Bob"]).
          entity_embeddings: A NumPy array containing all entity embeddings.
          triples_factory: The TriplesFactory object for the training data

      Returns:
          The cosine similarity between the two entity embeddings.
  """
  # Get tensor of entity identifiers
  entity_ids = torch.as_tensor(triples_factory.entities_to_ids(entity_pairs))
  embedding1 = entity_embeddings[entity_ids[0]]
  embedding2 = entity_embeddings[entity_ids[1]]
  entity_proximity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
  print(f"Social proximity between entities {entity_pairs[0]} and {entity_pairs[1]}: {entity_proximity}")
  return entity_proximity

# Function calling
sp_1 = social_proximity_given_entity_pairs(entity_pairs = ["Alice", "Bob"], entity_embeddings=entity_embeddings, triples_factory=triples_factory_train) # Calculate entity proximity score between Alice and Bob
sp_2 = social_proximity_given_entity_pairs(entity_pairs = ["Alice", "Charlie"], entity_embeddings=entity_embeddings, triples_factory=triples_factory_train) #Calculate entity proximity score between Alice and Charlie