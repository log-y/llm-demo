# llm-demo
Clone the repo, install the dependencies in the requirements.txt file, then run 'django manage.py runserver'

A brief summary:
- 100 million weights/biases
- Trained on 35 GB of text data (pulled from Common Crawl)
- Took $15-20 dollars to train (using cloud resources)
- Ran 1 epoch (unfortunately ran out of money I was willing to spend)

Things I learned:
- I learned the basics of the transformer architecture: how the self-attention mechanisms will analyze other tokens for relevance, how the MLP can draw connections between vector embeddings, etc
- I learned how to train models in the cloud using SSH keys
- Also gained a lot of experience coding in Python and using libraries that compiled to C/C++ for optimization

