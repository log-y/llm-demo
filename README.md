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

Here are some pictures of what it does:
Enter a question:
![l1](https://github.com/user-attachments/assets/bf9d413d-a4d3-4f28-9d83-0a3c82316f9f)
Get the response:
![l2](https://github.com/user-attachments/assets/cec6616a-a0d0-4f78-aedb-5598da08682f)
Look at the predicted next tokens for every step:
![l3](https://github.com/user-attachments/assets/3589bec0-1cd6-4e58-a01d-8c107326ab58)

Note: I uploaded the entire weights/parameters file using Git LFS, which only allows two 'git clones' of the file per month (otherwise it exceeds the 1GB limit). So if you're having any issue cloning and running the repository, it may be because too many other people have loaded it before.
This also is the reason I am having some trouble deploying to Vercel (vercel needs to clone the repository to host it).
