### 1. Why is it important to split the PDF text into chunks before embedding?

Splitting the PDF into chunks makes retrieval more accurate and efficient. An embedding represents the meaning of the text it receives. If a long document contains many unrelated topics, one embedding cannot represent every section precisely. By creating embeddings for smaller chunks, the vector database can compare the user's question with focused pieces of the document and retrieve the sections that are most relevant.

Chunking also helps the language model receive a manageable amount of context instead of the entire PDF. A small overlap between chunks reduces the chance that important information is lost when a sentence or idea falls across a chunk boundary.

### 2. What could happen if the entire document were embedded as a single vector?

If the entire PDF were represented by one vector, the embedding would become a broad summary of the document instead of a precise representation of individual sections. A question about one specific detail could still retrieve only that same whole-document vector, so the system would have no way to identify which part of the PDF is most relevant.

It would also force the application to send much more text to the language model, increasing token usage, cost, and the risk of exceeding the model's context limit. Retrieval quality would therefore be worse, especially for long documents containing several topics.
