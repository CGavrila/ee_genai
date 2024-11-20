# Initial requirements
Improve this simple RAG system to answer questions about Equal Experts
- you will need to download  a small Llama model - llama-2-7b-chat.Q4_K_M . You can get it from https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf Then put it into the resources directory
- We have supplied text data about EE scarped from our website in a csv file 'ee_case_studies'
- We have supplied a simple RAG tool (milvus-lite)

Tasks:
- Change the embeddings from a random  vector (already implemented)
- Add tests to validate the model output. 
- Right now query_rag only locates relevant text sections for the query. Using Llama-2, implement full text answers to the test questions you have created in the previous step. 
- Can you improve the chunking for the RAG?

Questions:
- How else would you improve this RAG system? 

Some notes and tips:
- We wanted to supply this exercise in docker, but the model just ran too slow.  
- You'll need llama_cpp to work. On a Mac M2 to install this in the environment I needed
CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64" pip install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python==0.2.11

# Feedback on the exercise

As far as my understanding goes, this exercise will go out to candidates for GenAI roles at EE. James Donovan has asked me to complete the exercise and provide feedback. I am happy to explore things further, of course!

Overall, I found this to be a great exercise to test knowledge of RAG/LLMs. The task is relatively simple, but the implementation requires a good understanding of the main concepts related to GenAI. It is also fun!

I don't exactly know the intent of the exercise, so take the following with a grain of salt:
- The initial code is quite opinionated, so it feels like the exercise is designed to be completed in a particular way. I feel like a large amount of my time was spent trying to understand the intent, rather than solving the problem. 
    - As an example, why was `embeddings=True` set for Llama? Am I meant to use Llama for embeddings as well or the provided MiniLM model?
    - The sample questions confused me to a certain extent. Are the sample questions supposed to be used as test cases/for the solution? Or are they purely used as examples?
- The README should be a bit clearer about the expected end result, maybe provide some basic examples. For example, it refers to the RAG part of the exercise as 'the model', while I consider that to be just one part of the overall solution.
- The CSV is slightly messy (missing data, duplicate sample questions etc.), which to be fair is typical in real-world data projects. Not completely sure if that was intentional or not.
- How involved is the solution expected to be? There are many things that could be tried or improved, of course. More on this below.
- What sort of questions is the exercise expected to be able to answer? My solution will only answer questions about individual case studies, and will likely not work for general questions (e.g. "What types of projects does EE work on?") very well.

# My implementation

As I mentioned above, this is simple for a prototype, but can be extended in many ways. Happy to discuss!

## Running instructions
```
pip install -r requirements.txt
python -m src.rag  # Build the RAG database
python -m src.model  # Ask questions!

python -m pytest test/test_rag.py
python -m pytest test/test_model.py  # Takes a bit longer.
```

## RAG
- I used the MiniLM model for embeddings, as it was already provided and reasonablydoes the job.
- I did not use the `title` from the dataset, as it didn't feel it would add a lot of value, but this is trivial to add and generally speaking should improve the results in a real-world solution.
- In terms of chunking, I implemented a small algorithm that tries to "fill" the context window of the MiniLM model to the 256 tokens mentioned in the HF docs. The text is first split into sentences using NLTK, and then the sentences are progressively added to a 'chunk' until the context window limit is reached. My intuition was that sentences that are close to each other are more likely to be related, so this should improve the results. This can be improved further by splitting the text based on the paragraph structure in the original posts.
- I would also improve the information the chunks contain. For example, it would be good to add the `title` of the case study to the chunk, so that the model knows which case study it is from, possibly with some formatting that will emphasise this (e.g. Markdown). 
- It might also be worth pre-processing the text in some way, potentially also with an LLM. For example, to extract certain information from the text (e.g. the project type, client name etc.) and add that to the chunks, similar to GraphRAG.
- There could also be a multi-step retrieval step, where the model first retrieves the most relevant case studies, and then the most relevant information from those.
- Maybe one of the bigger improvements to my solution would be to provide the full document in the context window, rather than just the embedded chunks. Given the size of the articles, this seems doable, but might require some heuristics. 

## Model & prompt
- Enabling the GPU vastly improved the speed on my M1 Macbook.
- I set the temperature to 0.0, to make it more deterministic. A seed might be a good idea too.
- I tried to keep the prompt as simple as possible, and focused on the task at hand. I do believe a bit more explanation about its goal and role would be helpful.
- It might have the tendency to say "I don't know" when any subjective information is requested (e.g. "Tell me a cool fact about EE").
- The prompt should be versioned in a production application. In the past, I've done this by having a template, separate from the actual code.

## Tests
- As far as I see it, there are two different parts to this: the RAG part, and the LLM part (mostly the prompt, but also metaparameters). From the README, I see these are conflated, but technically, they are different parts that can fail for different reasons, hence the two tests (`test_rag` and `test_model`).
- For the RAG test, I think it's important to test that the expected document IDs are returned based on the query. So the main thing I tested here was that the expected document IDs were in the returned list. But there could be many improvements around the similarity, for example.
- For the LLM test, I think it's important to test that the output is as expected. I added a few basic tests, but there is much more that could be done here, like testing that the output is in the correct format, or similarity, sentiment etc.
- From my point of view, the tests will never be perfect, but they could definitely be improved by embracing the probabilistic nature of the system and computing scores, rather than simple boolean checks.

