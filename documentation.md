The `main.py` script is a sophisticated NLP-based system designed for processing and responding to scientific questions. It leverages advanced machine learning models and natural language processing techniques. Below is a brief overview of the key components and their functionalities:

#### 1. **Model and Tokenizer Initialization**
   - **`pubmed_bert_tokenizer`, `pubmed_bert_model`**: These are initialized with the 'allenai/scibert_scivocab_uncased' model, optimized for scientific texts.
   - **`scibert_tokenizer`, `scibert_model`**: Initialized with the 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext' model, tailored for biomedical literature.
   - **`gpt2_tokenizer`, `gpt2_model`**: Utilizes the 'gpt2-large' model, a general-purpose language model for generating text.

#### 2. **Core Functions**
   - **`is_question(sentence: str) -> bool`**: Determines if a given sentence is a question. It checks for typical question structures and keywords, including those found in creative questions.
   - **`attempt_model_response(tokenizer, model, question: str) -> str`**: Generates a response to a question using the specified tokenizer and model. Handles errors gracefully, returning `None` in case of failure.
   - **`get_best_response(responses: list) -> str`**: Chooses the best response from a list of potential answers, prioritizing the most informative one.
   - **`generate_response(question: str) -> str`**: Orchestrates response generation. It first tries to get answers from specialized models and, if unsuccessful, resorts to GPT-2.
   - **`determine_temperature(question: str) -> float`**: Sets the 'temperature' for GPT-2 responses, adjusting creativity and randomness based on the question type.
   - **`is_valid_response(response: str, question: str) -> bool`**: Validates a response to ensure it's relevant and meaningful.
   - **`generate_gpt2_response(question: str, temperature: float) -> str`**: Generates a response using GPT-2, tailored to the question's nature.
   - **`cosine_similarity_between_texts(text1: str, text2: str) -> float`**: Computes the cosine similarity between two texts, a measure of their semantic similarity.
   - **`validate_and_format_response(response: str, question: str) -> str`**: Validates and formats a response, ensuring it's relevant and well-structured.
   - **`format_complete_sentence(response: str) -> str`**: Ensures that a response forms a complete sentence, improving readability.
   - **`is_factual_question(question: str) -> bool`

**: Identifies if a question is factual based on specific keywords commonly found in such questions.
   - **`is_creative_question(question: str) -> bool`**: Determines if a question is of a creative nature, using a set of keywords indicative of imaginative or hypothetical queries.
   - **`trim_before_sentence_with_second_question(response: str) -> str`**: Trims a response to end before a second question arises within it, ensuring focus on the initial query.

#### 3. **Execution Function**
   - **`execute(request: SimpleText, ray: Ray, state: State) -> SimpleText`**: This is the main execution function called for each processing cycle. It processes input text, generates responses, and formats them into the `SimpleText` schema.
   
### Summary of System Workflow

1. **Question Identification**: When a sentence is input, the system first determines whether it is a question using `is_question`.
2. **Response Generation**: For identified questions, `generate_response` orchestrates the response generation process, leveraging specialized models for scientific queries and GPT-2 for broader or creative questions.
3. **Response Validation and Formatting**: Generated responses undergo validation and formatting to ensure relevance and clarity, employing functions like `validate_and_format_response` and `format_complete_sentence`.
4. **Cosine Similarity Analysis**: To avoid responses too similar to the input question, `cosine_similarity_between_texts` is used, enhancing the system's ability to provide informative and distinct answers.
5. **Final Output**: The processed and validated responses are returned, aligning with the structure expected by the calling context or application.


### 4. After installing the requirements with `pip install -r requirements.txt`, run the following command to download the necessary spaCy model:
python -m spacy download en_core_web_md

### 5. Video presentation of the application link: https://www.loom.com/share/589c8db8ad744080ae93f79a46f82d81?sid=02fc51ca-b32b-4e2a-afcb-fa04c4a9d351
