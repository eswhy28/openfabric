import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, GPT2Tokenizer, GPT2LMHeadModel
import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from openfabric_pysdk.utility import SchemaUtil
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import Ray, State

# Initialize models and tokenizers
pubmed_bert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
pubmed_bert_model = AutoModelForQuestionAnswering.from_pretrained('allenai/scibert_scivocab_uncased')

scibert_tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
scibert_model = AutoModelForQuestionAnswering.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-large')

nlp = spacy.load("en_core_web_md")
nlp.add_pipe('sentencizer')

question_words = set(["what", "how", "who", "where", "when", "why", "which"])

############################################################
# Callback function called on update config
############################################################



def is_question(sentence):
    # Check if the sentence ends with a question mark
    if sentence.strip().endswith("?"):
        return True

    # Additional checks using SpaCy's NLP features
    doc = nlp(sentence)
    # Check for auxiliary verbs followed by a subject, which is a common structure in questions
    for token in doc:
        if token.dep_ == "aux" and token.head.dep_ == "nsubj":
            return True

    # Check for specific question words, including creative phrasings
    question_indicators = set(["what", "how", "who", "where", "when", "why", "which", "imagine", "suppose", "if", "could", "would"])
    if any(token.lower_ in question_indicators for token in doc):
        return True

    return False



def attempt_model_response(tokenizer, model, question):
    # Check if the tokenizer and model are initialized
    if tokenizer is None or model is None:
        print("Tokenizer or model is not initialized.")
        return None

    try:
        # Encode the question using the tokenizer. This process includes tokenization,
        # adding special tokens, and converting tokens to their corresponding IDs.
        inputs = tokenizer.encode_plus(question, add_special_tokens=True, return_tensors="pt")

        # Move input tensors to the same device as the model to avoid errors related to device mismatch
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Pass inputs to the model. This will return the start and end logits for the answer.
        outputs = model(**inputs)
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

        # Identify the most likely start and end positions of the answer in the input sequence
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Extract the answer tokens from the input sequence
        answer_tokens = inputs["input_ids"][0][answer_start:answer_end]

        # Check if the answer is meaningful (not just special tokens)
        if len(answer_tokens) <= 3:  # e.g., just [CLS] and [SEP]
            return None

        # Convert the tokens to a string and remove special tokens like [CLS] and [SEP]
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer_tokens))
        return answer.replace('[CLS]', '').replace('[SEP]', '').strip()
    except Exception as e:
        # Print error message and return None if an exception occurs
        print(f"Error during model response generation: {e}")
        return None



def get_best_response(responses):
    # Filter out any None or empty responses
    valid_responses = [resp for resp in responses if resp]

    if not valid_responses:
        return None

    # Select the response with the most information (longest response)
    return max(valid_responses, key=len)


def generate_response(question):
    # Determine the nature of the question and set the temperature accordingly
    temperature = determine_temperature(question)

    # Get responses from specialized models
    pubmed_response = attempt_model_response(pubmed_bert_tokenizer, pubmed_bert_model, question)
    scibert_response = attempt_model_response(scibert_tokenizer, scibert_model, question)

    # Evaluate and select the best specialized model response
    best_bert_response = get_best_response([pubmed_response, scibert_response])

    # Validate the specialized model response
    validated_bert_response = validate_and_format_response(best_bert_response, question)

    # If validated response is satisfactory, return it; otherwise, use GPT-2
    if is_valid_response(validated_bert_response):
        return validated_bert_response
    else:
        return generate_gpt2_response(question, temperature)

def determine_temperature(question):
    # This function determines the 'temperature' for the language model's response generation.
    # The temperature is a parameter that controls the randomness of the output.
    # A lower temperature makes the model more deterministic and vice versa.

    # If the question is factual, implying it requires a precise and specific answer,
    # set a lower temperature (0.65). This makes the model's responses more predictable and factual.
    if is_factual_question(question):
        return 0.65

    # If the question is creative, implying it allows for more inventive or imaginative responses,
    # set a higher temperature (0.75). This gives the model more freedom to generate diverse and creative answers.
    elif is_creative_question(question):
        return 0.75

    # For all other types of questions, use a default temperature (0.7).
    # This provides a balance between randomness and determinism in the model's responses.
    return 0.7


def is_valid_response(response):
    if not response or response == "I'm not sure about this. Could you rephrase or ask another question?":
        return False
    return True


def generate_gpt2_response(question, temperature):
    try:
        # Encode the question for GPT-2 model and generate a response
        # based on the specified temperature. The temperature affects
        # the randomness of the response.
        inputs = gpt2_tokenizer.encode(question, return_tensors="pt")
        outputs = gpt2_model.generate(inputs, max_length=250, min_length=10,
                                      num_return_sequences=1, pad_token_id=gpt2_tokenizer.eos_token_id,
                                      no_repeat_ngram_size=3, early_stopping=True, temperature=temperature)

        # Decode the generated response, ensuring it forms a complete sentence
        response = format_complete_sentence(gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True))

        # Trim the response to remove content after the second question mark if present
        return trim_before_sentence_with_second_question(response)
    except Exception as e:
        # Handle any exceptions during response generation and return a default message
        print(f"Error during GPT-2 model response generation: {e}")
        return "I'm not sure about this. Could you rephrase or ask another question?"


# Function to compute cosine similarity
def cosine_similarity_between_texts(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))[0][0]

def validate_and_format_response(response, question):
    # Remove potential leading/trailing whitespac
    if response is None or question is None:
        return "I'm not sure about this. Could you rephrase or ask another question?"

    # Check for direct similarity using cosine similarity
    similarity = cosine_similarity_between_texts(response, question)
    if similarity > 0.8:  # Consider tweaking the threshold based on testing
        return "I'm not sure about this. Could you rephrase or ask another question?"

    # Split the response and the question into words
    response_words = response.split()
    question_words = question.split()

    # Check the length difference between the response and the question
    length_difference = abs(len(response_words) - len(question_words))

    # If the length difference is within the threshold (+/- 2 words), ask for rephrasing
    if length_difference <= 2:
        return "I'm not sure about this. Could you rephrase or ask another question?"

    # Split the response at the "\Q" marker
    parts = response.split("\n\nQ:")
    relevant_response = parts[0].strip() if parts else ""

    # Check if the relevant response contains the initial question
    if relevant_response.startswith(question.strip()):
        # Return the response part after the initial question
         get_best_statemnt = relevant_response[len(question.strip()):].strip()
         final_response = trim_before_sentence_with_second_question(get_best_statemnt)
         return final_response

    else:
        final_response = trim_before_sentence_with_second_question(relevant_response)
        return final_response

def format_complete_sentence(response):
    # Remove unnecessary characters and excessive whitespace
    response = re.sub(r'\n+', '\n', response)  # Replace multiple newlines with a single newline
    response = re.sub(r'\s+', ' ', response)   # Replace multiple spaces with a single space
    response = re.sub(r'\.{2,}', '.', response)  # Replace multiple periods with a single period

    response = response.strip()  # Remove leading and trailing whitespace

    # Find the last period in the response to ensure it ends with a complete sentence
    last_period_index = response.rfind('.')
    if last_period_index != -1 and last_period_index < len(response) - 1:
        # Return the response up to the last full stop, plus one to include the period itself
        return response[:last_period_index + 1]
    else:
        # If there's no period, or it's the last character, return the whole response
        return response


def is_factual_question(question):
    # List of keywords often found in factual questions
    factual_keywords = set(["who", "what", "when", "where", "how much", "how many"])
    question_lower = question.lower()

    # Check if any factual keyword is in the question
    return any(keyword in question_lower for keyword in factual_keywords)


def is_creative_question(question):
    # List of keywords or phrases often found in creative questions
    creative_keywords = set(["imagine", "what if", "suppose", "would it be possible", "could we", "invent", "create"])
    question_lower = question.lower()

    # Check if any creative keyword is in the question
    return any(keyword in question_lower for keyword in creative_keywords)

def trim_before_sentence_with_second_question(response):
    # Split response into sentences
    sentences = re.split(r'(?<=[.!?])\s+', response)

    # Count questions and keep track of sentences to include
    question_count = 0
    sentences_to_include = []

    for sentence in sentences:
        if '?' in sentence:
            question_count += 1
            if question_count == 2:
                break
        sentences_to_include.append(sentence)

    # Combine the sentences to form the trimmed response
    trimmed_response = ' '.join(sentences_to_include).strip()
    return trimmed_response

#def config(configuration: Dict[str, ConfigClass], state: State):
#    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []
    for text in request.text:
        if not text.strip():
            response = "Please enter a valid scientific question."
        else:
            # Directly passing the question to generate_response
            response = generate_response(text)
        output.append(response)

    return SchemaUtil.create(SimpleText(), dict(text=output))


