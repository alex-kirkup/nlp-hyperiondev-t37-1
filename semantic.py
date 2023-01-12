import spacy

# function to run all code with different language models
def run_spacy_tasks(language_model: str):

    nlp = spacy.load(language_model)

    tokens = nlp('cat apple monkey banana fruit')
    
    print("Word comparisons:")
    for token1 in tokens:

        for token2 in tokens:

            print("  ", token1.text, token2.text, token1.similarity(token2))

    sentence_to_compare = "Why is my cat on the car"
    print(f"Sentence comparisons with {sentence_to_compare}")

    sentences = ["where did my dog go",
        "Hello, there is my car",
        "I\'ve lost my car in my car",
        "I\'d like my boat back",
        "I will name my dog Diana",
    ]
    model_sentence = nlp(sentence_to_compare)

    for sentence in sentences:
        similarity = nlp(sentence).similarity(model_sentence)
        print("  ", sentence + " - ", similarity)


print("Spacy comparisons using en_core_web_sm language model")
print("-----------------------------------------------------")
run_spacy_tasks('en_core_web_sm')

print()
print()
print("Spacy comparisons using en_core_web_md language model")
print("-----------------------------------------------------")
run_spacy_tasks('en_core_web_md')

print()
print()
print("Observations / similarities")
print("-----------------------------------------------------")
print(
"""
1. en_core_web_md language model makes significantly
   greater distinction between token differences e.g.
   'cat apple' was 0.62 using en_core_web_sm model but 
   0.20 using en_core_web_md. 
   
   In comparison, 'cat monkey' was 0.57 using en_core_web_sm 
   model but 0.59 using en_core_web_md.  

2. en_core_web_md language model notices greater similarity
   among sentences (possibly due to the 'The model you're 
   using has no word vectors loaded' warning for the 
   en_core_web_sm model). This is true for all sentences.

3. Additionally, it increases the difference in similarity
   between sentences based on the importance of the nouns in
   the sentences e.g. the similarity between 'Hello, there 
   is my car' and the comparison sentence 'Why is my cat on 
   the car' increases from 0.52 using en_core_web_sm model to
   0.80 using en_core_web_md model.
"""
)