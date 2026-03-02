from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Tokenizer helps to see the conversion of text -> tokens
tokenizer = classifier.tokenizer

print("AI Sentiment Analyzer (Type 'exit' to quit)")
print("-------------------------------------------")

while True:
    text = input("Enter text:").strip()

    if not text:
        print("Please enter some text!")
        continue

    if text.lower() == "exit":
        print("\nGoodbye!")
        break

    # tokens = tokenizer.tokenize(text) - this directly gives tokens but not CLS and SEP Tokens
    # tokenizer(text) gives us input ids aka tokens ids and attention mask
    encoded = tokenizer(text)
    # print("\nFull Encoding:")
    # print(encoded)
    # tokenizer.convert_ids_to_tokens - turns token ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    print("\nTokens:")
    print(tokens)

    print("\nToken ids:")
    print(encoded["input_ids"])

    result = classifier(text)
    label = result[0]["label"]
    score = result[0]["score"]

    if score < 0.6:
        label = "NEUTRAL"

    print("\nSentiment:", label)
    print("Confidence:", round(score * 100, 2), "%")
    print("\n")
