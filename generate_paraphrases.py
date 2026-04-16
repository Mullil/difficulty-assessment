from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto"
)

def paraphrase_with_cefr(text: str, cefr_level: str) -> str:
    cefr_descriptions = {
        "A1": "beginner level — very simple vocabulary, short sentences, basic present tense, concrete topics only",
        "A2": "elementary level — simple vocabulary, short sentences, basic tenses, everyday topics",
        "B1": "intermediate level — common vocabulary, moderate sentence complexity, main tenses, familiar topics",
        "B2": "upper-intermediate level — varied vocabulary, complex sentences, range of tenses, abstract topics",
        "C1": "advanced level — sophisticated vocabulary, complex syntax, nuanced expression, idiomatic language",
        "C2": "mastery level — extensive vocabulary, highly complex syntax, nominalisations, domain-specific terminology, dense subordination",
    }

    level_upper = cefr_level.upper()
    if level_upper not in cefr_descriptions:
        raise ValueError(f"Invalid CEFR level '{cefr_level}'. Must be one of: {', '.join(cefr_descriptions)}")

    description = cefr_descriptions[level_upper]

    prompt = f"""You are an expert linguist specialising in CEFR language proficiency levels.

    Your task is to paraphrase the following text while strictly preserving its CEFR difficulty level: {level_upper} ({description}).

    Rules:
    - Reword sentences and rephrase expressions — do not copy the original phrasing
    - Maintain the same language as the input text
    - Preserve the meaning and all key information
    - Match the vocabulary range, sentence complexity, and grammatical structures typical of {level_upper}
    - Do not simplify or elevate the difficulty — stay exactly at {level_upper}
    - Output only the paraphrased text, no commentary

    Text to paraphrase:
    {text}"""

    output = generator(
        prompt,
        max_new_tokens=512,
        temperature=0.7,
    )

    return output[0]["generated_text"].split("[/INST]")[-1].strip()


def main():
    print("=== CEFR-preserving paraphraser ===\n")

    text = input("Enter the text to paraphrase:\n> ").strip()
    if not text:
        print("No text provided.")
        return

    cefr = input("\nEnter CEFR level (A1/A2/B1/B2/C1/C2): ").strip()

    print("\nParaphrasing...\n")
    result = paraphrase_with_cefr(text, cefr)
    print("--- Paraphrase ---")
    print(result)


if __name__ == "__main__":
    main()