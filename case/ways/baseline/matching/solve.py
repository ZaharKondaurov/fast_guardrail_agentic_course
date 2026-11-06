"""Classify prompts using a dictionary of learned word n-grams."""
import json
import re
import sys
import unicodedata
from pathlib import Path

RULES_PATH = Path(__file__).with_name("rules.json")
WORD = re.compile(r"[^\W_]+")


def ngrams(text: str):
    words = WORD.findall(unicodedata.normalize("NFKC", text).lower())
    for index, word in enumerate(words):
        yield word
        if index + 1 < len(words):
            yield " ".join(words[index:index + 2])
        if index + 2 < len(words):
            yield " ".join(words[index:index + 3])
        if index + 3 < len(words):
            yield " ".join(words[index:index + 4])


def load_rules(path: Path = RULES_PATH) -> set[str]:
    return set(json.loads(path.read_text(encoding="utf-8")))


def classify(text: str, rules: set[str]) -> int:
    return int(any(phrase in rules for phrase in ngrams(text)))


def load_classifier(path: Path = RULES_PATH):
    rules = load_rules(path)

    def predict(text):
        return classify(text, rules)

    return predict


if __name__ == "__main__":
    task = json.load(sys.stdin)
    print(json.dumps({"answer": str(load_classifier()(task["input"]))}))
