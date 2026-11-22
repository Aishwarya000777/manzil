import gzip
import hashlib
import io
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn


# =====================================================
# CONFIGURATION
# =====================================================

FEATURE_DIM = 4096

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CRF_MODEL_PATH = PROJECT_ROOT / "models" / "manzil_linear_crf.pt"
INTENT_MODEL_PATH = PROJECT_ROOT / "models" / "intent_char_tfidf_logreg.joblib"


# =====================================================
# 1. TOKEN FEATURE EXTRACTION
# Must match the original training implementation
# =====================================================

def token_features(tokens, i):
    word = tokens[i]
    lower = word.lower()

    shape = "".join(
        "A" if c.isupper()
        else "a" if c.islower()
        else "0" if c.isdigit()
        else c
        for c in word
    )

    features = [
        f"w={lower}",
        f"shape={shape}",
        f"len={min(len(word), 15)}",
        f"p1={lower[:1]}",
        f"p2={lower[:2]}",
        f"p3={lower[:3]}",
        f"s1={lower[-1:]}",
        f"s2={lower[-2:]}",
        f"s3={lower[-3:]}"
    ]

    if i == 0:
        features.append("BOS")
    else:
        features.extend([
            f"prev={tokens[i-1].lower()}",
            f"prev2={tokens[i-1].lower()[:3]}"
        ])

    if i == len(tokens) - 1:
        features.append("EOS")
    else:
        features.extend([
            f"next={tokens[i+1].lower()}",
            f"next2={tokens[i+1].lower()[:3]}"
        ])

    if i:
        features.append(
            f"bp={tokens[i-1].lower()}_{lower}"
        )

    if i < len(tokens) - 1:
        features.append(
            f"bn={lower}_{tokens[i+1].lower()}"
        )

    return features


def encode(tokens):
    matrix = np.zeros(
        (len(tokens), FEATURE_DIM),
        dtype=np.float32
    )

    for i in range(len(tokens)):
        for feature in token_features(tokens, i):
            index = (
                int(hashlib.md5(feature.encode()).hexdigest()[:8], 16)
                % FEATURE_DIM
            )
            matrix[i, index] += 1

    return torch.from_numpy(matrix)


# =====================================================
# 2. LINEAR CHAIN CRF
# =====================================================

class LinearChainCRF(nn.Module):

    def __init__(self, num_tags):
        super().__init__()

        self.em = nn.Parameter(
            torch.zeros(FEATURE_DIM, num_tags)
        )

        self.trans = nn.Parameter(
            torch.zeros(num_tags, num_tags)
        )

        self.start = nn.Parameter(
            torch.zeros(num_tags)
        )

        self.end = nn.Parameter(
            torch.zeros(num_tags)
        )

    def decode(self, x):
        emissions = x @ self.em

        score = self.start + emissions[0]
        backpointers = []

        for t in range(1, len(emissions)):
            values, indices = (
                score[:, None] + self.trans
            ).max(0)

            score = values + emissions[t]
            backpointers.append(indices)

        last = int((score + self.end).argmax())

        path = [last]

        for pointer in reversed(backpointers):
            last = int(pointer[last])
            path.append(last)

        return list(reversed(path))


# =====================================================
# 3. MODEL LOADING
# Supports both compressed and uncompressed files
# =====================================================

def load_model_file(path, loader):
    path = Path(path)

    with open(path, "rb") as file:
        raw = file.read()

    if path.suffix == ".gz":
        raw = gzip.decompress(raw)

    return loader(io.BytesIO(raw))


# =====================================================
# 4. ENTITY SPAN EXTRACTION
# =====================================================

def extract_spans(query, tokens, tags):
    offsets = []
    current = 0

    for token in tokens:
        start = query.find(token, current)

        if start < 0:
            start = current

        offsets.append((start, start + len(token)))
        current = start + len(token)

    entities = []
    start = None
    label = None

    for i in range(len(tags) + 1):
        tag = tags[i] if i < len(tags) else "O"

        prefix, entity_label = (
            (tag[:2], tag[2:])
            if "-" in tag
            else ("O", None)
        )

        if start is not None and (
            prefix == "B-" or prefix == "O" or entity_label != label
        ):
            entity_start = offsets[start][0]
            entity_end = offsets[i - 1][1]

            entities.append({
                "text": query[entity_start:entity_end],
                "label": label,
                "start": entity_start,
                "end": entity_end
            })

            start = None
            label = None

        if prefix == "B-":
            start = i
            label = entity_label

        elif prefix == "I-" and start is None:
            start = i
            label = entity_label

    return entities


# =====================================================
# 5. MANZIL ML MODEL SERVICE
# =====================================================

class ManzilModelService:

    def __init__(self):
        self.crf_model = None
        self.intent_model = None
        self.tags = None
        self.ready = False

    def load_models(self):

        if not CRF_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"CRF model not found: {CRF_MODEL_PATH}"
            )

        if not INTENT_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Intent model not found: {INTENT_MODEL_PATH}"
            )

        # Load CRF model
        checkpoint = load_model_file(
            CRF_MODEL_PATH,
            lambda b: torch.load(
                b,
                map_location="cpu",
                weights_only=False
            )
        )

        self.tags = checkpoint["tags"]

        self.crf_model = LinearChainCRF(
            len(self.tags)
        )

        self.crf_model.load_state_dict(
            checkpoint["state_dict"]
        )

        self.crf_model.eval()

        # Load intent classifier
        self.intent_model = load_model_file(
            INTENT_MODEL_PATH,
            joblib.load
        )

        self.ready = True

        print("Manzil intent classifier loaded successfully.")
        print("Manzil CRF NER model loaded successfully.")

    def predict(self, query):

        if not self.ready:
            raise RuntimeError(
                "ML models are not loaded."
            )

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        tokens = query.split()



        # Intent classification
        vectorizer = self.intent_model["vectorizer"]
        classifier = self.intent_model["classifier"]

        intent_vector = vectorizer.transform([query])

        intent = classifier.predict(intent_vector)[0]

        intent_probabilities = None
        intent_confidence = None
        intent_margin = None

        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba(
                intent_vector
            )[0]

            intent_probabilities = {
                str(label): float(probability)
                for label, probability in zip(
                    classifier.classes_,
                    probabilities
                )
            }

            sorted_probabilities = sorted(
                probabilities,
                reverse=True
            )

            intent_confidence = float(sorted_probabilities[0])

            intent_margin = float(
                sorted_probabilities[0] -
                sorted_probabilities[1]
            )


        
        # CRF entity extraction
        encoded_tokens = encode(tokens)

        with torch.no_grad():
            predicted_ids = self.crf_model.decode(encoded_tokens)

        bio_tags = [
            self.tags[index]
            for index in predicted_ids
        ]

        entities = extract_spans(
            query,
            tokens,
            bio_tags
        )

        return {
            "query": query,
            "intent": str(intent),
            "intent_confidence": intent_confidence,
            "intent_probabilities": intent_probabilities,
            "intent_margin": intent_margin,
            "tokens": tokens,
            "bio_tags": bio_tags,
            "entities": entities

        }


# =====================================================
# 6. SHARED MODEL SERVICE INSTANCE
# =====================================================

model_service = ManzilModelService()
