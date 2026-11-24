#!/usr/bin/env python3
"""Run the unchanged partner backend model_service against its active model files."""
from backend.model_service import model_service

QUERY = "Madhapur mein jo mall hai uske peeche wali building ke paas ek ATM hai, wahan jaana hai."

if __name__ == '__main__':
    model_service.load_models()
    out = model_service.predict(QUERY)
    print({k: out.get(k) for k in ('intent','intent_confidence','tokens','bio_tags','entities')})
    labels = {(e.get('text'), e.get('label')) for e in out.get('entities', [])}
    required = {
        ('Madhapur','CONTEXT'), ('mall','REF'), ('uske peeche','REL'),
        ('wali building','DESC'), ('ke paas','REL'), ('ATM','DEST'), ('wahan','CONTEXT')
    }
    if not required.issubset(labels):
        raise SystemExit(f'Regression case failed. Missing: {sorted(required-labels)}')
    print('PASS: the unchanged backend model_service loaded the CRF checkpoint and extracted all target spans.')
