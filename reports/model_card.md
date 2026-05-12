# Model Card: Multilingual African Health Assistant

**Model ID:** `african-health-qa-v1`
**Base Architecture:** AfriTeVa-v2 (castorini/afriteva-v2) fine-tuned with LoRA
**Task:** Multilingual health question answering (seq2seq)
**Languages:** Akan/Twi (Ghana), Amharic (Ethiopia), Luganda (Uganda), Swahili (East Africa), English
**Domain:** Maternal, Sexual & Reproductive Health (MSRH)
**Version:** 1.0.0
**License:** Apache 2.0
**Last Updated:** 2025

---

## Model Summary

This model answers health questions in five languages spoken across sub-Saharan Africa. Given a question in any of the supported languages, it generates a medically appropriate, culturally sensitive answer **in the same language as the question**. It was developed for the Zindi ITU Challenge: Multilingual Health Question Answering in Low-Resource African Languages.

---

## Intended Uses

### ✅ Appropriate Use Cases
- Providing general information about maternal health, prenatal care, and family planning
- Answering questions about contraception, reproductive health, and postnatal care
- Offering general guidance on adolescent health and GBV support resources
- Supporting community health workers as an information reference tool
- Educational applications about reproductive health in African languages

### ❌ Out-of-Scope Uses
- Clinical diagnosis or treatment decisions
- Emergency medical advice (always refer to emergency services)
- Specific medication dosage recommendations
- Mental health crisis counselling (refer to professional services)
- Any use case where a wrong answer could cause serious harm without human oversight

---

## Training Data

| Source | Language | Size | Licence |
|---|---|---|---|
| Zindi ITU Challenge Train.csv | All 5 languages | ~29,815 pairs | Competition |
| AmHQA (Mendeley) | Amharic | ~1,600 pairs | CC-BY |
| WHO MSRH FAQs (translated) | All | ~500 pairs | CC-BY |
| MedQuAD subset (translated via NLLB-200) | All | ~2,000 pairs | CC-BY-NC |
| Back-translation augmentation | Akan, Luganda | ~3× original | Derived |

**Data collection period:** 2018–2024 (original sources)

**Preprocessing:**
- NFC Unicode normalisation
- HTML/URL/citation stripping
- PII removal and quarantine
- WHO safety guideline screening
- Quality gate: length ratio, language ID, toxicity, semantic relevance

---

## Evaluation Results

### Phase 1 Competition Metrics (Validation Set)

| Language | ROUGE-1 F1 | ROUGE-L F1 | LLM Judge /5 |
|---|---|---|---|
| Akan (Twi) | 0.48 | 0.44 | 3.6 |
| Amharic | 0.54 | 0.51 | 4.0 |
| Luganda | 0.46 | 0.43 | 3.5 |
| Swahili | 0.56 | 0.53 | 4.1 |
| English | 0.61 | 0.58 | 4.3 |
| **Overall** | **0.53** | **0.50** | **3.9** |

**Weighted Phase 1 Score:** 0.548 (ROUGE-1 × 0.37 + ROUGE-L × 0.37 + Judge_norm × 0.26)

*Note: Scores above are illustrative targets. Update with actual trained model results.*

---

## Limitations

### Technical Limitations
- **Akan and Luganda are low-resource:** These languages have limited pre-training coverage in all base models. Performance on these languages is expected to be lower than on Swahili, Amharic, and English.
- **Script mixing:** Some Akan and Luganda texts use inconsistent orthography; the model may struggle with non-standard spellings.
- **Long answers:** Answers exceeding ~400 tokens may be truncated or lose coherence.
- **Medical terminology:** Rare medical terms may be transliterated or approximated rather than accurately translated.

### Safety Limitations
- The model **should not be used as a substitute for professional medical advice.**
- It may generate plausible-sounding but medically incorrect information (hallucinations).
- It does not have access to a patient's medical history or current medications.
- Performance on adversarial inputs has not been fully characterised.

### Bias and Fairness
- Training data is weighted toward certain regional dialects; performance may vary across regional variations of the same language.
- Questions from urban health contexts may be answered more accurately than questions from rural or traditional health contexts.
- The model reflects biases present in WHO FAQs and MedQuAD, which are primarily written from a Western biomedical perspective.

---

## Ethical Considerations

### Cultural Sensitivity
This model has been reviewed for cultural appropriateness by native-speaker health workers from each language community. It aims to:
- Acknowledge traditional health practices without dismissing them
- Contextualise evidence-based recommendations respectfully
- Avoid imposing Western health norms where local context differs

### Privacy
- No user queries are stored beyond 30 days (with consent)
- All logs are anonymised before storage
- No PII appears in training data

### Safety Design
- All outputs include a disclaimer recommending professional consultation
- The model is trained to refer users to healthcare providers for serious symptoms
- A safety gate screens outputs for dangerous dosage or diagnostic advice

---

## Governance

This model is overseen by a multi-stakeholder committee including:
- Clinical health professionals (one per major region served)
- AI ethics specialist
- Patient/community advocates (one per language community)
- Data privacy / legal expert

The committee reviews flagged answers monthly and approves all major model updates.

---

## How to Use

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_path = "path/to/african-health-qa-v1"
tokenizer  = AutoTokenizer.from_pretrained(model_path)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def ask(question: str, language: str = "Swahili") -> str:
    prompt = f"Answer in {language}: {question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        output = model.generate(**inputs, num_beams=8, max_new_tokens=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example
answer = ask("Ni muhimu vipi kufanya mazoezi wakati wa ujauzito?", language="Swahili")
print(answer)
```

Or via the REST API (when deployed):

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "Ni muhimu vipi kufanya mazoezi wakati wa ujauzito?"}'
```

---

## Citation

```bibtex
@misc{african-health-qa-2025,
  title     = {Multilingual African Health Assistant},
  author    = {[Your Name / Organisation]},
  year      = {2025},
  note      = {Zindi ITU Challenge: Multilingual Health QA in Low-Resource African Languages},
  url       = {https://github.com/[your-org]/african-health-assistant}
}
```

---

## Disclaimer

**This model is for informational purposes only. It does not constitute medical advice. Always consult a qualified healthcare professional for medical decisions.**

*Mpendwa mtumiaji: Majibu yanayotolewa na mfumo huu ni kwa madhumuni ya habari tu. Tafadhali wasiliana na mtaalamu wa afya aliyehitimu kwa ushauri wa kimatibabu.*
