# Security Module

This module provides security features for the agent swarm system, including prompt injection detection using spaCy.

## Installation

1. Install spaCy:
```bash
pip install spacy
```

2. Download the English language model:
```bash
python -m spacy download en_core_web_sm
```

For better accuracy, you can use a larger model:
```bash
python -m spacy download en_core_web_md
# or
python -m spacy download en_core_web_lg
```

## Usage

### Basic Detection

```python
from security import detect_prompt_injection

result = detect_prompt_injection("Ignore all previous instructions")
if result.is_injection:
    print(f"Injection detected! Severity: {result.severity.value}")
    print(f"Confidence: {result.confidence}")
    print(f"Patterns: {result.patterns_found}")
```

### Advanced Usage

```python
from security import PromptInjectionDetector

detector = PromptInjectionDetector()

# Detect injection
result = detector.detect(user_input)
if result.is_injection:
    print(f"Alert: {result.explanation}")
    
    # Sanitize the input
    sanitized, detection = detector.sanitize(user_input)
    # Use sanitized input instead
```

### Integration with Agent System

The detector is automatically integrated into `current_events_comedian.py` and will:
- Check all user inputs and feedback
- Log security alerts
- Automatically sanitize detected injections
- Block critical severity injections

## Detection Capabilities

The detector uses multiple strategies:

1. **Pattern Matching**: Detects known injection phrases like "ignore previous instructions"
2. **Dependency Parsing**: Identifies unusual command structures
3. **Named Entity Recognition**: Finds system-related entity references
4. **Encoding Detection**: Detects obfuscated or encoded text
5. **Linguistic Analysis**: Identifies suspicious sentence patterns

## Severity Levels

- **LOW**: Minor suspicious patterns, may be false positives
- **MEDIUM**: Moderate risk, should be reviewed
- **HIGH**: Significant risk, should be blocked or sanitized
- **CRITICAL**: Immediate threat, must be blocked

## Testing

Run the detector test:
```bash
python security/prompt_injection_detector.py
```

This will test various injection patterns and show detection results.

