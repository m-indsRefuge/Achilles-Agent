import pytest
from training.dataset_parser import DatasetParser

def test_dataset_parser_efficiency():
    # Mock knowledge base data
    kb_data = {
        f"id_{i}": {"text": f"This is some sample code for snippet {i}. " * 10}
        for i in range(10)
    }

    parser = DatasetParser(kb_data)

    # Test generator output
    instructions = list(parser.to_instruction_format(max_samples=5))

    assert len(instructions) == 5
    for item in instructions:
        assert "instruction" in item
        assert "input" in item
        assert "output" in item
        # Ensure output is capped
        assert len(item["output"]) <= 1024

def test_dataset_parser_filtering():
    # Mock data with short entries
    kb_data = {
        "short": {"text": "too short"},
        "long": {"text": "This is a much longer piece of text that should pass the length filter."}
    }

    parser = DatasetParser(kb_data)
    instructions = list(parser.to_instruction_format())

    # Only "long" should remain
    assert len(instructions) == 1
    assert "longer" in instructions[0]["output"]
