import json
import os
from src.finetuning.data_prep import FineTuningDataPipeline


def test_create_chatml_record():
    pipeline = FineTuningDataPipeline()
    record = pipeline.create_chatml_record(
        user_query="Find cheap Nike shoes under 50",
        target_json_filter={"category": "shoes", "brand": "Nike", "price_max": 50.0},
    )

    assert "messages" in record
    assert len(record["messages"]) == 3
    assert record["messages"][0]["role"] == "system"
    assert record["messages"][1]["role"] == "user"
    assert record["messages"][2]["role"] == "assistant"
    
    # Assistant content must parse into JSON
    assistant_json = json.loads(record["messages"][2]["content"])
    assert assistant_json["brand"] == "Nike"
    assert assistant_json["price_max"] == 50.0


def test_validate_record_rejects_malformed():
    pipeline = FineTuningDataPipeline()
    
    # Missing system message
    bad_record = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    }
    is_valid, err = pipeline.validate_record(bad_record)
    assert not is_valid
    assert "Expected exactly 3 messages" in err


def test_validate_record_rejects_unparseable_json():
    pipeline = FineTuningDataPipeline()
    bad_json_record = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "Not valid JSON {"},
        ]
    }
    is_valid, err = pipeline.validate_record(bad_json_record)
    assert not is_valid
    assert "valid, parseable JSON" in err
