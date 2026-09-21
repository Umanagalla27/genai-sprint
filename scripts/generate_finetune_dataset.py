import os
import sys

# Configure UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.finetuning.data_prep import FineTuningDataPipeline

# Curated dataset of natural language queries mapped to exact structured JSON filter contracts
RAW_EXAMPLES = [
    ("Show me running shoes from Nike under $120 that are in stock", {"category": "shoes", "brand": "Nike", "price_max": 120.0, "in_stock": True, "sort_by": "relevance"}),
    ("Find cheap Sony wireless headphones under 50 dollars", {"category": "electronics", "brand": "Sony", "price_max": 50.0, "sort_by": "price_asc"}),
    ("Looking for Apple laptops between 800 and 1500 dollars", {"category": "computers", "brand": "Apple", "price_min": 800.0, "price_max": 1500.0, "sort_by": "relevance"}),
    ("Show top rated gaming keyboards in stock", {"category": "keyboards", "in_stock": True, "sort_by": "rating_desc"}),
    ("Find Adidas hoodies over 60 bucks", {"category": "apparel", "brand": "Adidas", "price_min": 60.0, "sort_by": "relevance"}),
    ("Puma soccer balls in stock under 30", {"category": "sports", "brand": "Puma", "price_max": 30.0, "in_stock": True, "sort_by": "relevance"}),
    ("Dell 4K monitors sorted from lowest price", {"category": "monitors", "brand": "Dell", "sort_by": "price_asc"}),
    ("Samsung smartphones with price above 500", {"category": "smartphones", "brand": "Samsung", "price_min": 500.0, "sort_by": "relevance"}),
    ("In stock Levi's jeans under 80 dollars", {"category": "apparel", "brand": "Levi's", "price_max": 80.0, "in_stock": True, "sort_by": "relevance"}),
    ("Anker power banks under 40 dollars", {"category": "accessories", "brand": "Anker", "price_max": 40.0, "sort_by": "relevance"}),
    ("Logitech wireless mouse priced between 25 and 75", {"category": "electronics", "brand": "Logitech", "price_min": 25.0, "price_max": 75.0, "sort_by": "relevance"}),
    ("Under Armour athletic shorts in stock", {"category": "apparel", "brand": "Under Armour", "in_stock": True, "sort_by": "relevance"}),
    ("Bose noise cancelling headphones over 250", {"category": "audio", "brand": "Bose", "price_min": 250.0, "sort_by": "price_desc"}),
    ("Casio digital watches under 50 in stock", {"category": "watches", "brand": "Casio", "price_max": 50.0, "in_stock": True, "sort_by": "relevance"}),
    ("KitchenAid stand mixers sorted by price high to low", {"category": "kitchen", "brand": "KitchenAid", "sort_by": "price_desc"}),
    ("Garmin GPS fitness watches between 150 and 300 dollars", {"category": "wearables", "brand": "Garmin", "price_min": 150.0, "price_max": 300.0, "sort_by": "relevance"}),
    ("North Face winter jackets in stock", {"category": "outerwear", "brand": "North Face", "in_stock": True, "sort_by": "relevance"}),
    ("Lego Star Wars sets under 100", {"category": "toys", "brand": "Lego", "price_max": 100.0, "sort_by": "relevance"}),
    ("Ray-Ban sunglasses under 150 dollars", {"category": "accessories", "brand": "Ray-Ban", "price_max": 150.0, "sort_by": "relevance"}),
    ("Canon DSLR cameras over 600 in stock", {"category": "cameras", "brand": "Canon", "price_min": 600.0, "in_stock": True, "sort_by": "price_desc"}),
]

def main():
    print("=" * 65)
    print("📦 Generating & Validating Fine-Tuning Dataset (ChatML)")
    print("=" * 65)

    pipeline = FineTuningDataPipeline(val_split_ratio=0.2, seed=42)
    
    formatted_records = [
        pipeline.create_chatml_record(query, target_filter)
        for query, target_filter in RAW_EXAMPLES
    ]

    result = pipeline.process_and_save(
        records=formatted_records,
        train_path="data/finetune_train.jsonl",
        val_path="data/finetune_val.jsonl",
    )

    print(f"Total Records Validated: {result.total_records}")
    print(f"Training Split:          {result.train_count} samples (data/finetune_train.jsonl)")
    print(f"Validation Split:        {result.val_count} samples (data/finetune_val.jsonl)")
    print(f"Dataset Valid:           {'✅ Yes' if result.is_valid else '❌ No'}")

    if result.errors:
        print("\nErrors encountered:")
        for err in result.errors:
            print(f"  - {err}")

    print("=" * 65)


if __name__ == "__main__":
    main()
