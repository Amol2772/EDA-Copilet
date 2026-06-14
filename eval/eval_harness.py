import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_core import agent, extract_text

TEST_CASES = [
    {
        "query": "How many rows and columns does the dataset have?",
        "expected_keywords": ["5851", "16"],
        "description": "Dataset shape"
    },
    {
        "query": "Which vehicle type has the highest average daily rate?",
        "expected_keywords": ["van"],
        "description": "Top vehicle type by rate"
    },
    {
        "query": "How many missing values does rating have?",
        "expected_keywords": ["501"],
        "description": "Missing value count"
    },
    {
        "query": "What is the average daily rate across all vehicles?",
        "expected_keywords": ["93"],
        "description": "Mean rate.daily"
    },
    {
        "query": "Plot rate.daily as a histogram",
        "expected_keywords": ["histogram", "daily"],
        "description": "Plotting tool triggered"
    },
    {
        "query": "Is there a correlation between renterTripsTaken and rating?",
        "expected_keywords": ["r =", "p-value", "correlation"],
        "description": "Correlation tool triggered"
    },
    {
        "query": "Detect outliers in rate.daily",
        "expected_keywords": ["334", "outliers", "207.50"],
        "description": "Outlier tool triggered"
    },
    {
        "query": "Using SQL, how many vehicles are there per fuel type?",
        "expected_keywords": ["ELECTRIC", "GASOLINE"],
        "description": "SQL tool triggered"
    },
    {
        "query": "What are the top 3 vehicle makes by number of listings?",
        "expected_keywords": ["Tesla", "Toyota", "BMW"],
        "description": "Top vehicle makes"
    },
    {
        "query": "What is the maximum daily rate in the dataset?",
        "expected_keywords": ["1500"],
        "description": "Max rate.daily"
    },
]

def run_eval():
    passed = 0
    failed = 0
    results = []

    print(f"\nRunning {len(TEST_CASES)} eval cases...\n{'='*60}")

    for i, test in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] {test['description']}")
        print(f"Query: {test['query']}")
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": test["query"]}]})
            response = extract_text(result)
            response_lower = response.lower()

            hits = [kw for kw in test["expected_keywords"] if kw.lower() in response_lower]
            success = len(hits) == len(test["expected_keywords"])

            if success:
                print(f"✅ PASS — keywords found: {hits}")
                passed += 1
            else:
                missed = [kw for kw in test["expected_keywords"] if kw.lower() not in response_lower]
                print(f"❌ FAIL — missing keywords: {missed}")
                print(f"   Response preview: {response[:200]}")
                failed += 1

            results.append({"test": test["description"], "pass": success, "response": response[:300]})

        except Exception as e:
            print(f"💥 ERROR — {type(e).__name__}: {str(e)[:200]}")
            failed += 1
            results.append({"test": test["description"], "pass": False, "response": str(e)[:300]})

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{len(TEST_CASES)} passed ({100*passed//len(TEST_CASES)}% success rate)")
    print(f"{'='*60}\n")
    return results

if __name__ == "__main__":
    run_eval()