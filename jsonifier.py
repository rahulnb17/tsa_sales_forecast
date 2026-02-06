import json

# =====================================================
# PATHS
# =====================================================
json_path = "results/all_state_model_results.json"

# Default sizes (same for all states as you said)
TRAIN_SIZE = 1429
TEST_SIZE  = 358

# =====================================================
# LOAD JSON
# =====================================================
with open(json_path, "r") as f:
    data = json.load(f)

# =====================================================
# ADD SIZES TO ALL STATES
# =====================================================
for state in data:
    
    # Only add if missing
    if "train_size" not in data[state]:
        data[state]["train_size"] = TRAIN_SIZE
    
    if "test_size" not in data[state]:
        data[state]["test_size"] = TEST_SIZE

# =====================================================
# SAVE UPDATED JSON
# =====================================================
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print("✅ Train/Test sizes added to all states.")
