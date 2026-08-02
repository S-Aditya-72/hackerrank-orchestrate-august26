from data_pipeline import ContextBuilder
import json

try:
    print("Loading datasets...")
    builder = ContextBuilder()
    
    print("Calculating scores...")
    builder.calculate_affinity_scores()
    
    # Let's dynamically get the first message_id from the messages dataframe!
    first_message_id = builder.messages['message_id'].iloc[0]
    print(f"Fetching context for {first_message_id}...")
    
    context = builder.get_message_context(first_message_id) 
    
    print("SUCCESS! Here is the context:")
    print(json.dumps(context, indent=2, default=str)) # default=str handles timestamps
except Exception as e:
    import traceback
    print(f"FAILED: {e}")
    traceback.print_exc()