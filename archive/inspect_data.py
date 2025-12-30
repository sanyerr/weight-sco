import json

# Your specific path
file_path = "/home/santerikoivula/Programming/weighted-sco/Data/diplomacy_data.jsonl"

print(f"Inspecting: {file_path}")

try:
    with open(file_path, 'r') as f:
        # Read just the first line
        line = f.readline()
        game = json.loads(line)
        
        print("\n--- ROOT KEYS ---")
        print(list(game.keys()))
        
        print("\n--- PHASES CHECK ---")
        if 'phases' in game:
            print(f"Number of phases: {len(game['phases'])}")
            if len(game['phases']) > 0:
                last_phase = game['phases'][-1]
                print(f"Last phase keys: {list(last_phase.keys())}")
                if 'state' in last_phase:
                    print(f"State keys: {list(last_phase['state'].keys())}")
                    if 'scs' in last_phase['state']:
                        print(f"SCs found: {len(last_phase['state']['scs'])} powers")
                    else:
                        print("WARNING: 'scs' NOT FOUND in state!")
        else:
            print("WARNING: 'phases' key NOT FOUND!")

        print("\n--- PLAYERS CHECK ---")
        if 'players' in game:
            print(f"Players Type: {type(game['players'])}")
            print(f"Players Data (First 200 chars): {str(game['players'])[:200]}")
        else:
            print("WARNING: 'players' key NOT FOUND!")
            
except FileNotFoundError:
    print("File not found. Check path.")
except Exception as e:
    print(f"Error reading file: {e}")