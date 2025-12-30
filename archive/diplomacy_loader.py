import json
import os
from tqdm import tqdm

def load_diplomacy(jsonl_path, min_phases=10):
    dataset = []
    player_map = {} # Maps user_id (str) -> ID (int)

    print(f"Parsing {jsonl_path}...")
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(tqdm(f)):
            try:
                game = json.loads(line)
                
                # Filter short games
                if len(game.get('phases', [])) < min_phases: continue
                
                # Extract Scores (SC Count)
                # Note: 'scs' is usually in the last phase's state
                scs = game['phases'][-1]['state']['scs']
                players_raw = game['players']
                
                game_results = []
                
                # --- FIX STARTS HERE ---
                # We normalize 'players' into a list, regardless of whether it's a list or dict in the file
                iterator = []
                if isinstance(players_raw, list):
                    iterator = players_raw
                elif isinstance(players_raw, dict):
                    for p_power, p_data in players_raw.items():
                        if isinstance(p_data, dict):
                            # Copy data and ensure 'power' is set
                            data_copy = p_data.copy()
                            data_copy['power'] = p_power
                            iterator.append(data_copy)
                # --- FIX ENDS HERE ---

                # Now process the normalized list
                for p_data in iterator:
                    if 'user_id' in p_data and 'power' in p_data:
                        pid = p_data['user_id']
                        power = p_data['power']
                        score = len(scs.get(power, [])) # 0 if wiped out
                        
                        # Get/Create Integer ID
                        if pid not in player_map:
                            player_map[pid] = len(player_map) + 1
                        
                        game_results.append((score, player_map[pid]))

                if len(game_results) < 2: continue

                # Sort by score descending
                game_results.sort(key=lambda x: x[0], reverse=True)
                
                # Calculate ranks (handling ties)
                ranks = []
                for i in range(len(game_results)):
                    if i > 0 and game_results[i][0] == game_results[i-1][0]:
                        ranks.append(ranks[-1])
                    else:
                        ranks.append(i)

                # Generate Weighted Pairs
                for i in range(len(game_results)):
                    for j in range(i + 1, len(game_results)):
                        if game_results[i][0] > game_results[j][0]:
                            winner = game_results[i][1]
                            loser = game_results[j][1]
                            weight = (1.0 / (ranks[i] + 1)) + (1.0 / (ranks[j] + 1))
                            dataset.append((winner, loser, weight))

            except Exception as e:
                # Debug print only for the first error
                if line_num == 0:
                    print(f"DEBUG: Error on first line: {e}")
                continue

    return dataset, len(player_map)

if __name__ == "__main__":
    # Point this to your actual file location
    # Note: Ensure this path matches where you saved the file!
    FILE_PATH = "/home/santerikoivula/Programming/weighted-sco/Data/diplomacy_data.jsonl"
    
    if os.path.exists(FILE_PATH):
        data, n_players = load_diplomacy(FILE_PATH)
        print(f"SUCCESS: Loaded {len(data)} pairs from {n_players} players.")
    else:
        print(f"File {FILE_PATH} not found.")