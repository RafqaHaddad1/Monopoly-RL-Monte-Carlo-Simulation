import random
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import gym
from gym import spaces
import numpy as np

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

class Property:
    def __init__(self, name, base_rent, house_cost, color_group):
        self.name = name
        self.base_rent = base_rent
        self.house_cost = house_cost
        self.color_group = color_group
        self.owner = None
        self.house_count = 0  # 0–4 = houses, 5 = hotel

    @property
    def has_hotel(self):
        return self.house_count == 5

    def get_rent(self):
        if self.has_hotel:
            return self.base_rent * 5  # Example multiplier
        return self.base_rent + (self.house_count * (self.base_rent // 2))

    def can_buy_house(self, player, all_properties):
        owns_group = all(
            prop.owner == player
            for prop in all_properties
            if prop.color_group == self.color_group
        )
        return (
            self.owner == player
            and owns_group
            and self.house_count < 5
            and player.money >= self.house_cost
        )

    def buy_house(self, player):
        if self.can_buy_house(player, []):  # You’ll need to pass properties list in real use
            self.house_count += 1
            player.money -= self.house_cost


class MonopolyEnv(gym.Env):
    metadata = {'render.modes': ['human']} # Gym convention

    def __init__(self, go_reward=200, start_money=1500, board_size=40, num_players=2):
        super().__init__() # Initialize Gym Env

        self.go_reward = go_reward
        self.start_money = start_money
        self.board_size = board_size
        self.jail_position = 10
        self.go_to_jail_position = 30
        self.jail_turns = 3
        self.num_players = num_players
        self.fee_positions = {
            4: 200,   # Income tax
            12: 150,  # Water bill - Often utility, let's keep as fee for now
            28: 150,  # Electric company - Often utility, let's keep as fee for now
            38: 100,  # Luxury tax
        }
        self.chance_deck = [
            {"name": "Advance to Go", "effect": self.advance_to_go},
            {"name": "Go to Jail", "effect": self.go_to_jail},
            {"name": "Bank pays you dividend", "effect": lambda p: self.adjust_money(p, 50)},
            {"name": "Pay poor tax", "effect": lambda p: self.adjust_money(p, -15)},
            # Add more as needed
        ]

        self.chest_deck = [
            {"name": "Doctor's fee", "effect": lambda p: self.adjust_money(p, -50)},
            {"name": "Income tax refund", "effect": lambda p: self.adjust_money(p, 20)},
            {"name": "Go to Jail", "effect": self.go_to_jail},
            {"name": "Advance to Go", "effect": self.advance_to_go},
            # Add more as needed
        ]

        # Set the board squares that are Chance or Chest
        self.chance_positions = {7, 22, 36}
        self.chest_positions = {2, 17, 33}
        # Define standard Monopoly property prices/rents (simplified)
        # You could load this from a file for a real game
        # Format: {position: {"price": price, "rent": rent, "name": name}}
        self.property_details = {
             1: {"price": 60, "rent": 2, "name": "Mediterranean Avenue"},
             3: {"price": 60, "rent": 4, "name": "Baltic Avenue"},
             5: {"price": 200, "rent": 25, "name": "Reading Railroad"},
             6: {"price": 100, "rent": 6, "name": "Oriental Avenue"},
             8: {"price": 100, "rent": 6, "name": "Vermont Avenue"},
             9: {"price": 120, "rent": 8, "name": "Connecticut Avenue"},
             11: {"price": 140, "rent": 10, "name": "St. Charles Place"},
             12: {"price": 150, "rent": 10, "name": "Electric Company"},
             13: {"price": 140, "rent": 10, "name": "States Avenue"},
             14: {"price": 160, "rent": 12, "name": "Virginia Avenue"},
             15: {"price": 200, "rent": 25, "name": "Pennsylvania Railroad"},
             16: {"price": 180, "rent": 14, "name": "St. James Place"},
             18: {"price": 180, "rent": 14, "name": "Tennessee Avenue"},
             19: {"price": 200, "rent": 16, "name": "New York Avenue"},
             21: {"price": 220, "rent": 18, "name": "Kentucky Avenue"},
             23: {"price": 220, "rent": 18, "name": "Indiana Avenue"},
             24: {"price": 240, "rent": 20, "name": "Illinois Avenue"},
             25: {"price": 200, "rent": 25, "name": "B. & O. Railroad"},
             26: {"price": 260, "rent": 22, "name": "Atlantic Avenue"},
             27: {"price": 260, "rent": 22, "name": "Ventnor Avenue"},
             28: {"price": 150, "rent": 10, "name": "Water Works"},
             29: {"price": 280, "rent": 24, "name": "Marvin Gardens"},
             31: {"price": 300, "rent": 26, "name": "Pacific Avenue"},
             32: {"price": 300, "rent": 26, "name": "North Carolina Avenue"},
             34: {"price": 320, "rent": 28, "name": "Pennsylvania Avenue"},
             35: {"price": 200, "rent": 25, "name": "Short Line Railroad"},
             37: {"price": 350, "rent": 35, "name": "Park Place"},
             39: {"price": 400, "rent": 50, "name": "Boardwalk"},
        }
        # Add placeholder for non-property squares
        for i in range(self.board_size):
             if i not in self.property_details:
                 self.property_details[i] = {"price": 0, "rent": 0, "name": f"Square {i}"} # GO, Jail, Taxes etc.

        # --- Action Space ---
        # Action 0: Don't Buy / Continue
        # Action 1: Buy Property (if applicable)
        self.action_space = spaces.Discrete(2)

        # --- Observation Space (State) ---
        # Simplified: Combines key info. More complex states are possible.
        # We need low/high values for each component.
        low = np.array(
            [0] * self.num_players +             # positions
            [0] * self.num_players +             # money (or -ve if bankrupt)
            [0] * self.num_players +             # in_jail flags
            [-1] * self.board_size +            # owners (-1 for no owner)
            [0]                                  # current player index
            # We could add more state here (e.g., property details player landed on)
        )
        high = np.array(
            [self.board_size - 1] * self.num_players +
            [self.start_money * 10] * self.num_players + # Arbitrarily high money limit
            [1] * self.num_players +
            [self.num_players - 1] * self.board_size +
            [self.num_players - 1]
        )
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

        self.reset()
    def adjust_money(self, player, amount):
          self.money[player] += amount
          reward = amount # Reward is the money change
          # Check for bankruptcy *immediately* after money adjustment
          done = self.money[player] < 0
          info = {"card": f"Money change: {amount}", "card_specific_desc": f" Adjusted money by {amount}."}
          # Note: We don't return the full log entry here, step() will assemble it.
          # We just return the core results of the card effect.
          return self._get_obs(), reward, done, info

    def adjust_money(self, player, amount):
        """Adjusts player money and returns reward contribution + description."""
        self.money[player] += amount
        reward_contribution = amount # Reward is the direct change
        desc = f"Adjusted money by {amount}."
        # Note: Bankruptcy check happens *later* in the main step function
        return {"reward": reward_contribution, "card_specific_desc": desc}

    def advance_to_go(self, player):
        """Moves player to GO, adds GO reward, returns reward contribution + description."""
        passed_go = self.positions[player] > 0 # Only collect if not already at GO
        self.positions[player] = 0
        reward_contribution = 0
        desc = "Advanced to GO."
        if passed_go:
            self.money[player] += self.go_reward
            reward_contribution += self.go_reward
            desc += f" Collected ${self.go_reward}."
        return {"reward": reward_contribution, "card_specific_desc": desc}

    def go_to_jail(self, player):
        """Moves player to Jail position, sets jail status, returns reward contribution + description."""
        self.positions[player] = self.jail_position
        self.in_jail[player] = True
        self.jail_counters[player] = 0
        reward_contribution = 0 # Or a small penalty like -50 if desired
        desc = "Moved to Jail."
        return {"reward": reward_contribution, "card_specific_desc": desc}

    def reset(self):
        self.positions = [0] * self.num_players
        self.money = [self.start_money] * self.num_players
        self.in_jail = [False] * self.num_players
        self.jail_counters = [0] * self.num_players
        # Initialize properties with owners set to None
        # During environment initialization
        self.properties = []
        for pos in range(40):  # Assuming a standard Monopoly board with 40 positions
            if pos in self.property_details:
                prop = self.property_details[pos].copy()
                prop["owner"] = None
                prop["houses"] = 0  # New: Initialize with 0 houses
            else:
                prop = {"price": 0, "rent": 0, "owner": None, "houses": 0}
            self.properties.append(prop)

        self.current_player = 0
        self.steps_taken = 0 # Renamed from 'steps' to avoid conflict
        self.done = False
        return self._get_obs() # Return initial observation

    # Renamed from _get_state to follow Gym convention
    def _get_obs(self):
        owners = [prop["owner"] if prop["owner"] is not None else -1 for prop in self.properties]
        obs = np.array(
            self.positions +
            self.money +
            [int(j) for j in self.in_jail] + # Convert bool to int
            owners +
            [self.current_player],
            dtype=np.int32
        )
        # Ensure obs fits within the defined observation space boundaries
        # This involves clamping money to avoid exceeding the high value
        obs[self.num_players : 2 * self.num_players] = np.clip(
            obs[self.num_players : 2 * self.num_players],
            self.observation_space.low[self.num_players],
            self.observation_space.high[self.num_players]
        )
        return obs

    def _player_has_properties(self, player_index):
        """Checks if the specified player owns any properties."""
        for prop in self.properties:
            if prop.get("owner") == player_index: # Use .get for safety, although owner should exist
                return True # Found at least one property
        return False # Looped through all properties, none owned by player

    def step(self, action): # Action is 0 (Pass) or 1 (Buy)
        if self.done:
            # If game already ended, return current state and 0 reward
            # Ensure a minimal log entry format if needed, but often just {} is fine.
            return self._get_obs(), 0, self.done, {"action_desc": "Game already ended."}

        p = self.current_player
        reward = 0 # Base reward for the step
        log_action_desc = ""
        money_before_turn = self.money[p]
        fee_paid = 0 # Track fees/rent paid this turn
        card_name_drawn = "" # Store card name if drawn
        card_spec_desc_drawn = "" # Store specific card description
        card_reward_contribution = 0 # Store reward specifically from card

        # --- Jail Logic ---
        if self.in_jail[p]:
            self.jail_counters[p] += 1
            dice1 = random.randint(1, 6)
            dice2 = random.randint(1, 6)
            rolled_doubles = (dice1 == dice2)
            turn_limit_reached = (self.jail_counters[p] >= self.jail_turns)

            if rolled_doubles:
                self.in_jail[p] = False
                self.jail_counters[p] = 0
                log_action_desc = f"Player {p} rolled doubles ({dice1}) to get out of jail. "
                # Player will now proceed to normal dice roll below
            elif turn_limit_reached:
                self.in_jail[p] = False
                self.jail_counters[p] = 0
                jail_fee = 50
                self.money[p] -= jail_fee
                fee_paid += jail_fee # Log the fee paid
                card_reward_contribution -= jail_fee # Apply penalty for paying
                log_action_desc = f"Player {p} paid ${jail_fee} to get out of jail (turn limit). "
                # Player will now proceed to normal dice roll below
            else: # Failed to roll doubles, turn ends here
                log_action_desc = f"Player {p} failed to roll doubles in jail (Turn {self.jail_counters[p]})."
                self._next_player()
                # Log turn spent in jail
                final_reward = reward + card_reward_contribution # Total reward for this step
                info = self._create_log_entry(
                    player=p, pos_before=self.positions[p], dice=0, # No move dice roll
                    pos_after=self.positions[p], money_before=money_before_turn,
                    money_after=self.money[p], reward=final_reward, fee_paid=fee_paid,
                    log_desc=log_action_desc, action_taken=action, # Log agent action even if unused
                    card_drawn=card_name_drawn, card_spec_desc=card_spec_desc_drawn,
                    landed_on=self.positions[p] # Didn't land anywhere new
                )
                # Need to increment step counter here for the skipped turn
                self.steps_taken += 1
                return self._get_obs(), final_reward, self.done, info # Return for the jail turn

        # --- Normal Turn: Dice Roll and Movement ---
        prev_position = self.positions[p]
        dice1 = random.randint(1, 6)
        dice2 = random.randint(1, 6)
        dice_total = dice1 + dice2

        # Calculate initial landing position
        landed_position_this_turn = (prev_position + dice_total) % self.board_size

        # Check for passing GO based on initial landing
        # Note: This check should happen BEFORE potential card move effects change position again
        passed_go = landed_position_this_turn < prev_position
        if passed_go:
            self.money[p] += self.go_reward
            reward += self.go_reward # Add GO reward to base step reward
            log_action_desc += f"Passed GO, collected ${self.go_reward}. "

        # Tentatively update position
        self.positions[p] = landed_position_this_turn
        pos = self.positions[p] # Current position for evaluation

        # --- Card Handling ---
        if pos in self.chance_positions:
            card = random.choice(self.chance_deck)
            card_name_drawn = card["name"]
            log_action_desc += f"Landed on Chance ({pos}), drew '{card_name_drawn}'. "
            card_effect_info = card["effect"](p) # Effect function modifies state
            card_reward_contribution += card_effect_info.get("reward", 0)
            card_spec_desc_drawn = card_effect_info.get("card_specific_desc", "")
            pos = self.positions[p] # IMPORTANT: Update pos in case card moved the player

        elif pos in self.chest_positions:
            card = random.choice(self.chest_deck)
            card_name_drawn = card["name"]
            log_action_desc += f"Landed on Community Chest ({pos}), drew '{card_name_drawn}'. "
            card_effect_info = card["effect"](p) # Effect function modifies state
            card_reward_contribution += card_effect_info.get("reward", 0)
            card_spec_desc_drawn = card_effect_info.get("card_specific_desc", "")
            pos = self.positions[p] # IMPORTANT: Update pos in case card moved the player

        # Append the specific card description to the main log description
        if card_spec_desc_drawn:
            log_action_desc += card_spec_desc_drawn + " "

        # --- Process Square Actions (based on final position 'pos' after potential card move) ---
        current_property = self.properties[pos]
        prop_price = current_property["price"]
        prop_rent = current_property["rent"]
        prop_owner = current_property["owner"]
        prop_houses = current_property.get("houses", 0)

        # 1. Go To Jail Square
        if pos == self.go_to_jail_position:
            # No double penalty if card already sent player here
            if not (card_name_drawn == "Go to Jail"):
                log_action_desc += f"Landed on Go To Jail ({pos}). Moved to Jail. "
                effect_info = self.go_to_jail(p) # Call effect to set state
                card_reward_contribution += effect_info.get("reward", 0) # Add potential penalty/reward
                # Note: go_to_jail already updates self.positions[p]
                pos = self.positions[p] # Ensure pos reflects Jail position (10)

        # 2. Fee Squares
        elif pos in self.fee_positions:
            fee = self.fee_positions[pos]
            self.money[p] -= fee
            fee_paid += fee
            card_reward_contribution -= fee # Apply fee penalty via card reward accumulator
            log_action_desc += f"Paid fee of ${fee} on square {pos} ({current_property['name']}). "

        # 3. Property Squares
        elif prop_price > 0:
            # a) Unowned
            if prop_owner is None:
                can_afford = self.money[p] >= prop_price
                if can_afford:
                    if action == 1:
                        self.money[p] -= prop_price
                        self.properties[pos]["owner"] = p
                        self.properties[pos]["houses"] = 0
                        fee_paid += prop_price
                        log_action_desc += f"Player {p} chose to BUY property {pos} ({current_property['name']}) for ${prop_price}. "
                    else:
                        log_action_desc += f"Player {p} chose NOT to buy property {pos} ({current_property['name']}) (${prop_price}). "
                else:
                     log_action_desc += f"Player {p} cannot afford property {pos} ({current_property['name']}) (${prop_price}). "
            # b) Owned by opponent
            elif prop_owner != p:
                num_houses = current_property["houses"]
                rent_due = prop_rent * (num_houses + 1) # Simplified rent
                payment = min(rent_due, self.money[p])
                self.money[p] -= payment
                self.money[prop_owner] += payment
                fee_paid += payment
                card_reward_contribution -= payment # Negative reward for paying rent
                log_action_desc += f"Paid ${payment} rent to Player {prop_owner} at property {pos} ({current_property['name']}) with {num_houses} houses. "
            # c) Owned by self
            else:
                 log_action_desc += f"Landed on own property {pos} ({current_property['name']}). "

        # 4. Other non-action squares (like Just Visiting, Free Parking)
        elif pos not in [0, self.jail_position, self.go_to_jail_position] and pos not in self.fee_positions and pos not in self.chance_positions and pos not in self.chest_positions:
             log_action_desc += f"Landed on non-action square {pos} ({current_property['name']}). "

        # --- Check for Bankruptcy (at the very end of money changes) ---
        if self.money[p] < 0:
            self.done = True
            card_reward_contribution -= 1000 # Bankruptcy penalty
            log_action_desc += f"Player {p} went bankrupt! "
            # Asset liquidation
            for i, prop in enumerate(self.properties):
                if prop["owner"] == p:
                    self.properties[i]["owner"] = None
                    self.properties[i]["houses"] = 0

               # --- Check for Need to Resolve Debt (AFTER all normal turn actions) ---
        if self.money[p] < 0 and not self.done:
            log_action_desc += f"Player {p} is bankrupt (${self.money[p]}). Attempting to sell assets. "
            bankruptcy_resolved = False

            # --- Phase 1: Sell Houses/Hotels ---
            # Create a list of properties owned by the player to iterate over
            owned_property_indices = [i for i, prop in enumerate(self.properties) if prop.get("owner") == p]

            # Sell houses evenly is complex, simplification: sell all houses everywhere first
            houses_sold_total_value = 0
            for i in owned_property_indices:
                prop = self.properties[i]
                house_cost = prop.get("house_cost", 0) # Get house cost, default 0 if not applicable (railroad/utility)
                if house_cost > 0 and prop["houses"] > 0:
                    num_houses_to_sell = prop["houses"]
                    sell_value_per_house = house_cost // 2 # Sell houses for half cost
                    money_from_houses = num_houses_to_sell * sell_value_per_house

                    self.money[p] += money_from_houses
                    prop["houses"] = 0 # Remove all houses/hotel
                    houses_sold_total_value += money_from_houses
                    log_action_desc += f"Sold {num_houses_to_sell} houses/hotel on {prop['name']} for ${money_from_houses}. "

                    # Check if solvent after selling houses on this property
                    if self.money[p] >= 0:
                        bankruptcy_resolved = True
                        log_action_desc += f"Player {p} is now solvent (${self.money[p]}) after selling houses. "
                        break # Stop selling houses

            # If still bankrupt after trying to sell all houses, proceed to sell properties
            if not bankruptcy_resolved and self.money[p] < 0:
                log_action_desc += "Still bankrupt after selling houses. Selling properties. "

                # --- Phase 2: Sell Properties (like mortgaging) ---
                # Simplification: Sell in the order they appear in the list for half price
                properties_sold_total_value = 0
                # Iterate over a copy of the indices, as we modify the underlying list properties
                indices_to_potentially_sell = list(owned_property_indices)

                for i in indices_to_potentially_sell:
                    # Re-check ownership in case something changed (unlikely here)
                    if self.properties[i].get("owner") == p:
                        prop = self.properties[i]
                        # Can only sell if it has no houses (should be true after Phase 1)
                        if prop["houses"] == 0:
                            sell_price = prop["price"] // 2 # Sell for half purchase price (like mortgage)
                            self.money[p] += sell_price
                            self.properties[i]["owner"] = None # Forfeit property to bank
                            properties_sold_total_value += sell_price
                            log_action_desc += f"Sold property {prop['name']} for ${sell_price}. "

                            # Check if solvent after selling this property
                            if self.money[p] >= 0:
                                bankruptcy_resolved = True
                                log_action_desc += f"Player {p} is now solvent (${self.money[p]}) after selling properties. "
                                break # Stop selling properties
                        else:
                            # Should not happen if Phase 1 worked correctly
                             log_action_desc += f"Skipped selling {prop['name']} because it still has houses (error?). "


            # --- Final Verdict ---
            if not bankruptcy_resolved and self.money[p] < 0:
                 # Still bankrupt after selling everything possible
                 self.done = True # Set game end flag
                 # card_reward_contribution -= 1000 # Apply bankruptcy penalty AFTER trying to resolve
                 log_action_desc += f"Player {p} could not raise enough funds. Final balance: ${self.money[p]}. Game Over! "
                 # Forfeit any remaining properties (shouldn't be any, but just in case)
                 for i, prop in enumerate(self.properties):
                      if prop["owner"] == p:
                          self.properties[i]["owner"] = None
                          self.properties[i]["houses"] = 0
            elif bankruptcy_resolved:
                 # Player managed to survive this time
                 log_action_desc += f"Player {p} survived bankruptcy. Current balance: ${self.money[p]}. "
                 # No game-ending penalty applied if they survive
            else:
                 # This case means money became >= 0 during the checks, but resolved flag wasn't set? Error.
                 log_action_desc += f"Bankruptcy resolution logic error. Final balance: ${self.money[p]}. "
                 if self.money[p] < 0: # Double check if truly bankrupt
                    self.done = True
                    log_action_desc += " Still bankrupt despite flag. Game Over! "


        # --- Finalize Step (Rest of the code remains the same) ---
        self.steps_taken += 1
        # Note: reward calculation happens BEFORE bankruptcy check,
        # but penalty is added via card_reward_contribution if game truly ends
        final_reward = reward + card_reward_contribution

        info = self._create_log_entry(
            player=p,
            pos_before=prev_position,
            dice=dice_total,
            pos_after=self.positions[p],
            money_before=money_before_turn,
            money_after=self.money[p], # Log final money after potential selling
            reward=final_reward,
            fee_paid=fee_paid,
            log_desc=log_action_desc.strip(),
            action_taken=action,
            card_drawn=card_name_drawn,
            card_spec_desc=card_spec_desc_drawn,
            landed_on=landed_position_this_turn
        )

        # Advance player ONLY if the game is not done
        if not self.done:
            self._next_player()

        return self._get_obs(), final_reward, self.done, info


    def _next_player(self):
        self.current_player = (self.current_player + 1) % self.num_players
        # Skip bankrupt players (basic implementation)
        # while self.money[self.current_player] < 0:
        #      self.current_player = (self.current_player + 1) % self.num_players

        # Add 'card_spec_desc' parameter with a default value
    def _create_log_entry(self, player, pos_before, dice, pos_after, money_before, money_after, reward, fee_paid, log_desc, action_taken=None, card_drawn="", card_spec_desc="", landed_on=-1): # <-- ADDED card_spec_desc="" HERE
        entry = {
            "episode_id": -1, # Will be overwritten by generate_episode
            "step": self.steps_taken,
            "player": player,
            "position_before": pos_before,
            "dice_roll": dice,
            "landed_on_position": landed_on, # Initial landing
            "position_after": pos_after,     # Final position
            "money_before": money_before,
            "money_after": money_after,
            "reward": reward,
            "done": self.done,
            "in_jail": self.in_jail[player],
            "fee_paid": fee_paid,
            "action_desc": log_desc.strip(),
            "agent_action": action_taken,
            "owned_properties": [
                {
                    "position": i,
                    "name": prop["name"],
                    "houses": prop["houses"]
                }
                for i, prop in enumerate(self.properties)
                if prop["owner"] == player
            ],
            "card": card_drawn,
            "card_specific_desc": card_spec_desc # Now uses the accepted parameter
        }
        return entry
    def render(self, mode='human'):
        # Simple text-based rendering
        print("-" * 20)
        print(f"Step: {self.steps_taken}, Current Player: {self.current_player}")
        for p in range(self.num_players):
            jail_status = "In Jail" if self.in_jail[p] else ""
            print(f"  Player {p}: Pos={self.positions[p]}, Money=${self.money[p]} {jail_status}")
        owners_str = [str(p['owner']) if p['owner'] is not None else '.' for p in self.properties]
        print(f"  Owners: [{' '.join(owners_str[:10])}]")
        print(f"          [{' '.join(owners_str[10:20])}]")
        print(f"          [{' '.join(owners_str[20:30])}]")
        print(f"          [{' '.join(owners_str[30:40])}]")


# --- Agent Class ---
class MonteCarloAgent:
    def __init__(self, action_space, num_players, epsilon=0.1): # Added num_players parameter
        self.epsilon = epsilon
        self.action_space = action_space
        self.num_players = num_players  # <--- MAKE SURE THIS LINE IS PRESENT
        self.returns = defaultdict(lambda: defaultdict(list))
        self.q_values = defaultdict(lambda: defaultdict(float))
        # Policy is implicitly epsilon-greedy based on Q-values

    def _get_state_tuple(self, obs):
        # Use the stored self.num_players
        num_players = self.num_players # <-- FIX: Use stored value

        # The rest of the method uses num_players correctly now
        positions = tuple(obs[0 : num_players])
        money = tuple(obs[num_players : 2 * num_players])
        money_bins = tuple(m // 100 for m in money)
        in_jail = tuple(obs[2 * num_players : 3 * num_players])
        owners_start_idx = 3 * num_players
        current_player_idx = obs[-1]
        current_pos = obs[current_player_idx]
        current_prop_owner = obs[owners_start_idx + current_pos]
        state_tuple = (
            positions[current_player_idx],
            money_bins[current_player_idx],
            current_prop_owner,
            in_jail[current_player_idx]
        )
        return state_tuple

    def select_action(self, state_tuple, current_obs, env):
        """Selects action (0 or 1) based on epsilon-greedy policy."""

        # --- Determine if a 'Buy' decision is even possible ---
        p = env.current_player # Get current player from env
        pos = env.positions[p]
        prop = env.properties[pos]
        is_buyable = prop["price"] > 0 and prop["owner"] is None and env.money[p] >= prop["price"]

        # If not on a buyable square, the only logical action is 0 (Pass/Continue)
        if not is_buyable:
            return 0

        # --- If buyable, use Epsilon-Greedy ---
        possible_actions = [0, 1] # 0: Don't Buy, 1: Buy

        if random.random() < self.epsilon:
            return random.choice(possible_actions)  # Explore
        else:
            # Exploit: Choose action with highest Q-value for this state
            q_vals = [self.q_values[state_tuple][a] for a in possible_actions]
            max_q = max(q_vals)

            # Handle cases where Q-values might be zero or equal
            if max_q == 0.0 and all(q == 0.0 for q in q_vals):
                 best_actions = possible_actions # If all Q=0, explore among options
            else:
                 best_actions = [a for a, q in zip(possible_actions, q_vals) if q == max_q]

            return random.choice(best_actions) # Break ties randomly

    def generate_episode(self, env, episode_id=None):
        """Generates one episode playing the game."""
        obs = env.reset()
        done = False
        episode_history = [] # Stores (state_tuple, action, reward) for MC update
        detailed_logs = []   # Stores the detailed log dict from env.step
        step_count = 0

        while not done:
            current_player = env.current_player # Who's turn is it?
            state_tuple = self._get_state_tuple(obs) # Get the simplified, hashable state for the agent

            # Agent selects action based on its policy and the *potential* decision
            action = self.select_action(state_tuple, obs, env)

            # Environment processes the turn based on dice rolls and the agent's action
            next_obs, reward, done, info = env.step(action)

            # Store data for MC update *using the state the decision was made in*
            episode_history.append((state_tuple, action, reward))

            # Store detailed log, adding episode_id
            info["episode_id"] = episode_id
            detailed_logs.append(info)

            obs = next_obs
            step_count += 1
            if step_count > 500: # Add a max step limit to prevent infinite loops
                # print(f"Episode {episode_id} reached step limit.")
                done = True # Force end episode

            # Optional: Render the game state periodically
            # if episode_id % 1000 == 0:
            #     env.render()


        return episode_history, detailed_logs

    def update(self, episode_history):
        """Updates Q-values using First-Visit Monte Carlo."""
        G = 0  # Cumulative reward (Return)
        visited_state_actions = set() # Keep track for first-visit MC

        # Iterate backwards through the episode
        for state_tuple, action, reward in reversed(episode_history):
            G += reward # Update return G

            state_action_pair = (state_tuple, action)

            # First-visit Monte Carlo check: only update the first time this (s,a) was visited
            if state_action_pair not in visited_state_actions:
                # Append return G to the list for this state-action pair
                self.returns[state_tuple][action].append(G)
                # Update Q-value as the average of observed returns
                self.q_values[state_tuple][action] = sum(self.returns[state_tuple][action]) / len(self.returns[state_tuple][action])

                visited_state_actions.add(state_action_pair)
                # Policy improvement is implicit via epsilon-greedy action selection in the next episode


# Simulation Parameters
num_episodes = 10000 # Number of episodes to run
epsilon_value = 0.2 # Exploration rate (start higher, maybe decay later)
log_filename = "monopoly_rl_log_properties_with_house.csv" # Name for the CSV log file

# Remove old log file if it exists
if os.path.exists(log_filename):
    try:
        os.remove(log_filename)
        print(f"Removed existing log file: {log_filename}")
    except OSError as e:
        print(f"Warning: Could not remove old log file '{log_filename}': {e}.")

# Initialize Environment and Agent
# Initialize Environment and Agent
env = MonopolyEnv(num_players=2)
# Pass num_players to the agent's constructor
agent = MonteCarloAgent(action_space=env.action_space, num_players=env.num_players, epsilon=epsilon_value) # Ensure num_players is passed here

# Define log headers
log_headers = [
    "episode_id", "step", "player", "position_before", "dice_roll", "landed_on_position","position_after",
    "money_before", "money_after", "reward", "done", "in_jail", "fee_paid",
    "agent_action", "action_desc", "owned_properties", "card","card_specific_desc"
]

#print(f"Starting Monte Carlo simulation for {num_episodes} episodes...")

# Run Simulation and Log Data
all_episode_logs = []
try:
    for episode_id in range(num_episodes):
        # Print progress periodically
        if (episode_id + 1) % 1000 == 0:
            print(f"Running episode {episode_id + 1}/{num_episodes}...")

        # Generate an episode using the agent's policy and get logs
        episode_history, detailed_logs = agent.generate_episode(env, episode_id=episode_id)

        # Update the agent's Q-values based on the episode
        agent.update(episode_history)

        # Accumulate logs
        all_episode_logs.extend(detailed_logs)

        # Optional: Decay epsilon over time
        # if epsilon_value > 0.05:
        #    epsilon_value *= 0.999 # Slow decay
        #    agent.epsilon = epsilon_value


    print("Simulation finished.")

    # Write logs to CSV after the simulation completes
    print(f"Writing log data to {log_filename}...")
    if all_episode_logs: # Check if there's anything to write
         with open(log_filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=log_headers)
            writer.writeheader()
            writer.writerows(all_episode_logs)
         print("Log data saved.")
    else:
         print("No log data generated.")



except Exception as e:
    print(f"\nAn error occurred during simulation or logging: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback

# --- Analysis and Plotting ---
print("Analyzing results...")

if os.path.exists(log_filename):
    try:
        df = pd.read_csv(log_filename)

        if df.empty:
             print("Log file is empty. No analysis performed.")
        else:
            # Calculate rolling window size
            # rolling_window = max(1, num_episodes // 50) # Use a smaller % for potentially shorter episodes

            # --- Calculate Winner/End Game Stats ---
            # Find the last step for each episode
            final_steps_df = df.loc[df.groupby('episode_id')['step'].idxmax()]

            # Determine winner (player with most money at the end, or the non-bankrupt one)
            def get_winner(row):
                # This assumes 2 players and bankruptcy is the primary end condition
                if row['player'] == 0 and row['done'] and row['money_after'] < 0: return 1 # Player 1 wins if Player 0 bankrupt
                if row['player'] == 1 and row['done'] and row['money_after'] < 0: return 0 # Player 0 wins if Player 1 bankrupt
                # If ended by step limit, player with more money wins
                # Need to get both players' final money - requires merging or smarter query
                # Simplified: assume the player in the last log row is indicative (might be wrong if other player went bankrupt earlier)
                if row['done']: return row['player'] # Placeholder - needs better winner logic if not ending by bankruptcy
                return -1 # Episode didn't finish properly?


            # Note: Accurate winner determination requires knowing BOTH players' final states.
            # This analysis section might need refinement based on how 'done' is triggered.
            # Let's focus on plots that don't rely heavily on accurate 'winner' status for now.


            # 1. Average Final Money per Player Over Time (Rolling Average)
            plt.figure(figsize=(12, 6))
            for p in range(env.num_players):
                 player_final_money = final_steps_df[final_steps_df['player'] == p]['money_after']
                 # Need to align indices if episodes end on different player turns
                 player_final_money = player_final_money.reindex(range(num_episodes)).ffill() # Fill gaps
                 if len(player_final_money) >= rolling_window:
                     rolling_avg = player_final_money.rolling(window=rolling_window).mean()
                     plt.plot(rolling_avg.index, rolling_avg, label=f'Player {p} Avg Final Money ({rolling_window} ep roll)', alpha=0.8)
                 else:
                     plt.plot(player_final_money.index, player_final_money, label=f'Player {p} Final Money', alpha=0.3)

            plt.title("Average Final Money per Player Over Time")
            plt.xlabel("Episode")
            plt.ylabel("Average Final Money ($)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # 3. Frequency of Landing on Each Board Position
            plt.figure(figsize=(15, 5))
            # Ensure plot uses integer bins covering the whole board
            bins = np.arange(env.board_size + 1) - 0.5
            plt.hist(df['landed_on_position'], bins=bins, edgecolor='black', density=True, alpha=0.7) # Use density=True for probability
            plt.title("Frequency Distribution of Landing on Board Positions")
            plt.xlabel("Board Position")
            plt.ylabel("Probability")
            plt.xticks(range(env.board_size))
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

            # 4. Agent Actions: Buy vs. Don't Buy Decisions Over Time
            # Filter logs for steps where a buy decision was possible and made
            buy_decision_df = df[df['agent_action'].notna() & (df['action_desc'].str.contains("chose to BUY") | df['action_desc'].str.contains("chose NOT to buy"))].copy()

            if not buy_decision_df.empty:
                buy_decision_df['episode_group'] = (buy_decision_df['episode_id'] // rolling_window) * rolling_window
                buy_rate_over_time = buy_decision_df.groupby('episode_group')['agent_action'].mean() # Avg action (1=Buy, 0=Pass) gives buy rate

                plt.figure(figsize=(12, 6))
                plt.plot(buy_rate_over_time.index, buy_rate_over_time, marker='o', linestyle='-', label=f'Buy Rate (Avg Action) per {rolling_window} Episodes')
                plt.title("Agent's Propensity to Buy Property Over Time (When Possible)")
                plt.xlabel(f"Episode Group (Start Episode of {rolling_window})")
                plt.ylabel("Average Action (1 = Buy, 0 = Don't Buy)")
                plt.ylim(-0.05, 1.05)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                print("No buy decisions were logged for action analysis.")

    except FileNotFoundError:
        print(f"Log file '{log_filename}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Log file '{log_filename}' is empty. No analysis performed.")
    except Exception as e:
        print(f"An error occurred during analysis or plotting: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"Log file '{log_filename}' not found.")
