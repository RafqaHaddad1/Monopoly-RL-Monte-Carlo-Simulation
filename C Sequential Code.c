%%writefile monopoly.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <errno.h>
#include <math.h>

// --- Constants ---
#define BOARD_SIZE 40
#define MAX_PLAYERS 8
#define MAX_PROPERTIES BOARD_SIZE
#define MAX_NAME_LEN 64
#define MAX_DESC_LEN 256
#define MAX_DECK_SIZE 16
#define Q_TABLE_INITIAL_SIZE 1024
#define VISITED_SET_INITIAL_SIZE 256
#define MAX_EPISODE_STEPS 500
#define LOG_BUFFER_SIZE 1000

// --- Structures ---

// Forward declaration
struct MonopolyEnv;

// Result structure for card effects
typedef struct {
    double reward;
    char card_specific_desc[MAX_DESC_LEN];
} CardEffectResult;

// Function pointer type for card effects
typedef CardEffectResult (*CardEffectFunc)(struct MonopolyEnv* env, int player_index);

// Card structure
typedef struct {
    char name[MAX_NAME_LEN];
    CardEffectFunc effect;
} Card;

// Property structure
typedef struct {
    int price;
    int rent;
    char name[MAX_NAME_LEN];
    int owner; // Player index, -1 for unowned/bank
    int houses;
    int house_cost; // Needed for bankruptcy selling logic
} Property;

// Log entry structure (mimics Python dictionary)
typedef struct {
    int step;
    int player;
    int position_before;
    int dice_roll;
    int landed_on_position;
    int position_after;
    int money_before;
    int money_after;
    double reward;
    bool done;
    bool in_jail;
    int fee_paid;
    char action_desc[MAX_DESC_LEN * 2]; // Allow longer descriptions
    int agent_action; // 0 or 1
    int num_owned_properties;
    char card_drawn[MAX_NAME_LEN];
    char card_specific_desc[MAX_DESC_LEN];
    int episode_id; // Added missing field
} LogEntry;


// Main environment structure
typedef struct MonopolyEnv {
    // Configuration
    int go_reward;
    int start_money;
    int board_size;
    int jail_position;
    int go_to_jail_position;
    int jail_turns;
    int num_players;

    // Game State
    int positions[MAX_PLAYERS];
    int money[MAX_PLAYERS];
    bool in_jail[MAX_PLAYERS];
    int jail_counters[MAX_PLAYERS];
    Property properties[MAX_PROPERTIES];
    int current_player;
    int steps_taken;
    bool done;

    // Decks
    Card chance_deck[MAX_DECK_SIZE];
    int chance_deck_size;
    Card chest_deck[MAX_DECK_SIZE];
    int chest_deck_size;

    // Observation space bounds
    int obs_money_high;

    // Log entry for the last step
    LogEntry last_log;

} MonopolyEnv;

// Result structure for the step function
typedef struct {
    double reward;
    bool done;
    LogEntry log;
    // Observation array is filled by the caller's provided pointer
} StepResult;


// --- Helper Functions ---

// Forward declarations for card effects
static CardEffectResult card_advance_to_go(MonopolyEnv* env, int player);
static CardEffectResult card_go_to_jail(MonopolyEnv* env, int player);
static CardEffectResult card_bank_dividend(MonopolyEnv* env, int player);
static CardEffectResult card_pay_poor_tax(MonopolyEnv* env, int player);
static CardEffectResult card_doctors_fee(MonopolyEnv* env, int player);
static CardEffectResult card_tax_refund(MonopolyEnv* env, int player);

// Helper to check if a position requires paying a fee
static int get_fee_for_position(int position) {
    switch (position) {
        case 4: return 200; // Income tax
        case 38: return 100; // Luxury tax
        default: return 0;
    }
}

// Helper to check if a position is Chance
static bool is_chance_position(int position) {
    return position == 7 || position == 22 || position == 36;
}

// Helper to check if a position is Community Chest
static bool is_chest_position(int position) {
    return position == 2 || position == 17 || position == 33;
}

// Helper function to adjust money and create part of the log description
static CardEffectResult adjust_money(MonopolyEnv* env, int player, int amount) {
    env->money[player] += amount;
    CardEffectResult result;
    result.reward = (double)amount; // Reward is the direct change
    snprintf(result.card_specific_desc, MAX_DESC_LEN, "Adjusted money by %d.", amount);
    return result;
}

// Helper: Advance player to GO
static CardEffectResult advance_to_go(MonopolyEnv* env, int player) {
    CardEffectResult result = {0.0, ""};
    bool passed_go = env->positions[player] != 0; // Collect if not already at GO
    env->positions[player] = 0;
    strncat(result.card_specific_desc, "Advanced to GO.", MAX_DESC_LEN - strlen(result.card_specific_desc) - 1);

    if (passed_go) {
        env->money[player] += env->go_reward;
        result.reward += env->go_reward;
        char go_desc[64];
        snprintf(go_desc, sizeof(go_desc), " Collected $%d.", env->go_reward);
        strncat(result.card_specific_desc, go_desc, MAX_DESC_LEN - strlen(result.card_specific_desc) - 1);
    }
    return result;
}

// Helper: Send player to Jail
static CardEffectResult go_to_jail(MonopolyEnv* env, int player) {
    CardEffectResult result = {0.0, ""}; // No immediate reward/penalty unless desired
    env->positions[player] = env->jail_position;
    env->in_jail[player] = true;
    env->jail_counters[player] = 0; // Reset jail turn counter
    snprintf(result.card_specific_desc, MAX_DESC_LEN, "Moved to Jail (Position %d).", env->jail_position);
    return result;
}


// --- Card Effect Implementations ---
static CardEffectResult card_advance_to_go(MonopolyEnv* env, int player) {
    return advance_to_go(env, player);
}

static CardEffectResult card_go_to_jail(MonopolyEnv* env, int player) {
    return go_to_jail(env, player);
}

static CardEffectResult card_bank_dividend(MonopolyEnv* env, int player) {
    return adjust_money(env, player, 50);
}

static CardEffectResult card_pay_poor_tax(MonopolyEnv* env, int player) {
    return adjust_money(env, player, -15);
}

static CardEffectResult card_doctors_fee(MonopolyEnv* env, int player) {
    return adjust_money(env, player, -50);
}

static CardEffectResult card_tax_refund(MonopolyEnv* env, int player) {
    return adjust_money(env, player, 20);
}


// --- Core Environment Functions ---

// Initialize Property Details (called by create_monopoly_env)
static void initialize_properties(Property properties[MAX_PROPERTIES]) {
    // Default all to non-properties first
    for (int i = 0; i < BOARD_SIZE; ++i) {
        properties[i] = (Property){.price = 0, .rent = 0, .name = "", .owner = -1, .houses = 0, .house_cost = 0};
        snprintf(properties[i].name, MAX_NAME_LEN, "Square %d", i);
    }

    // Overwrite with actual property data
    properties[1] = (Property){60, 2, "Mediterranean Avenue", -1, 0, 50};
    properties[3] = (Property){60, 4, "Baltic Avenue", -1, 0, 50};
    properties[5] = (Property){200, 25, "Reading Railroad", -1, 0, 100};
    properties[6] = (Property){100, 6, "Oriental Avenue", -1, 0, 50};
    properties[8] = (Property){100, 6, "Vermont Avenue", -1, 0, 50};
    properties[9] = (Property){120, 8, "Connecticut Avenue", -1, 0, 50};
    properties[11] = (Property){140, 10, "St. Charles Place", -1, 0, 100};
    properties[12] = (Property){150, 10, "Electric Company", -1, 0, 75}; // Utility
    properties[13] = (Property){140, 10, "States Avenue", -1, 0, 100};
    properties[14] = (Property){160, 12, "Virginia Avenue", -1, 0, 100};
    properties[15] = (Property){200, 25, "Pennsylvania Railroad", -1, 0, 100};
    properties[16] = (Property){180, 14, "St. James Place", -1, 0, 100};
    properties[18] = (Property){180, 14, "Tennessee Avenue", -1, 0, 100};
    properties[19] = (Property){200, 16, "New York Avenue", -1, 0, 100};
    properties[21] = (Property){220, 18, "Kentucky Avenue", -1, 0, 150};
    properties[23] = (Property){220, 18, "Indiana Avenue", -1, 0, 150};
    properties[24] = (Property){240, 20, "Illinois Avenue", -1, 0, 150};
    properties[25] = (Property){200, 25, "B. & O. Railroad", -1, 0, 100};
    properties[26] = (Property){260, 22, "Atlantic Avenue", -1, 0, 150};
    properties[27] = (Property){260, 22, "Ventnor Avenue", -1, 0, 150};
    properties[28] = (Property){150, 10, "Water Works", -1, 0, 75}; // Utility
    properties[29] = (Property){280, 24, "Marvin Gardens", -1, 0, 150};
    properties[31] = (Property){300, 26, "Pacific Avenue", -1, 0, 200};
    properties[32] = (Property){300, 26, "North Carolina Avenue", -1, 0, 200};
    properties[34] = (Property){320, 28, "Pennsylvania Avenue", -1, 0, 200};
    properties[35] = (Property){200, 25, "Short Line Railroad", -1, 0, 100};
    properties[37] = (Property){350, 35, "Park Place", -1, 0, 200};
    properties[39] = (Property){400, 50, "Boardwalk", -1, 0, 200};

    // Special square names
    strncpy(properties[0].name, "GO", MAX_NAME_LEN - 1);
    strncpy(properties[4].name, "Income Tax", MAX_NAME_LEN - 1);
    strncpy(properties[10].name, "Jail/Just Visiting", MAX_NAME_LEN - 1);
    strncpy(properties[20].name, "Free Parking", MAX_NAME_LEN - 1);
    strncpy(properties[30].name, "Go To Jail", MAX_NAME_LEN - 1);
    strncpy(properties[38].name, "Luxury Tax", MAX_NAME_LEN - 1);
    strncpy(properties[7].name, "Chance", MAX_NAME_LEN - 1);
    strncpy(properties[22].name, "Chance", MAX_NAME_LEN - 1);
    strncpy(properties[36].name, "Chance", MAX_NAME_LEN - 1);
    strncpy(properties[2].name, "Community Chest", MAX_NAME_LEN - 1);
    strncpy(properties[17].name, "Community Chest", MAX_NAME_LEN - 1);
    strncpy(properties[33].name, "Community Chest", MAX_NAME_LEN - 1);

    // Ensure null termination for all property names
    for (int i = 0; i < BOARD_SIZE; ++i) {
        properties[i].name[MAX_NAME_LEN - 1] = '\0';
    }
}

// Initialize Card Decks
static void initialize_decks(MonopolyEnv* env) {
    // Chance Deck
    env->chance_deck_size = 0;
    env->chance_deck[env->chance_deck_size++] = (Card){"Advance to Go", card_advance_to_go};
    env->chance_deck[env->chance_deck_size++] = (Card){"Go to Jail", card_go_to_jail};
    env->chance_deck[env->chance_deck_size++] = (Card){"Bank pays you dividend", card_bank_dividend};
    env->chance_deck[env->chance_deck_size++] = (Card){"Pay poor tax", card_pay_poor_tax};
    // Add more chance cards here if needed...

    // Community Chest Deck
    env->chest_deck_size = 0;
    env->chest_deck[env->chest_deck_size++] = (Card){"Doctor's fee", card_doctors_fee};
    env->chest_deck[env->chest_deck_size++] = (Card){"Income tax refund", card_tax_refund};
    env->chest_deck[env->chest_deck_size++] = (Card){"Go to Jail", card_go_to_jail};
    env->chest_deck[env->chest_deck_size++] = (Card){"Advance to Go", card_advance_to_go};
    // Add more chest cards here if needed...
}

// Get Observation (internal helper)
// Fills the provided obs array
static void get_observation(MonopolyEnv* env, int* obs) {
    int k = 0;
    // Positions
    for (int i = 0; i < env->num_players; ++i) obs[k++] = env->positions[i];
    // Money (clamped)
    for (int i = 0; i < env->num_players; ++i) {
        obs[k++] = (env->money[i] > env->obs_money_high) ? env->obs_money_high : env->money[i];
    }
    // In Jail flags
    for (int i = 0; i < env->num_players; ++i) obs[k++] = (int)env->in_jail[i];
    // Property Owners
    for (int i = 0; i < env->board_size; ++i) obs[k++] = env->properties[i].owner;
    // Current Player
    obs[k++] = env->current_player;
}

// Calculate observation size based on num_players
static int get_observation_size(int num_players) {
    return num_players + num_players + num_players + BOARD_SIZE + 1;
}


// Create and initialize the environment
MonopolyEnv* create_monopoly_env(int num_players, int start_money, int go_reward) {
    if (num_players <= 0 || num_players > MAX_PLAYERS) {
        fprintf(stderr, "Error: Invalid number of players (%d). Max is %d.\n", num_players, MAX_PLAYERS);
        return NULL;
    }

    MonopolyEnv* env = (MonopolyEnv*)malloc(sizeof(MonopolyEnv));
    if (!env) {
        perror("Failed to allocate memory for MonopolyEnv");
        return NULL;
    }

    // --- Configuration ---
    env->go_reward = go_reward;
    env->start_money = start_money;
    env->board_size = BOARD_SIZE; // Fixed size
    env->jail_position = 10;
    env->go_to_jail_position = 30;
    env->jail_turns = 3;
    env->num_players = num_players;
    env->obs_money_high = start_money * 10; // Arbitrary high limit for observation

    // --- Initialize State ---
    initialize_properties(env->properties);
    initialize_decks(env);
    env->current_player = 0;
    env->steps_taken = 0;
    env->done = false;

    // Reset player-specific state
    for (int i = 0; i < num_players; ++i) {
        env->positions[i] = 0;
        env->money[i] = start_money;
        env->in_jail[i] = false;
        env->jail_counters[i] = 0;
    }
    // Clear state for unused player slots
    for (int i = num_players; i < MAX_PLAYERS; ++i) {
        env->positions[i] = -1;
        env->money[i] = 0;
        env->in_jail[i] = false;
        env->jail_counters[i] = 0;
    }
    // Initialize log entry
    memset(&env->last_log, 0, sizeof(LogEntry));

    return env;
}

// Destroy the environment and free memory
void destroy_monopoly_env(MonopolyEnv* env) {
    if (env) {
        free(env);
    }
}

// Reset the environment state
// Fills the provided obs array with the initial observation
void reset_monopoly_env(MonopolyEnv* env, int* obs) {
    if (!env || !obs) return;

    // Reset properties
    for (int i = 0; i < env->board_size; ++i) {
        env->properties[i].owner = -1;
        env->properties[i].houses = 0;
        // Price, rent, name, house_cost remain as initialized
    }

    // Reset player state
    for (int i = 0; i < env->num_players; ++i) {
        env->positions[i] = 0;
        env->money[i] = env->start_money;
        env->in_jail[i] = false;
        env->jail_counters[i] = 0;
    }

    env->current_player = 0;
    env->steps_taken = 0;
    env->done = false;
    memset(&env->last_log, 0, sizeof(LogEntry));

    // Get initial observation
    get_observation(env, obs);
}


// Advance to the next player
static void next_player(MonopolyEnv* env) {
    env->current_player = (env->current_player + 1) % env->num_players;
}

// Create Log Entry (internal helper)
static LogEntry create_log_entry(MonopolyEnv* env, int player, int pos_before, int dice, int landed_on, int pos_after,
                                 int money_before, int money_after, double reward, int fee_paid,
                                 const char* log_desc, int action_taken, const char* card_drawn,
                                 const char* card_spec_desc)
{
    LogEntry entry;
    entry.step = env->steps_taken;
    entry.player = player;
    entry.position_before = pos_before;
    entry.dice_roll = dice;
    entry.landed_on_position = landed_on;
    entry.position_after = pos_after;
    entry.money_before = money_before;
    entry.money_after = money_after;
    entry.reward = reward;
    entry.done = env->done;
    entry.in_jail = env->in_jail[player];
    entry.fee_paid = fee_paid;
    entry.episode_id = 0; // Will be set by caller

    // Safe string copying
    strncpy(entry.action_desc, log_desc ? log_desc : "", sizeof(entry.action_desc) - 1);
    entry.action_desc[sizeof(entry.action_desc) - 1] = '\0'; // Ensure null termination

    entry.agent_action = action_taken;

    // Count owned properties for the log
    entry.num_owned_properties = 0;
    for (int i = 0; i < env->board_size; ++i) {
        if (env->properties[i].owner == player) {
            entry.num_owned_properties++;
        }
    }

    // Safe string copying for card info
    strncpy(entry.card_drawn, card_drawn ? card_drawn : "", sizeof(entry.card_drawn) - 1);
    entry.card_drawn[sizeof(entry.card_drawn) - 1] = '\0';

    strncpy(entry.card_specific_desc, card_spec_desc ? card_spec_desc : "", sizeof(entry.card_specific_desc) - 1);
    entry.card_specific_desc[sizeof(entry.card_specific_desc) - 1] = '\0';

    return entry;
}


// Perform one step in the environment
// Action: 0=Pass/Don't Buy, 1=Buy
// Fills the provided obs array with the next observation
StepResult step_monopoly_env(MonopolyEnv* env, int action, int* obs) {
    StepResult result = {0.0, env->done, {0}}; // Initialize result

    if (env->done) {
        // Game already ended, return current state and 0 reward
        get_observation(env, obs);
        strncpy(result.log.action_desc, "Game already ended.", sizeof(result.log.action_desc) - 1);
        result.log = env->last_log; // Return last log entry
        result.log.step = env->steps_taken; // Update step count if needed
        result.log.action_desc[0] = '\0'; // Clear desc
        strncat(result.log.action_desc, "Game already ended.", sizeof(result.log.action_desc) - 1);
        return result;
    }

    int p = env->current_player;
    double current_step_reward = 0.0;
    char log_buffer[MAX_DESC_LEN * 2] = ""; // Buffer for building the log description
    int money_before_turn = env->money[p];
    int fee_paid_this_turn = 0;
    char card_name_drawn[MAX_NAME_LEN] = "";
    char card_spec_desc_drawn[MAX_DESC_LEN] = "";
    double card_reward_contribution = 0.0;
    int dice_total = 0;
    int prev_position = env->positions[p];
    int landed_position_this_turn = prev_position; // Initial value

    // --- Jail Logic ---
    if (env->in_jail[p]) {
        env->jail_counters[p]++;
        int dice1 = (rand() % 6) + 1;
        int dice2 = (rand() % 6) + 1;
        bool rolled_doubles = (dice1 == dice2);
        bool turn_limit_reached = (env->jail_counters[p] >= env->jail_turns);

        if (rolled_doubles) {
            env->in_jail[p] = false;
            env->jail_counters[p] = 0;
            snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                     "Player %d rolled doubles (%d) to get out of jail. ", p, dice1);
            // Player proceeds to normal dice roll below
        } else if (turn_limit_reached) {
            env->in_jail[p] = false;
            env->jail_counters[p] = 0;
            int jail_fee = 50;
            env->money[p] -= jail_fee;
            fee_paid_this_turn += jail_fee;
            card_reward_contribution -= jail_fee; // Penalty for paying
            snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                     "Player %d paid $%d to get out of jail (turn limit). ", p, jail_fee);
            // Player proceeds to normal dice roll below
        } else {
            // Failed to roll doubles, turn ends here
            snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                     "Player %d failed to roll doubles in jail (Turn %d).", p, env->jail_counters[p]);

            env->steps_taken++; // Increment step counter for the turn spent in jail
            result.reward = card_reward_contribution; // Only reward/penalty from jail fee attempt
            result.done = env->done; // Done status might change if fee caused bankruptcy (checked later)

            // Check immediate bankruptcy from paying fee if turn limit reached
             if (env->money[p] < 0 && !env->done) {
                 // Attempt to resolve bankruptcy (Simplified check here, full check later)
                  if (env->money[p] < 0) { // Still bankrupt after trying to pay
                        env->done = true; // Game over
                        card_reward_contribution -= 1000; // Bankruptcy penalty
                        snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                                 "Player %d went bankrupt paying jail fee! ", p);
                        // Forfeit properties
                        for(int i=0; i<env->board_size; ++i) {
                            if(env->properties[i].owner == p) {
                                env->properties[i].owner = -1;
                                env->properties[i].houses = 0;
                            }
                        }
                 }
                 result.done = env->done; // Update done status
                 result.reward += card_reward_contribution; // Include bankruptcy penalty
            }

            // Log and return for the jail turn
            env->last_log = create_log_entry(env, p, prev_position, 0, prev_position, env->positions[p],
                                             money_before_turn, env->money[p], result.reward, fee_paid_this_turn,
                                             log_buffer, action, "", ""); // No move/card
            result.log = env->last_log;
            get_observation(env, obs);
            next_player(env); // Advance player
            return result;
        }
        // If got out of jail, continue to normal turn roll
    }

    // --- Normal Turn: Dice Roll and Movement ---
    int dice1 = (rand() % 6) + 1;
    int dice2 = (rand() % 6) + 1;
    dice_total = dice1 + dice2;

    landed_position_this_turn = (prev_position + dice_total) % env->board_size;

    // Check for passing GO
    bool passed_go = landed_position_this_turn < prev_position && !(env->in_jail[p]); // Don't collect if just got out of jail and landed before GO
     if (passed_go && prev_position != env->jail_position) { // Ensure not passing GO due to leaving jail on pos 10
        env->money[p] += env->go_reward;
        current_step_reward += env->go_reward;
        snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                 "Passed GO, collected $%d. ", env->go_reward);
    }

    // Tentatively update position
    env->positions[p] = landed_position_this_turn;
    int pos = env->positions[p]; // Current position for evaluation

    // --- Card Handling ---
    CardEffectResult card_result = {0.0, ""};
    bool card_drawn = false;

    if (is_chance_position(pos)) {
        int card_index = rand() % env->chance_deck_size;
        Card drawn_card = env->chance_deck[card_index];
        strncpy(card_name_drawn, drawn_card.name, MAX_NAME_LEN - 1);
        card_name_drawn[MAX_NAME_LEN - 1] = '\0';
        snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                 "Landed on Chance (%d), drew '%s'. ", pos, card_name_drawn);
        card_result = drawn_card.effect(env, p); // Effect function modifies state
        card_reward_contribution += card_result.reward;
        strncpy(card_spec_desc_drawn, card_result.card_specific_desc, MAX_DESC_LEN - 1);
        card_spec_desc_drawn[MAX_DESC_LEN - 1] = '\0';
        pos = env->positions[p]; // IMPORTANT: Update pos in case card moved the player
        card_drawn = true;
    } else if (is_chest_position(pos)) {
        int card_index = rand() % env->chest_deck_size;
        Card drawn_card = env->chest_deck[card_index];
        strncpy(card_name_drawn, drawn_card.name, MAX_NAME_LEN - 1);
        card_name_drawn[MAX_NAME_LEN - 1] = '\0';
        snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                 "Landed on Community Chest (%d), drew '%s'. ", pos, card_name_drawn);
        card_result = drawn_card.effect(env, p); // Effect function modifies state
        card_reward_contribution += card_result.reward;
        strncpy(card_spec_desc_drawn, card_result.card_specific_desc, MAX_DESC_LEN - 1);
        card_spec_desc_drawn[MAX_DESC_LEN - 1] = '\0';
        pos = env->positions[p]; // IMPORTANT: Update pos in case card moved the player
        card_drawn = true;
    }

    // Append the specific card description to the main log description
    if (strlen(card_spec_desc_drawn) > 0) {
        strncat(log_buffer, card_spec_desc_drawn, sizeof(log_buffer) - strlen(log_buffer) - 1);
        strncat(log_buffer, " ", sizeof(log_buffer) - strlen(log_buffer) - 1);
    }


    // --- Process Square Actions (based on final position 'pos' after potential card move) ---
    Property* current_property = &env->properties[pos];
    int prop_price = current_property->price;
    int prop_rent = current_property->rent;
    int prop_owner = current_property->owner;
    int prop_houses = current_property->houses;


    // 1. Go To Jail Square
    if (pos == env->go_to_jail_position) {
        // Avoid double penalty if card already sent player here
        if (!card_drawn || strcmp(card_name_drawn, "Go to Jail") != 0) {
             snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                     "Landed on Go To Jail (%d). ", pos);
             CardEffectResult jail_effect = go_to_jail(env, p); // Call effect to set state
             card_reward_contribution += jail_effect.reward; // Add potential penalty/reward
             strncat(log_buffer, jail_effect.card_specific_desc, sizeof(log_buffer) - strlen(log_buffer) - 1);
             pos = env->positions[p]; // Ensure pos reflects Jail position (10)
        }
    }
    // 2. Fee Squares
    else if (get_fee_for_position(pos) > 0) {
        int fee = get_fee_for_position(pos);
        env->money[p] -= fee;
        fee_paid_this_turn += fee;
        card_reward_contribution -= fee; // Apply fee penalty via card reward accumulator
        snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                 "Paid fee of $%d on square %d (%s). ", fee, pos, current_property->name);
    }
    // 3. Property Squares
    else if (prop_price > 0) {
        // a) Unowned
        if (prop_owner == -1) {
            bool can_afford = (env->money[p] >= prop_price);
            if (can_afford) {
                if (action == 1) { // Agent chose to buy
                    current_step_reward += 100.0; // bonus reward for buying a property
                    env->money[p] -= prop_price;
                    env->properties[pos].owner = p;
                    env->properties[pos].houses = 0; // Ensure houses reset on purchase
                    snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                             "Player %d chose to BUY property %d (%s) for $%d. ", p, pos, current_property->name, prop_price);
                } else { // Agent chose not to buy
                    snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                             "Player %d chose NOT to buy property %d (%s) for $%d. ", p, pos, current_property->name, prop_price);
                }
            } else { // Cannot afford
                 snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                         "Player %d cannot afford property %d (%s) ($%d). ", p, pos, current_property->name, prop_price);
            }
        }
       // b) Owned by opponent
      else if (prop_owner != p) {
          int rent_due = prop_rent; // Base rent

          // More realistic rent multipliers based on house count
          if (prop_houses > 0) {
              switch (prop_houses) {
                  case 1:
                      rent_due = prop_rent * 5;    // Typically 5x base rent
                      break;
                  case 2:
                      rent_due = prop_rent * 15;   // Typically 15x base rent
                      break;
                  case 3:
                      rent_due = prop_rent * 45;   // Typically 45x base rent
                      break;
                  case 4:
                      rent_due = prop_rent * 80;   // Typically 80x base rent
                      break;
                  case 5: // Hotel
                      rent_due = prop_rent * 125;  // Typically 125x base rent
                      break;
                  default:
                      rent_due *= (prop_houses + 1); // Fallback
              }
          }

          int payment = (env->money[p] < rent_due) ? env->money[p] : rent_due; // Pay what you can

          env->money[p] -= payment;
          if (prop_owner >= 0 && prop_owner < env->num_players) { // Ensure owner is valid
              env->money[prop_owner] += payment;
          }
          fee_paid_this_turn += payment; // Rent counts as a fee paid
          card_reward_contribution -= payment; // Negative reward for paying rent

          // Update log message
          const char* property_state = (prop_houses == 5) ? "hotel" :
                                    (prop_houses > 0) ? "houses" : "no houses";

          snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                  "Paid $%d rent to Player %d at property %d (%s) with %d %s. ",
                  payment, prop_owner, pos, current_property->name,
                  (prop_houses == 5) ? 1 : prop_houses, property_state);
      }
        // c) Owned by self
        else {
            // Check if player can buy houses on this property
            bool can_buy_houses = (env->properties[pos].house_cost > 0 && env->money[p] >= env->properties[pos].house_cost);
            int current_houses = env->properties[pos].houses;
            int max_houses = 5; // 4 houses + 1 hotel

            // Only allow buying houses if we have less than the maximum
            if (can_buy_houses && current_houses < max_houses) {
                // Determine how many houses the player can afford
                int affordable_houses = env->money[p] / env->properties[pos].house_cost;
                // Limit to how many more houses can be added
                int max_new_houses = max_houses - current_houses;
                int houses_can_buy = (affordable_houses < max_new_houses) ? affordable_houses : max_new_houses;

                // Agent decides whether to buy houses and how many (using action)
                // For simplicity, if action is 1 (buy), buy as many as possible up to 1
                int houses_to_buy = 0;
                if (action == 1 && houses_can_buy > 0) {
                    houses_to_buy = 1; // Buy one house at a time

                    // Update property and player money
                    int house_cost = env->properties[pos].house_cost;
                    env->properties[pos].houses += houses_to_buy;
                    env->money[p] -= house_cost * houses_to_buy;

                    // Log the purchase
                    snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                            "Landed on own property %d (%s). Bought %d house(s) for $%d. Now has %d houses. ",
                            pos, current_property->name, houses_to_buy, house_cost * houses_to_buy,
                            env->properties[pos].houses);
                } else {
                    // Chose not to buy houses
                    snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                            "Landed on own property %d (%s). Chose not to buy houses (current: %d). ",
                            pos, current_property->name, current_houses);
                }
            } else if (current_houses >= max_houses) {
                // Already has maximum houses
                snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                        "Landed on own property %d (%s). Already has maximum houses/hotel (%d). ",
                        pos, current_property->name, current_houses);
            } else if (env->properties[pos].house_cost <= 0) {
                // Property doesn't support houses (like railroads or utilities)
                snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                        "Landed on own property %d (%s). This property type doesn't support houses. ",
                        pos, current_property->name);
            } else {
                // Can't afford houses
                snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                        "Landed on own property %d (%s). Cannot afford houses (cost: $%d). ",
                        pos, current_property->name, env->properties[pos].house_cost);
            }
        }
    }
     // 4. Other non-action squares
    else if (pos != 0 && pos != env->jail_position && !is_chance_position(pos) && !is_chest_position(pos)) {
         snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                 "Landed on non-action square %d (%s). ", pos, current_property->name);
    }


     // --- Check for Bankruptcy (AFTER all turn actions and money changes) ---
    if (env->money[p] < 0 && !env->done) {
        snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                 "Player %d is bankrupt ($%d). Attempting to sell assets. ", p, env->money[p]);
        bool bankruptcy_resolved = false;

        // --- Phase 1: Sell Houses ---
        int houses_sold_total_value = 0;
        for (int i = 0; i < env->board_size; ++i) {
            if (env->properties[i].owner == p && env->properties[i].houses > 0) {
                int house_cost = env->properties[i].house_cost; // Get house cost
                if (house_cost > 0) { // Can only sell houses if they have a cost basis
                    int num_houses_to_sell = env->properties[i].houses;
                    int sell_value_per_house = house_cost / 2; // Sell for half cost
                    int money_from_houses = num_houses_to_sell * sell_value_per_house;

                    env->money[p] += money_from_houses;
                    env->properties[i].houses = 0; // Remove all houses/hotel
                    houses_sold_total_value += money_from_houses;

                    snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                             "Sold %d houses/hotel on %s (%d) for $%d. ",
                             num_houses_to_sell, env->properties[i].name, i, money_from_houses);

                    // Check if solvent after selling houses on this property
                    if (env->money[p] >= 0) {
                        bankruptcy_resolved = true;
                        snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                                 "Player %d is now solvent ($%d) after selling houses. ", p, env->money[p]);
                        break; // Stop selling houses
                    }
                }
            }
        }

        // --- Phase 2: Sell Properties (like mortgaging, sell for half price) ---
        if (!bankruptcy_resolved && env->money[p] < 0) {
            snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                     "Still bankrupt after selling houses. Selling properties. ");

            int properties_sold_total_value = 0;
            // Sell in board order
            for (int i = 0; i < env->board_size; ++i) {
                 if (env->properties[i].owner == p) {
                     // Can only sell if it has NO houses (should be true after Phase 1)
                     if (env->properties[i].houses == 0 && env->properties[i].price > 0) {
                        int sell_price = env->properties[i].price / 2; // Sell for half purchase price
                        env->money[p] += sell_price;
                        env->properties[i].owner = -1; // Forfeit property to bank
                        properties_sold_total_value += sell_price;
                        snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                                 "Sold property %s (%d) for $%d. ", env->properties[i].name, i, sell_price);

                        // Check if solvent after selling this property
                        if (env->money[p] >= 0) {
                            bankruptcy_resolved = true;
                             snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                                     "Player %d is now solvent ($%d) after selling properties. ", p, env->money[p]);
                            break; // Stop selling properties
                        }
                    } else if (env->properties[i].houses > 0) {
                         // Should not happen if Phase 1 worked correctly
                         snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                                  "Skipped selling %s (%d) because it still has houses (error?). ", env->properties[i].name, i);
                    }
                 }
                 // If still bankrupt after checking all properties, loop finishes
                 if (bankruptcy_resolved) break;
            }
        }

        // --- Final Verdict ---
        if (!bankruptcy_resolved && env->money[p] < 0) {
             // Still bankrupt after selling everything possible
             env->done = true; // Set game end flag
             card_reward_contribution -= 1000; // Apply bankruptcy penalty AFTER trying to resolve
             snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                      "Player %d could not raise enough funds. Final balance: $%d. Game Over! ", p, env->money[p]);
             // Ensure all properties are forfeited
             for (int i = 0; i < env->board_size; ++i) {
                  if (env->properties[i].owner == p) {
                      env->properties[i].owner = -1;
                      env->properties[i].houses = 0;
                  }
             }
        } else if (bankruptcy_resolved) {
             // Player managed to survive this time
             snprintf(log_buffer + strlen(log_buffer), sizeof(log_buffer) - strlen(log_buffer),
                      "Player %d survived bankruptcy. Current balance: $%d. ", p, env->money[p]);
        }
    }


    // --- Finalize Step ---
    env->steps_taken++;
    result.reward = current_step_reward + card_reward_contribution; // Combine base reward and card/fee effects
    result.done = env->done;

    // Create the log entry for this step
    env->last_log = create_log_entry(env, p, prev_position, dice_total, landed_position_this_turn, env->positions[p],
                                     money_before_turn, env->money[p], result.reward, fee_paid_this_turn,
                                     log_buffer, action, card_name_drawn, card_spec_desc_drawn);
    result.log = env->last_log;


    // Get the observation for the *next* state
    get_observation(env, obs);

    // Advance player ONLY if the game is not done
    if (!env->done) {
        next_player(env);
    }

    return result;
}


// Render the current state to the console
void render_monopoly_env(MonopolyEnv* env) {
    if (!env) return;

    printf("----------------------------------------\n");
    printf("Step: %d, Current Player: %d%s\n", env->steps_taken, env->current_player, env->done ? " (Game Over)" : "");
    for (int p = 0; p < env->num_players; ++p) {
        const char* jail_status = env->in_jail[p] ? " (In Jail)" : "";
        printf("  Player %d: Pos=%2d, Money=$%5d %s\n", p, env->positions[p], env->money[p], jail_status);
    }

    printf("  Board Owners (-1 = Bank/None):\n");
    printf("  [ ");
    for (int i = 0; i < 10; ++i) printf("%2d ", env->properties[i].owner);
    printf("]\n");
    printf("  [ ");
    for (int i = 10; i < 20; ++i) printf("%2d ", env->properties[i].owner);
    printf("]\n");
    printf("  [ ");
    for (int i = 20; i < 30; ++i) printf("%2d ", env->properties[i].owner);
    printf("]\n");
    printf("  [ ");
    for (int i = 30; i < 40; ++i) printf("%2d ", env->properties[i].owner);
    printf("]\n");
    printf("----------------------------------------\n");

    // Optionally print last log message
    if (strlen(env->last_log.action_desc) > 0) {
       printf("Last Action Log: %s\n", env->last_log.action_desc);
       printf("  Reward: %.2f, Fee Paid: %d, Card: '%s'\n", env->last_log.reward, env->last_log.fee_paid, env->last_log.card_drawn);
       printf("----------------------------------------\n");
    }
}


// --- Agent Data Structures ---

// Represents the simplified state used as a key in the Q-table
typedef struct {
    int position;
    int money_bin; // Discretized money
    int current_prop_owner; // Owner of the property player is on (-1, 0, 1, ...)
    int in_jail; // 0 or 1
} StateTuple;

// Data stored for each ACTION within a state entry in the Q-table
typedef struct {
    double sum_returns;
    int count;
    double q_value; // q_value = sum_returns / count
} QValueData;

// An entry in the Q-value hash table (using chaining for collisions)
typedef struct QTableEntry {
    StateTuple key;
    QValueData values[2]; // Index 0 for action 0 (Pass), 1 for action 1 (Buy)
    struct QTableEntry* next; // Pointer for chaining
} QTableEntry;

// The Q-value hash table structure
typedef struct {
    QTableEntry** table;
    int size;
    int count; // Number of entries
} QHashTable;

// Represents a (StateTuple, action) pair for the visited set during update
typedef struct {
    StateTuple state;
    int action;
} VisitedKey;

// An entry in the visited set hash table (using chaining)
typedef struct VisitedSetEntry {
    VisitedKey key;
    struct VisitedSetEntry* next;
} VisitedSetEntry;

// The visited set hash table structure
typedef struct {
    VisitedSetEntry** table;
    int size;
    int count;
} VisitedSet;


// Structure to hold one step of an episode's history
typedef struct {
    StateTuple state; // The state tuple *before* taking the action
    int action;
    double reward;
} EpisodeStep;

// Structure to hold the history of an entire episode
typedef struct {
    EpisodeStep* steps;
    int count;
    int capacity;
} EpisodeHistory;


// The Monte Carlo Agent structure
typedef struct {
    double epsilon;
    int num_players;
    QHashTable* q_table; // Pointer to the Q-value hash table
} MonteCarloAgent;


// --- Hash Table & State Tuple Helper Functions ---

// Hash function for StateTuple
static unsigned int hash_state_tuple(StateTuple s, int table_size) {
    // Simple combination hash - adjust multipliers for better distribution if needed
    unsigned int hash = 17;
    hash = (hash * 31 + s.position) % table_size;
    hash = (hash * 31 + s.money_bin) % table_size;
    // Add 10 to owner to handle -1 gracefully in hashing (avoids negative intermediate)
    hash = (hash * 31 + (s.current_prop_owner + 10)) % table_size;
    hash = (hash * 31 + s.in_jail) % table_size;
    return hash;
}

// Comparison function for StateTuple
static bool compare_state_tuples(StateTuple s1, StateTuple s2) {
    return s1.position == s2.position &&
           s1.money_bin == s2.money_bin &&
           s1.current_prop_owner == s2.current_prop_owner &&
           s1.in_jail == s2.in_jail;
}

// Create a new Q-value hash table
static QHashTable* create_q_hash_table(int size) {
    QHashTable* ht = (QHashTable*)malloc(sizeof(QHashTable));
    if (!ht) return NULL;
    ht->size = size;
    ht->count = 0;
    ht->table = (QTableEntry**)calloc(size, sizeof(QTableEntry*)); // Initialize all to NULL
    if (!ht->table) {
        free(ht);
        return NULL;
    }
    return ht;
}

// Find or create an entry in the Q-value hash table
static QTableEntry* find_or_create_q_entry(QHashTable* ht, StateTuple key) {
    unsigned int index = hash_state_tuple(key, ht->size);
    QTableEntry* entry = ht->table[index];

    // Search existing chain
    while (entry != NULL) {
        if (compare_state_tuples(entry->key, key)) {
            return entry; // Found it
        }
        entry = entry->next;
    }

    // Not found, create a new entry
    QTableEntry* new_entry = (QTableEntry*)malloc(sizeof(QTableEntry));
    if (!new_entry) return NULL; // Allocation failed

    new_entry->key = key;
    // Initialize Q-value data (sum=0, count=0, q=0)
    memset(new_entry->values, 0, sizeof(new_entry->values));
    new_entry->next = ht->table[index]; // Link into chain (at the front)
    ht->table[index] = new_entry;
    ht->count++;

    return new_entry;
}


// Destroy the Q-value hash table
static void destroy_q_hash_table(QHashTable* ht) {
    if (!ht) return;
    for (int i = 0; i < ht->size; ++i) {
        QTableEntry* entry = ht->table[i];
        while (entry != NULL) {
            QTableEntry* temp = entry;
            entry = entry->next;
            free(temp);
        }
    }
    free(ht->table);
    free(ht);
}


// --- Visited Set Hash Table Functions ---

// Hash function for VisitedKey
static unsigned int hash_visited_key(VisitedKey k, int table_size) {
     unsigned int state_hash = hash_state_tuple(k.state, table_size);
     // Combine state hash with action
     unsigned int hash = (state_hash * 31 + k.action) % table_size;
     return hash;
}

// Comparison function for VisitedKey
static bool compare_visited_keys(VisitedKey k1, VisitedKey k2) {
    return k1.action == k2.action && compare_state_tuples(k1.state, k2.state);
}

// Create a new visited set hash table
static VisitedSet* create_visited_set(int size) {
    VisitedSet* vs = (VisitedSet*)malloc(sizeof(VisitedSet));
    if (!vs) return NULL;
    vs->size = size;
    vs->count = 0;
    vs->table = (VisitedSetEntry**)calloc(size, sizeof(VisitedSetEntry*));
    if (!vs->table) {
        free(vs);
        return NULL;
    }
    return vs;
}

// Check if a key exists and add it if it doesn't (returns true if it was already present)
static bool check_and_add_visited(VisitedSet* vs, VisitedKey key) {
    unsigned int index = hash_visited_key(key, vs->size);
    VisitedSetEntry* entry = vs->table[index];

    // Search chain
    while (entry != NULL) {
        if (compare_visited_keys(entry->key, key)) {
            return true; // Already visited
        }
        entry = entry->next;
    }

    // Not visited, add it
    VisitedSetEntry* new_entry = (VisitedSetEntry*)malloc(sizeof(VisitedSetEntry));
    if (!new_entry) return false; // Allocation error, treat as not visited

    new_entry->key = key;
    new_entry->next = vs->table[index];
    vs->table[index] = new_entry;
    vs->count++;

    return false; // Was not previously visited
}

// Destroy the visited set hash table (call this after each episode update)
static void destroy_visited_set(VisitedSet* vs) {
     if (!vs) return;
    for (int i = 0; i < vs->size; ++i) {
        VisitedSetEntry* entry = vs->table[i];
        while (entry != NULL) {
            VisitedSetEntry* temp = entry;
            entry = entry->next;
            free(temp);
        }
    }
    free(vs->table);
    free(vs);
}


// --- Episode History Functions ---

// Initialize episode history
static void init_episode_history(EpisodeHistory* history, int initial_capacity) {
    history->steps = (EpisodeStep*)malloc(initial_capacity * sizeof(EpisodeStep));
    history->count = 0;
    history->capacity = history->steps ? initial_capacity : 0;
}

// Add a step to the history, resizing if necessary
static void add_episode_step(EpisodeHistory* history, StateTuple state, int action, double reward) {
    if (history->count >= history->capacity) {
        int new_capacity = history->capacity > 0 ? history->capacity * 2 : 10; // Double capacity
        EpisodeStep* new_steps = (EpisodeStep*)realloc(history->steps, new_capacity * sizeof(EpisodeStep));
        if (!new_steps) {
            fprintf(stderr, "Error: Failed to reallocate episode history\n");
            // Keep old data, but can't add more
            return;
        }
        history->steps = new_steps;
        history->capacity = new_capacity;
    }
    history->steps[history->count].state = state;
    history->steps[history->count].action = action;
    history->steps[history->count].reward = reward;
    history->count++;
}

// Free memory used by episode history
static void free_episode_history(EpisodeHistory* history) {
    if (history && history->steps) {
        free(history->steps);
        history->steps = NULL;
        history->count = 0;
        history->capacity = 0;
    }
}

// --- Agent Implementation ---

// Create the Monte Carlo agent
MonteCarloAgent* create_monte_carlo_agent(int num_players, double epsilon) {
    MonteCarloAgent* agent = (MonteCarloAgent*)malloc(sizeof(MonteCarloAgent));
    if (!agent) return NULL;

    agent->epsilon = epsilon;
    agent->num_players = num_players;
    agent->q_table = create_q_hash_table(Q_TABLE_INITIAL_SIZE);
    if (!agent->q_table) {
        free(agent);
        return NULL;
    }
    return agent;
}

// Destroy the Monte Carlo agent
void destroy_monte_carlo_agent(MonteCarloAgent* agent) {
    if (agent) {
        destroy_q_hash_table(agent->q_table);
        free(agent);
    }
}

// Helper function to extract the simplified state tuple from the full observation
static StateTuple _get_state_tuple_c(const int* obs, int num_players, int board_size) {
    StateTuple current_state_tuple = {0};
    int current_player_idx = obs[num_players * 3 + board_size]; // Last element is current player index

    // Extract relevant parts for the *current* player
    int player_pos = obs[current_player_idx];
    int player_money = obs[num_players + current_player_idx];
    int player_in_jail = obs[2 * num_players + current_player_idx];

    // Find owner of the property the current player landed on
    int owners_start_idx = 3 * num_players;
    int current_prop_owner = -1; // Default if not on a property square or index out of bounds
    if (player_pos >= 0 && player_pos < board_size) {
         current_prop_owner = obs[owners_start_idx + player_pos];
    }

    current_state_tuple.position = player_pos;
    current_state_tuple.money_bin = player_money / 100; // Bin money by 100
    current_state_tuple.current_prop_owner = current_prop_owner;
    current_state_tuple.in_jail = player_in_jail;

    return current_state_tuple;
}


// Select action using epsilon-greedy policy based on Q-values
int select_action_mc(MonteCarloAgent* agent, StateTuple state_tuple, MonopolyEnv* env) {
    // --- Determine if a 'Buy' decision (action 1) is even possible ---
    int p = env->current_player;
    int pos = env->positions[p];
    bool is_buyable = false;

    // Check bounds and if currently in jail (can't buy from jail)
    if (!env->in_jail[p] && pos >= 0 && pos < env->board_size) {
        Property* prop = &env->properties[pos];
        is_buyable = (prop->price > 0 && prop->owner == -1 && env->money[p] >= prop->price);
    }

    // If not on a buyable square, the only logical action is 0 (Pass/Continue)
    if (!is_buyable) {
        return 0;
    }

    // --- If buyable, use Epsilon-Greedy ---
    // Explore with probability epsilon
    if (((double)rand() / RAND_MAX) < agent->epsilon) {
        // Since buy is possible, randomly choose between 0 and 1
        return rand() % 2;
    } else {
        // Exploit: Choose action with highest Q-value
        QTableEntry* entry = find_or_create_q_entry(agent->q_table, state_tuple);
        if (!entry) {
             fprintf(stderr, "Warning: Failed to find/create Q-table entry in select_action. Defaulting to random.\n");
             return rand() % 2; // Fallback if allocation failed
        }

        double q_val_0 = entry->values[0].q_value;
        double q_val_1 = entry->values[1].q_value;

        // Choose the action with the higher Q-value, break ties randomly
        if (fabs(q_val_0 - q_val_1) < 1e-9) { // Floats are equal (or both 0 initially)
            return rand() % 2; // Break tie randomly
        } else if (q_val_1 > q_val_0) {
            return 1; // Buy has higher value
        } else {
            return 0; // Don't Buy has higher value
        }
    }
}

// Generate one episode using the agent's policy
EpisodeHistory generate_episode_mc(MonteCarloAgent* agent, MonopolyEnv* env, int episode_id, LogEntry** out_logs, int* out_log_count) {
    EpisodeHistory history;
    init_episode_history(&history, 100); // Initial capacity 100 steps

    // --- Manage Detailed Logs ---
    #define MAX_LOG_ENTRIES 1000
    static LogEntry log_buffer[MAX_LOG_ENTRIES]; // Static buffer for simplicity
    int log_count = 0;

    int obs_size = get_observation_size(agent->num_players);
    int* obs = (int*)malloc(obs_size * sizeof(int));
    if (!obs) {
        fprintf(stderr, "Error: Failed to allocate observation buffer in generate_episode\n");
        history.count = -1; // Indicate error
        return history;
    }

    reset_monopoly_env(env, obs);
    bool done = false;
    int step_count = 0;

    while (!done && step_count < MAX_EPISODE_STEPS) {
        int current_player = env->current_player; // Who's turn is it?
        StateTuple state_tuple = _get_state_tuple_c(obs, agent->num_players, env->board_size);

        // Agent selects action based on its policy
        int action = select_action_mc(agent, state_tuple, env);

        // Environment processes the turn
        StepResult result = step_monopoly_env(env, action, obs);

        // Store data for MC update *using the state the decision was made in*
        add_episode_step(&history, state_tuple, action, result.reward);

        // Store detailed log
        if (log_count < MAX_LOG_ENTRIES) {
            log_buffer[log_count] = result.log; // Copy log entry
            log_buffer[log_count].episode_id = episode_id; // Add episode ID
            log_count++;
        } else {
             fprintf(stderr, "Warning: Log buffer overflow in episode %d\n", episode_id);
        }

        done = result.done;
        step_count++;
    }

    free(obs);

    // Pass log data back (if requested)
    if (out_logs && out_log_count) {
        *out_logs = log_buffer; // Point to the static buffer
        *out_log_count = log_count;
    } else if (out_logs) {
        *out_logs = NULL;
        *out_log_count = 0;
    }

    return history; // Remember to call free_episode_history on this later
}


// Update Q-values using First-Visit Monte Carlo based on an episode history
void update_mc(MonteCarloAgent* agent, EpisodeHistory* history) {
    double G = 0.0; // Cumulative reward (Return)
    // Create a temporary set to track visited (state, action) pairs for this episode *only*
    VisitedSet* visited_state_actions = create_visited_set(VISITED_SET_INITIAL_SIZE);
    if (!visited_state_actions) {
         fprintf(stderr, "Error: Failed to create visited set for update. Skipping update.\n");
         return;
    }

    // Iterate backwards through the episode
    for (int i = history->count - 1; i >= 0; --i) {
        StateTuple state_tuple = history->steps[i].state;
        int action = history->steps[i].action;
        double reward = history->steps[i].reward;

        G += reward; // Update return G

        VisitedKey current_key = {state_tuple, action};

        // First-visit Monte Carlo check: only update the first time this (s,a) was visited *in this backward pass*
        if (!check_and_add_visited(visited_state_actions, current_key)) {
            // This is the first visit for this (s,a) pair in this episode traverse
            QTableEntry* entry = find_or_create_q_entry(agent->q_table, state_tuple);
            if (!entry) {
                fprintf(stderr, "Warning: Failed to find/create Q-table entry during update. Skipping step.\n");
                continue; // Skip if allocation failed
            }

            // Update the sum of returns and count for the specific action
            entry->values[action].sum_returns += G;
            entry->values[action].count++;

            // Update Q-value as the average of observed returns
            entry->values[action].q_value = entry->values[action].sum_returns / entry->values[action].count;

            // Policy improvement is implicit via epsilon-greedy action selection in the next episode
        }
    }

    // Clean up the temporary visited set for this episode
    destroy_visited_set(visited_state_actions);
}
void export_q_table_to_csv(QHashTable* q_table, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open Q-table CSV file '%s' for writing.\n", filename);
        return;
    }
    // Write header
    fprintf(fp, "position,money_bin,current_prop_owner,in_jail,action,q_value,count\n");
    for (int i = 0; i < q_table->size; ++i) {
        QTableEntry* entry = q_table->table[i];
        while (entry != NULL) {
            for (int action = 0; action < 2; ++action) {
                if (entry->values[action].count > 0) {
                    fprintf(fp, "%d,%d,%d,%d,%d,%.6f,%d\n",
                        entry->key.position,
                        entry->key.money_bin,
                        entry->key.current_prop_owner,
                        entry->key.in_jail,
                        action,
                        entry->values[action].q_value,
                        entry->values[action].count
                    );
                }
            }
            entry = entry->next;
        }
    }
    fclose(fp);
}
static char* escape_csv_string(const char* input) {
    if (!input) return NULL;

    int quotes_to_escape = 0;
    for (const char* p = input; *p; ++p) {
        if (*p == '"') {
            quotes_to_escape++;
        }
    }

    // Allocate space: original length + quotes to escape + 2 for surrounding quotes + 1 for null terminator
    size_t new_len = strlen(input) + quotes_to_escape + 2 + 1;
    char* output = (char*)malloc(new_len);
    if (!output) return NULL;

    char* q = output;
    *q++ = '"'; // Start with a quote

    for (const char* p = input; *p; ++p) {
        if (*p == '"') {
            *q++ = '"'; // Escape internal quote by doubling it
        }
        *q++ = *p; // Copy original character
    }

    *q++ = '"'; // End with a quote
    *q = '\0';  // Null terminate

    return output;
}


// Writes a single LogEntry to the CSV file
static void write_log_to_csv(FILE* fp, const LogEntry* log) {
    if (!fp || !log) return;

    // Escape potentially problematic string fields
    char* escaped_action_desc = escape_csv_string(log->action_desc);
    char* escaped_card_name = escape_csv_string(log->card_drawn);
    char* escaped_card_spec_desc = escape_csv_string(log->card_specific_desc);

    // Handle potential allocation failures during escaping
    if (!escaped_action_desc) escaped_action_desc = strdup("\"\""); // Use empty quoted string if allocation fails
    if (!escaped_card_name) escaped_card_name = strdup("\"\"");
    if (!escaped_card_spec_desc) escaped_card_spec_desc = strdup("\"\"");

    fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%d,%d,%d,%d,%d,%s,%s,%s\n",
            log->episode_id,
            log->step,
            log->player,
            log->position_before,
            log->dice_roll,
            log->landed_on_position,
            log->position_after,
            log->money_before,
            log->money_after,
            log->reward,
            (int)log->done, // bool to int (0 or 1)
            (int)log->in_jail, // bool to int
            log->fee_paid,
            log->agent_action,
            log->num_owned_properties,
            escaped_card_name ? escaped_card_name : "\"\"", // Use "" if escaping failed completely
            escaped_card_spec_desc ? escaped_card_spec_desc : "\"\"",
            escaped_action_desc ? escaped_action_desc : "\"\""
           );

    // Free the allocated escaped strings
    free(escaped_action_desc);
    free(escaped_card_name);
    free(escaped_card_spec_desc);
}

// --- Main Function ---
int main(int argc, char *argv[]) {
    // --- Parameters ---
    int num_players = 2;
    int start_money = 1500;
    int go_reward = 200;
    int num_episodes = 500; // Default number of episodes - MATCH CUDA VERSION
    double epsilon = 0.1;
    const char* csv_filename = "monopoly_training_log_seq.csv"; // Different filename

    // --- Command Line Arguments (Optional) ---
    if (argc > 1) {
        num_episodes = atoi(argv[1]);
        if (num_episodes <= 0) {
            fprintf(stderr, "Warning: Invalid number of episodes specified. Using default %d.\n", 10000);
            num_episodes = 500;
        }
    }
    if (argc > 2) {
        csv_filename = argv[2];
    }

    // --- Initialization ---
    srand(time(NULL)); // Seed random number generator

    printf("Initializing Host Environment...\n");
    MonopolyEnv* env = create_monopoly_env(num_players, start_money, go_reward);
    MonteCarloAgent* agent = create_monte_carlo_agent(num_players, epsilon);

    if (!env || !agent) {
        fprintf(stderr, "Error: Failed to initialize environment or agent.\n");
        destroy_monopoly_env(env);
        destroy_monte_carlo_agent(agent);
        return 1;
    }

    // --- Open CSV File ---
    printf("Opening CSV file '%s'...\n", csv_filename);
    FILE* csv_file = fopen(csv_filename, "w");
    if (!csv_file) {
        fprintf(stderr, "Error: Could not open CSV file '%s' for writing: %s\n", csv_filename, strerror(errno));
        destroy_monopoly_env(env);
        destroy_monte_carlo_agent(agent);
        return 1;
    }

    // --- Write CSV Header ---
    fprintf(csv_file, "episode_id,step,player,position_before,dice_roll,landed_on_position,position_after,money_before,money_after,reward,done,in_jail,fee_paid,agent_action,num_owned_properties,card_drawn,card_specific_desc,action_desc\n");
    fflush(csv_file); // Ensure header is written

    printf("Starting Sequential Monte Carlo Training for %d episodes...\n", num_episodes);

    // --- Training Loop with Timing ---
    clock_t start_time = clock();

    for (int ep = 0; ep < num_episodes; ++ep) {
        LogEntry* episode_logs = NULL;
        int log_count = 0;

        // Generate an episode using the current policy and capture logs
        EpisodeHistory history = generate_episode_mc(agent, env, ep, &episode_logs, &log_count);

        if (history.count < 0) {
            fprintf(stderr, "Error during episode generation %d. Stopping.\n", ep);
            break;
        }

        // Write the logs for this episode to the CSV file
        if (episode_logs != NULL) {
            for (int i = 0; i < log_count; ++i) {
                write_log_to_csv(csv_file, &episode_logs[i]);
            }
        }

        // Update the agent's Q-values based on the episode history
        update_mc(agent, &history);

        // Free the memory used by the episode history struct itself
        free_episode_history(&history);

        // Print progress (less frequently)
        if ((ep + 1) % 5000 == 0 || ep == num_episodes - 1) {
            printf("Episode %d/%d completed. Q-Table size: %d\n", ep + 1, num_episodes, agent->q_table ? agent->q_table->count : 0);
            fflush(csv_file);
        }
    }

    clock_t end_time = clock();
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    printf("\n--- Performance Metrics ---\n");
    printf("CPU Training Time: %.2f milliseconds\n", cpu_time_used * 1000.0);
    printf("Training throughput: %.2f episodes/second\n", num_episodes / cpu_time_used);
    printf("------------------------\n");

    // --- Close CSV File ---
    if (fclose(csv_file) != 0) {
        fprintf(stderr, "Warning: Error closing CSV file '%s': %s\n", csv_filename, strerror(errno));
    } else {
        printf("Log saved to '%s'.\n", csv_filename);
    }

    // --- Optional: Print some learned Q-values ---
    printf("\nExample Q-values (State: Pos, MoneyBin, PropOwner, InJail):\n");
    int print_count = 0;
    if (agent && agent->q_table && agent->q_table->table) {
        for (int i = 0; i < agent->q_table->size && print_count < 20; ++i) {
            QTableEntry* entry = agent->q_table->table[i];
            while (entry != NULL && print_count < 20) {
                StateTuple s = entry->key;
                if (entry->values[0].count > 0 || entry->values[1].count > 0) {
                    printf(" State (%2d, %3d, %2d, %d): Q(Pass)=%8.2f (%5d visits), Q(Buy)=%8.2f (%5d visits)\n",
                           s.position, s.money_bin, s.current_prop_owner, s.in_jail,
                           entry->values[0].q_value, entry->values[0].count,
                           entry->values[1].q_value, entry->values[1].count);
                    print_count++;
                }
                entry = entry->next;
            }
        }
        if (print_count == 0) {
            printf(" No Q-values learned or printed.\n");
        } else {
            printf(" Printed top %d Q-value entries found.\n", print_count);
        }
    } else {
        printf(" Q-Table not available for printing.\n");
    }

    // --- Clean up ---
    printf("\nCleaning up...\n");
    destroy_monte_carlo_agent(agent);
    destroy_monopoly_env(env);
    printf("Done.\n");

    return 0;
}