import random
import csv
import uuid
from datetime import datetime, timedelta
import os

# ================= CONFIGURATION =================
TOTAL_POSTS = 120000          
NUM_HUMANS = 4500             # Valid pool of humans
NUM_BOT_COMMUNITIES = 8       # Increased variety
BOTS_PER_COMMUNITY = 40       # 320 Bots total
PLATFORM = "twitter"
START_TIME = datetime(2024, 1, 1, 8, 0, 0)
DURATION_DAYS = 90            # Spread tweets over 3 months
# =================================================

random.seed(42)

# --- 1. DYNAMIC CONTENT GENERATORS ---

# Word banks for constructing sentences
HUMAN_VOCAB = {
    "openers": ["Honestly,", "Just realized", "Can't believe", "Why is", "Finally,", "Thinking about"],
    "verbs": ["eating", "watching", "coding", "reading", "hating", "loving", "buying", "fixing"],
    "nouns": ["coffee", "the weather", "my code", "this movie", "traffic", "ChatGPT", "bitcoin", "the gym"],
    "adjectives": ["amazing", "terrible", "weird", "boring", "great", "annoying", "overrated", "underrated"],
    "hashtags": ["#life", "#random", "#thoughts", "#vibes", "#work", "#weekend", "#mood"]
}

BOT_TEMPLATES = {
    "promo": {
        "templates": [
            "🔥 Flash Sale! Get {pct}% OFF on {item} today! {link}",
            "Don't miss out! {item} is selling fast. Buy here: {link}",
            "Best deal of the year on {item}. Limited time! ⏳"
        ],
        "items": ["SmartWatches", "VPN Services", "Online Courses", "LED Lights", "Supplements"],
        "hashtags": ["#deal", "#sale", "#discount", "#offer"],
        "night_bias": 0.2
    },
    "crypto": {
        "templates": [
            "🚀 {coin} is going to the moon! 100x incoming!",
            "Claim your FREE {coin} airdrop now: {link}",
            "Why is nobody talking about {coin}? Next big thing! 💎🙌"
        ],
        "items": ["$BTC", "$ETH", "$DOGE", "$SHIB", "$SOL", "$PEPE"],
        "hashtags": ["#crypto", "#bullrun", "#altcoins", "#money"],
        "night_bias": 0.8
    },
    "political": {
        "templates": [
            "They are lying to you about {topic}! Wake up!",
            "Breaking: The truth about {topic} revealed.",
            "Retweet if you stand with {topic} against the corruption!"
        ],
        "items": ["the election", "the border bill", "tax reform", "climate policy", "the new law"],
        "hashtags": ["#truth", "#politics", "#news", "#wakeup"],
        "night_bias": 0.4
    },
    "tech_spam": {
        "templates": [
            "Lost access to your account? We can recover it! 🛡️",
            "DM us for instant {item} recovery services.",
            "Hacked? Contact our support team immediately: {link}"
        ],
        "items": ["Instagram", "Facebook", "Wallet", "Email"],
        "hashtags": ["#hacked", "#recovery", "#support", "#help"],
        "night_bias": 0.5
    },
    "nft_hype": {
        "templates": [
            "Sweeping the floor on this collection! 🧹 {link}",
            "GM! Who is holding {item} today? WAGMI!",
            "Minting starts in 1 hour. Don't miss {item}!"
        ],
        "items": ["BoredApes", "PudgyPenguins", "Azuki", "CloneX"],
        "hashtags": ["#NFT", "#Web3", "#ETH", "#Community"],
        "night_bias": 0.6
    },
    "engagement": {
        "templates": [
            "Drop a ❤️ if you agree with this!",
            "Tag a friend who needs to see this 👇",
            "Type 'YES' to claim your blessing today! 🙏"
        ],
        "items": ["motivation", "love", "success"],
        "hashtags": ["#viral", "#explore", "#fyp", "#engage"],
        "night_bias": 0.1
    },
    "news_aggregator": {
        "templates": [
            "🚨 BREAKING: New report released regarding {item}.",
            "Update: Official statement on {item} just published.",
            "Timeline: How the {item} situation unfolded."
        ],
        "items": ["Economy", "Foreign Policy", "Tech Sector", "Health Crisis"],
        "hashtags": ["#breaking", "#news", "#update", "#world"],
        "night_bias": 0.05
    },
    "giveaway": {
        "templates": [
            "🎁 BIG GIVEAWAY! Win a {item} now!",
            "Follow + RT to enter to win {item}! ending soon.",
            "We are giving away 5x {item} to lucky winners!"
        ],
        "items": ["iPhone 15", "PS5", "$500 Cash", "Tesla Model 3"],
        "hashtags": ["#giveaway", "#win", "#free", "#contest"],
        "night_bias": 0.3
    }
}

# --- 2. HELPER FUNCTIONS ---

def generate_human_text():
    """Generates a varied sentence using grammar slots."""
    structure = random.choice([1, 2, 3])
    if structure == 1:
        return f"{random.choice(HUMAN_VOCAB['openers'])} {random.choice(HUMAN_VOCAB['verbs'])} {random.choice(HUMAN_VOCAB['nouns'])}."
    elif structure == 2:
        return f"This {random.choice(HUMAN_VOCAB['nouns'])} is {random.choice(HUMAN_VOCAB['adjectives'])}."
    else:
        return f"Spending the day {random.choice(HUMAN_VOCAB['verbs'])}. {random.choice(HUMAN_VOCAB['hashtags'])}"

def generate_bot_text(community):
    """Generates a template-filled bot sentence."""
    cfg = BOT_TEMPLATES.get(community, BOT_TEMPLATES['promo'])
    template = random.choice(cfg["templates"])
    item = random.choice(cfg["items"])
    # Generate a fake shortened link
    link = f"bit.ly/{uuid.uuid4().hex[:6]}" 
    pct = random.choice([10, 20, 50, 70])
    
    return template.replace("{item}", item).replace("{link}", link).replace("{pct}", str(pct)).replace("{topic}", item)

def random_timestamp(start, days, night_bias=0.2):
    """
    Generates a timestamp. 
    night_bias: Probability of posting between 12 AM - 6 AM.
    """
    random_seconds = random.randint(0, days * 24 * 3600)
    dt = start + timedelta(seconds=random_seconds)
    
    # Adjust hour based on bias (Bot vs Human sleep cycles)
    hour = dt.hour
    is_night = (hour < 6)
    
    # If it's night and we want low night bias (Human), reroll 80% of time
    if is_night and random.random() > night_bias:
        dt += timedelta(hours=random.randint(8, 12))
        
    return dt

# --- 3. INITIALIZATION ---

rows = []
print("Initializing users...")

# Create Humans
human_users = []
for i in range(NUM_HUMANS):
    username = f"user_{random.choice(['cool','real','dev','fan'])}_{i}_{random.randint(10,99)}"
    human_users.append(username)

# Create Bots
bot_users = []
# Ensure we only use communities present in BOT_TEMPLATES
available_communities = list(BOT_TEMPLATES.keys())[:NUM_BOT_COMMUNITIES]

for comm in available_communities:
    for i in range(BOTS_PER_COMMUNITY):
        username = f"{comm[:4]}_bot_{random.randint(1000,9999)}"
        bot_users.append({"username": username, "community": comm})

print(f"Created {len(human_users)} Humans and {len(bot_users)} Bots.")
print(f"Generating {TOTAL_POSTS} posts... (This may take a moment)")

# --- 4. DATA GENERATION LOOP ---

# We'll generate proportionally to hit TOTAL_POSTS
# Let's assume Bots post 3x more frequently than humans on average
weight_bot = 3
weight_human = 1
total_weight = (len(bot_users) * weight_bot) + (len(human_users) * weight_human)

posts_generated = 0

# Pre-calculate counts to speed up loop
bot_post_count = int(TOTAL_POSTS * (len(bot_users) * weight_bot / total_weight))
human_post_count = TOTAL_POSTS - bot_post_count

# -- Generate Human Posts --
for _ in range(human_post_count):
    user = random.choice(human_users)
    ts = random_timestamp(START_TIME, DURATION_DAYS, night_bias=0.05) # Humans sleep at night
    
    text = generate_human_text()
    hashtags = random.choice(HUMAN_VOCAB["hashtags"]) if random.random() < 0.3 else ""
    mentions = f"@{random.choice(human_users)}" if random.random() < 0.2 else ""
    
    rows.append([
        str(uuid.uuid4().int)[:18],  # Fake Tweet ID
        user,
        text,
        ts.isoformat() + "Z",
        "", # Reply ID
        mentions,
        hashtags,
        PLATFORM,
        "human" # Label
    ])

# -- Generate Bot Posts --
for _ in range(bot_post_count):
    bot_obj = random.choice(bot_users)
    user = bot_obj["username"]
    comm = bot_obj["community"]
    bias = BOT_TEMPLATES[comm]["night_bias"]
    
    ts = random_timestamp(START_TIME, DURATION_DAYS, night_bias=bias)
    text = generate_bot_text(comm)
    
    # Bots use many hashtags
    tags = " ".join(random.sample(BOT_TEMPLATES[comm]["hashtags"], k=random.randint(1, 3)))
    
    # Bots mention others to spam
    mentions = ""
    if random.random() < 0.4:
        target = random.choice(human_users) # Bots target humans
        mentions = f"@{target}"

    rows.append([
        str(uuid.uuid4().int)[:18],
        user,
        text,
        ts.isoformat() + "Z",
        "",
        mentions,
        tags,
        PLATFORM,
        comm # Label
    ])

# Sort by time so it looks real
print("Sorting data by timestamp...")
rows.sort(key=lambda x: x[3])

# --- 5. EXPORT ---

file_name = "test_data.csv"
folder_path = r'/Users/vanshgoel/Desktop/nlpr/data/raw'
full_path = os.path.join(folder_path, file_name)
print(f"Writing to {full_path}...")

with open(full_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Added "label" column which is useful for Machine Learning training later
    writer.writerow([
        "post_id", "user_id", "text", "timestamp",
        "reply_to", "mentions", "hashtags", "platform", "label"
    ])
    writer.writerows(rows)

print("Done! Summary:")
print(f"Total Rows: {len(rows)}")
print(f"Bot Posts: {bot_post_count}")
print(f"Human Posts: {human_post_count}")