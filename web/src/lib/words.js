// Practice content.
// Words avoid J and Z (motion letters, not supported) and T (the dataset's T isn't standard ASL,
// see RESULTS.md), and never mix P and Q (the model's most confused pair on the webcam).
// Checked by scripts/check-words.mjs.

export const LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("");

// Letters shown with a warning in the app
export const LETTER_NOTES = {
  T: "Dataset version, not standard ASL. A real deployment needs T re-recorded with the correct handshape.",
};

export const CATEGORIES = [
  {
    id: "alphabet", name: "Alphabet", emoji: "🔤",
    description: "Every letter on its own. Start here.",
    items: LETTERS,
  },
  {
    id: "everyday", name: "Everyday", emoji: "👋",
    items: ["HELLO", "LOVE", "HAND", "SIGN", "BOOK", "HOME", "NAME", "FRIEND", "FAMILY", "WORLD",
            "HAPPY", "SMILE", "SCHOOL", "LEARN", "SPELL", "MUSIC", "DANCE", "CHAIR", "PHONE", "MONEY", "HOUSE"],
  },
  {
    id: "food", name: "Food", emoji: "🍎",
    items: ["FOOD", "MILK", "CAFE", "APPLE", "BREAD", "FISH", "COFFEE", "RICE", "CAKE", "EGG",
            "CHEESE", "HONEY", "LEMON", "MANGO", "CANDY", "SALAD", "CORN", "BEAN", "SOUP", "PIE", "NOODLE"],
  },
  {
    id: "animals", name: "Animals", emoji: "🐾",
    items: ["DOG", "BIRD", "HORSE", "MOUSE", "LION", "BEAR", "DUCK", "COW", "SHEEP", "SNAKE",
            "FROG", "SEAL", "WOLF", "DEER", "CRAB", "EAGLE", "SWAN", "PANDA", "MONKEY", "CAMEL", "KOALA"],
  },
  {
    id: "colors", name: "Colors", emoji: "🎨",
    items: ["RED", "BLUE", "GREEN", "YELLOW", "PURPLE", "ORANGE", "BLACK", "PINK", "GRAY", "GOLD",
            "BROWN", "SILVER"],
  },
  {
    id: "nature", name: "Nature", emoji: "🌿",
    items: ["SUN", "MOON", "SNOW", "RAIN", "FIRE", "OCEAN", "CLOUD", "RAINBOW", "RIVER", "SKY",
            "LEAF", "ROCK", "SAND", "WIND", "ISLAND", "FLOWER"],
  },
  {
    id: "doubles", name: "Double letters", emoji: "✌️",
    description: "Words with a repeated letter: slide your hand sideways for the second one.",
    items: ["HELLO", "BOOK", "GOOD", "HAPPY", "APPLE", "QUEEN", "COFFEE", "SPELL", "BALL", "DOOR",
            "FEEL", "KEEP", "BEE", "BALLOON", "COOKIE", "SHEEP", "GREEN", "YELLOW"],
  },
];

export const categoryById = (id) => CATEGORIES.find((c) => c.id === id);

const shuffle = (arr) => {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};

/** A shuffled round of a category's items; each item once per round. */
export const makeQueue = (category) => shuffle(category.items);
