// Practice content.
// Words avoid J and Z (motion letters, not supported) and T (the dataset's T isn't standard ASL,
// see RESULTS.md), and never mix P and Q (the model's most confused pair on the webcam).
// Checked by scripts/check-words.mjs.

export const LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("");

// Letters shown with a warning in the app
export const LETTER_NOTES = {
  T: "Dataset version, not standard ASL. A real deployment needs T re-recorded with the correct handshape.",
};

// One-line handshape tips for the Learn page, plus the letters each is easily confused with
// (from the model's confusions on the dataset and on my webcam, see RESULTS.md).
export const LETTER_TIPS = {
  A: { tip: "Fist, thumb resting against the side of the index finger.", confusedWith: "S, E" },
  B: { tip: "Flat hand, fingers together pointing up, thumb folded across the palm." },
  C: { tip: "Curve fingers and thumb into a C, like holding a cup.", confusedWith: "O" },
  D: { tip: "Index finger up; the other fingertips touch the thumb, making a circle." },
  E: { tip: "Fingertips bent down to rest on the thumb, which is tucked across the palm.", confusedWith: "S, A" },
  F: { tip: "Thumb and index tips touch in a circle; the other three fingers up and spread." },
  G: { tip: "Index finger and thumb point sideways, parallel; other fingers closed.", confusedWith: "H" },
  H: { tip: "Index and middle fingers together, pointing sideways.", confusedWith: "G" },
  I: { tip: "Pinky up, other fingers in a fist, thumb across them." },
  K: { tip: "Index and middle up in a V, thumb against the middle finger.", confusedWith: "V" },
  L: { tip: "Thumb and index finger form an L." },
  M: { tip: "Thumb tucked under the first three fingers.", confusedWith: "N" },
  N: { tip: "Thumb tucked under the first two fingers.", confusedWith: "M" },
  O: { tip: "All fingertips touch the thumb tip, making an O. If it reads as C, tilt your hand slightly.", confusedWith: "C" },
  P: { tip: "Index pointing forward and down, middle finger touching the thumb.", confusedWith: "Q" },
  Q: { tip: "Like G turned down: index and thumb point at the floor.", confusedWith: "P" },
  R: { tip: "Index and middle fingers crossed." },
  S: { tip: "Fist with the thumb across the front of the fingers.", confusedWith: "E, A" },
  T: { tip: "This app follows the dataset's T (see the note); standard ASL tucks the thumb between index and middle." },
  U: { tip: "Index and middle fingers up, together.", confusedWith: "V, R" },
  V: { tip: "Index and middle fingers up, spread apart.", confusedWith: "K, W" },
  W: { tip: "Index, middle and ring fingers up and spread." },
  X: { tip: "Index finger bent into a hook, other fingers in a fist." },
  Y: { tip: "Thumb and pinky out, other fingers folded." },
};

export const CATEGORIES = [
  {
    id: "alphabet", name: "Alphabet", emoji: "🔤", sample: "ABCD",
    description: "Every letter on its own. Start here.",
    items: LETTERS,
  },
  {
    id: "everyday", name: "Everyday", emoji: "👋", sample: "HELLO",
    items: ["HELLO", "LOVE", "HAND", "SIGN", "BOOK", "HOME", "NAME", "FRIEND", "FAMILY", "WORLD",
            "HAPPY", "SMILE", "SCHOOL", "LEARN", "SPELL", "MUSIC", "DANCE", "CHAIR", "PHONE", "MONEY", "HOUSE"],
  },
  {
    id: "food", name: "Food", emoji: "🍎", sample: "COFFEE",
    items: ["FOOD", "MILK", "CAFE", "APPLE", "BREAD", "FISH", "COFFEE", "RICE", "CAKE", "EGG",
            "CHEESE", "HONEY", "LEMON", "MANGO", "CANDY", "SALAD", "CORN", "BEAN", "SOUP", "PIE", "NOODLE"],
  },
  {
    id: "animals", name: "Animals", emoji: "🐾", sample: "PANDA",
    items: ["DOG", "BIRD", "HORSE", "MOUSE", "LION", "BEAR", "DUCK", "COW", "SHEEP", "SNAKE",
            "FROG", "SEAL", "WOLF", "DEER", "CRAB", "EAGLE", "SWAN", "PANDA", "MONKEY", "CAMEL", "KOALA"],
  },
  {
    id: "colors", name: "Colors", emoji: "🎨", sample: "YELLOW",
    items: ["RED", "BLUE", "GREEN", "YELLOW", "PURPLE", "ORANGE", "BLACK", "PINK", "GRAY", "GOLD",
            "BROWN", "SILVER"],
  },
  {
    id: "nature", name: "Nature", emoji: "🌿", sample: "RAINBOW",
    items: ["SUN", "MOON", "SNOW", "RAIN", "FIRE", "OCEAN", "CLOUD", "RAINBOW", "RIVER", "SKY",
            "LEAF", "ROCK", "SAND", "WIND", "ISLAND", "FLOWER"],
  },
  {
    id: "doubles", name: "Double letters", emoji: "✌️", sample: "BALLOON",
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
